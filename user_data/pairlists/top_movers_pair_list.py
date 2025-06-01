"""
TopMoversPairList Configuration Documentation

To use the `TopMoversPairList` in your Freqtrade configuration (`config.json`),
add an entry to your `pairlists` array. This pairlist handler selects pairs
based on their percentage price change over a defined lookback period and can
optionally filter them by market capitalization and supported leverage (for Bybit futures).

Example `config.json` entry:

```json
{
  // ... other Freqtrade configurations ...

  "pairlists": [
    {
      "method": "TopMoversPairList", // Must match the class name
      "number_assets": 15,          // Number of top pairs to select
      "refresh_period": 3600,       // How often to refresh the pairlist (in seconds, e.g., 1 hour)

      // Percentage Change Configuration
      "percentage_change_timeframe": "1h",     // Timeframe for candles used in % change calculation (e.g., '5m', '1h', '4h')
      "percentage_change_lookback_candles": 24, // Number of candles to look back for % change (e.g., 24 * 1h candles = 24 hours)
      "sort_direction": "desc",                // "desc" for top gainers, "asc" for top losers

      // Market Cap Filter (Optional, requires 'pycoingecko' library)
      "market_cap_filter": {
        "enabled": true,                       // Set to true to enable this filter
        "min_market_cap_usd": 50000000,        // Minimum market cap in USD (e.g., 50 million). Set to 0 for no minimum.
        "max_market_cap_usd": 10000000000,     // Maximum market cap in USD (e.g., 10 billion). Omit or set to null for no maximum.
        "fetch_historical_for_backtest": false, 
        "use_live_on_backtest_fallback": true,
        "top_n_market_cap_coins": 500 
      },

      // Bybit Leverage Filter (Optional, for Bybit futures trading_mode)
      "bybit_leverage_filter": {
        "enabled": true,                       // Set to true to enable this filter
        "min_leverage": 20                     // Minimum leverage the pair must support on Bybit (e.g., 20x)
      },

      // Logging Configuration
      "selection_log_path": "user_data/logs/top_movers_selection_log.csv" 
    }
  ],
}
```
"""
# -*- coding: utf-8 -*-
import logging
from typing import Any, Dict, List, Optional, Tuple
import csv
import os
from datetime import datetime, timezone
import time 
import copy # For deepcopying cached lists

import pandas as pd
from pandas import DataFrame
try:
    from pycoingecko import CoinGeckoAPI
except ImportError:
    CoinGeckoAPI = None

from freqtrade.constants import Config, ListPairsWithTimeframes
from freqtrade.enums import CandleType, RunMode
from freqtrade.exchange import Exchange
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting

logger = logging.getLogger(__name__)

CG_MAX_PER_PAGE = 250

class TopMoversPairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.YES

    def __init__(self, exchange: Exchange, pairlistmanager,
                 config: Config, pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._number_assets = pairlistconfig.get('number_assets', 10)
        self._pct_timeframe = pairlistconfig.get('percentage_change_timeframe', '1h')
        self._pct_lookback = pairlistconfig.get('percentage_change_lookback_candles', 24)
        self._sort_direction = pairlistconfig.get('sort_direction', 'desc').lower()

        mc_filter_config = pairlistconfig.get('market_cap_filter', {})
        self._market_cap_enabled = mc_filter_config.get('enabled', False) and CoinGeckoAPI is not None
        self._min_market_cap_usd = mc_filter_config.get('min_market_cap_usd', 0)
        self._max_market_cap_usd = mc_filter_config.get('max_market_cap_usd')
        self._fetch_historical_mc_backtest = mc_filter_config.get('fetch_historical_for_backtest', False)
        self._use_live_mc_backtest_fallback = mc_filter_config.get('use_live_on_backtest_fallback', True)
        self._top_n_market_cap_coins = mc_filter_config.get('top_n_market_cap_coins', 500)

        if self._market_cap_enabled and CoinGeckoAPI is None:
            logger.warning("Market cap filter enabled, but 'pycoingecko' not installed. Disabling.")
            self._market_cap_enabled = False
        
        self._cg = CoinGeckoAPI() if self._market_cap_enabled else None
        
        self._historical_market_caps_cache: Dict[Tuple[str, str], Optional[float]] = {} 
        self._live_fallback_market_caps_data: Dict[str, float] = {} 
        self._live_fallback_symbol_to_id_map: Dict[str, str] = {} 
        self._coingecko_coins_list_cache: Optional[List[Dict]] = None 

        # Caches for results of initial filtering steps (within a single Freqtrade run)
        self._cached_initial_candidate_pairs_data: Optional[List[Dict[str, Any]]] = None
        self._cached_mc_filtered_pairs_data: Optional[List[Dict[str, Any]]] = None
        self._cached_leverage_filtered_pairs_data: Optional[List[Dict[str, Any]]] = None
        self._dp_max_date_cache_validity: Optional[datetime] = None # Stores the dp max_date for which caches are valid


        if self._market_cap_enabled and self._config['runmode'] == RunMode.BACKTEST and self._use_live_mc_backtest_fallback:
            self._fetch_and_cache_top_market_caps() # Populates live fallback caches

        self._leverage_filter_config = pairlistconfig.get('bybit_leverage_filter', {})
        self._leverage_filter_enabled = self._leverage_filter_config.get('enabled', False)
        self._min_leverage = self._leverage_filter_config.get('min_leverage', 1.0)

        self._selection_log_path = pairlistconfig.get('selection_log_path', 
                                                      'user_data/logs/top_movers_selection_log.csv')

        if self._sort_direction not in ['asc', 'desc']:
            raise ValueError("sort_direction must be 'asc' or 'desc'")

    def _fetch_and_cache_top_market_caps(self):
        # ... (this method remains the same) ...
        if not self._cg or not self._market_cap_enabled:
            return

        logger.info(f"Attempting to fetch and cache top {self._top_n_market_cap_coins} market cap coins for live fallback...")
        self._live_fallback_market_caps_data = {}
        self._live_fallback_symbol_to_id_map = {}
        
        num_pages = (self._top_n_market_cap_coins + CG_MAX_PER_PAGE - 1) // CG_MAX_PER_PAGE 

        for page in range(1, num_pages + 1):
            try:
                logger.info(f"Fetching page {page}/{num_pages} of top market cap coins from CoinGecko...")
                markets = self._cg.get_coins_markets(
                    vs_currency='usd', 
                    order='market_cap_desc', 
                    per_page=min(self._top_n_market_cap_coins - (page-1)*CG_MAX_PER_PAGE, CG_MAX_PER_PAGE), 
                    page=page, 
                    sparkline=False,
                    price_change_percentage='false'
                )
                if markets:
                    for coin_data in markets:
                        asset_id = coin_data.get('id')
                        symbol = coin_data.get('symbol','').lower()
                        market_cap = coin_data.get('market_cap')
                        if asset_id and symbol and market_cap is not None:
                            self._live_fallback_market_caps_data[asset_id] = float(market_cap)
                            self._live_fallback_symbol_to_id_map[symbol] = asset_id
                else:
                    logger.warning(f"Received no data from CoinGecko for page {page}.")
                
                if page < num_pages: 
                    time.sleep(1) 

            except Exception as e:
                logger.error(f"Error fetching top market cap coins (page {page}) from CoinGecko: {e}")
                break 
        logger.info(f"Cached live market cap data for {len(self._live_fallback_market_caps_data)} assets "
                    f"and {len(self._live_fallback_symbol_to_id_map)} symbols.")


    @property
    def short_desc(self) -> str:
        return (f"{self.name} - Top {self._number_assets} by % change "
                f"({self._pct_lookback}x{self._pct_timeframe} candles, {self._sort_direction}).")

    @property 
    def needstickers(self) -> bool:
        return False

    # ... (description, available_parameters, _ensure_dir, _log_selection_to_csv remain the same) ...
    # ... (_get_coingecko_coins_list, _get_asset_id_from_symbol, _get_market_cap_usd, _get_historical_market_cap_usd remain same) ...
    # ... (_calculate_single_pair_percentage_change remains same) ...

    @staticmethod
    def description() -> str:
        return "Selects pairs based on top percentage movers with market cap and leverage filters."

    @staticmethod
    def available_parameters() -> Dict[str, PairlistParameter]:
        return {
            "number_assets": {
                "type": "number", "default": 10, "description": "Number of assets to return",
                "help": "Number of assets to return", "category": "filter"
            },
            "percentage_change_timeframe": {
                "type": "string", "default": "1h", "description": "Candle timeframe for % change calc",
                "help": "E.g., '5m', '1h', '4h', '1d'", "category": "filter"
            },
            "percentage_change_lookback_candles": {
                "type": "number", "default": 24, "description": "Number of candles for % change lookback",
                "help": "Number of 'percentage_change_timeframe' candles to look back", "category": "filter"
            },
            "sort_direction": {
                "type": "string", "default": "desc", "description": "Sort: 'desc' for gainers, 'asc' for losers",
                "help": "'desc' for top gainers, 'asc' for top losers", "category": "filter"
            },
            "market_cap_filter.enabled": {
                "type": "boolean", "default": False, "description": "Enable Market Cap Filter (requires pycoingecko)",
                "help": "Set to true to enable filtering by market cap.", "category": "filter"
            },
            "market_cap_filter.min_market_cap_usd": {
                "type": "number", "default": 0, "description": "Minimum Market Cap in USD",
                "help": "Minimum market cap in USD. 0 for no minimum.", "category": "filter"
            },
            "market_cap_filter.max_market_cap_usd": {
                "type": "number", "default": None, "description": "Maximum Market Cap in USD (optional)",
                "help": "Maximum market cap in USD. Leave empty for no maximum.", "category": "filter"
            },
            "market_cap_filter.fetch_historical_for_backtest": {
                "type": "boolean", "default": False, "description": "Fetch daily historical MC for backtests (CoinGecko)",
                "help": "If true, attempts to fetch daily historical market cap from CoinGecko during backtests.", "category": "filter"
            },
            "market_cap_filter.use_live_on_backtest_fallback": {
                "type": "boolean", "default": True, "description": "Use current live MC in backtest if historical fetch is off/fails",
                "help": "If historical fetch is off/disabled, use current live MC (cached once per asset) for backtest.", "category": "filter"
            },
            "market_cap_filter.top_n_market_cap_coins": {
                "type": "number", "default": 500, "description": "Number of top coins to fetch for live MC fallback cache",
                "help": "For backtest live fallback, how many top coins by MC to pre-fetch from CoinGecko.", "category": "filter"
            },
            "bybit_leverage_filter.enabled": {
                "type": "boolean", "default": False, "description": "Enable Bybit Leverage Filter",
                "help": "Set to true to enable filtering by minimum leverage on Bybit futures.", "category": "filter"
            },
            "bybit_leverage_filter.min_leverage": {
                "type": "number", "default": 1.0, "description": "Minimum required leverage",
                "help": "Minimum leverage a Bybit futures pair must support.", "category": "filter"
            },
            "selection_log_path": {
                "type": "string", "default": "user_data/logs/top_movers_selection_log.csv",
                "description": "Path to CSV file for logging selected pairs",
                "help": "Path relative to Freqtrade root, e.g., user_data/logs/top_movers_log.csv", "category": "output"
            }
        }

    def _ensure_dir(self, file_path: str):
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
            except OSError as e:
                logger.error(f"Could not create directory {directory}: {e}")

    def _log_selection_to_csv(self, selected_pairs_details: List[Dict[str, Any]]):
        if not self._selection_log_path or not selected_pairs_details:
            return
        self._ensure_dir(self._selection_log_path)
        file_exists = os.path.isfile(self._selection_log_path)
        write_header = not file_exists or os.path.getsize(self._selection_log_path) == 0
        current_ts = datetime.now(timezone.utc).isoformat()
        fieldnames = ['timestamp', 'rank', 'pair', 'percentage_change', 'market_cap_usd', 'max_leverage']
        try:
            with open(self._selection_log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                if write_header:
                    writer.writeheader()
                for i, p_data in enumerate(selected_pairs_details):
                    writer.writerow({
                        'timestamp': current_ts, 'rank': i + 1, 'pair': p_data.get('pair'),
                        'percentage_change': f"{p_data.get('pct_change', 0.0):.4f}",
                        'market_cap_usd': p_data.get('market_cap_usd'),
                        'max_leverage': p_data.get('max_leverage')
                    })
            logger.debug(f"Logged {len(selected_pairs_details)} pairs to {self._selection_log_path}")
        except IOError as e:
            logger.error(f"Error writing selection log to {self._selection_log_path}: {e}")

    def _get_coingecko_coins_list(self) -> List[Dict]:
        if self._coingecko_coins_list_cache is None and self._cg:
            try:
                logger.info("Fetching full coin list from CoinGecko for symbol/ID mapping...")
                self._coingecko_coins_list_cache = self._cg.get_coins_list(include_platform=False) or []
                logger.info(f"Cached {len(self._coingecko_coins_list_cache)} coins from CoinGecko list.")
            except Exception as e:
                logger.error(f"Error fetching coin list from CoinGecko: {e}")
                self._coingecko_coins_list_cache = [] 
        return self._coingecko_coins_list_cache or []


    def _get_asset_id_from_symbol(self, symbol: str) -> Optional[str]:
        if not self._cg: return None
        
        symbol_lower = symbol.lower()
        if self._config['runmode'] == RunMode.BACKTEST and self._use_live_mc_backtest_fallback:
            if symbol_lower in self._live_fallback_symbol_to_id_map:
                return self._live_fallback_symbol_to_id_map[symbol_lower]
            
        coins_list = self._get_coingecko_coins_list()
        if not coins_list: 
             logger.debug(f"CoinGecko coin list is empty or unavailable for symbol {symbol}.")
             return None

        for coin in coins_list:
            if coin['symbol'].lower() == symbol_lower:
                return coin['id']
        
        logger.debug(f"Could not find CoinGecko ID for symbol: {symbol}")
        return None

    def _get_market_cap_usd(self, asset_id: str, in_backtest_mode: bool) -> Optional[float]:
        if not self._cg: return None

        if in_backtest_mode and self._use_live_mc_backtest_fallback:
            if asset_id in self._live_fallback_market_caps_data:
                return self._live_fallback_market_caps_data[asset_id]
            else:
                logger.debug(f"Asset ID {asset_id} not found in pre-fetched top N market cap data for live fallback.")
                return None 
        
        try:
            logger.debug(f"Fetching live market cap for {asset_id} from CoinGecko (individual call for live/dry run)...")
            market_data = self._cg.get_coin_by_id(id=asset_id, localization='false', tickers='false',
                                                 market_data='true', community_data='false',
                                                 developer_data='false', sparkline='false')
            return market_data.get('market_data', {}).get('market_cap', {}).get('usd')
        except Exception as e:
            logger.warning(f"Error fetching live market cap for {asset_id} from CoinGecko (individual call): {e}")
        return None

    def _get_historical_market_cap_usd(self, asset_id: str, date_obj: datetime) -> Optional[float]:
        if not self._cg or not self._fetch_historical_mc_backtest: return None
        date_str_coingecko = date_obj.strftime('%d-%m-%Y')
        cache_key = (asset_id, date_str_coingecko)
        if cache_key in self._historical_market_caps_cache:
            return self._historical_market_caps_cache[cache_key]
        try:
            logger.debug(f"Fetching historical MC for {asset_id} on {date_str_coingecko}...")
            history = self._cg.get_coin_history_by_id(id=asset_id, date=date_str_coingecko, localization='false')
            market_cap = history.get('market_data', {}).get('market_cap', {}).get('usd')
            if market_cap is None:
                logger.debug(f"CoinGecko: No market cap data for {asset_id} on {date_str_coingecko}.")
            self._historical_market_caps_cache[cache_key] = market_cap
            return market_cap
        except Exception as e:
            logger.warning(f"Error fetching historical MC for {asset_id} on {date_str_coingecko}: {e}")
            self._historical_market_caps_cache[cache_key] = None
            return None

    def _filter_by_market_cap(self, pairs_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._market_cap_enabled: return pairs_data
        if not self._cg:
            logger.warning("Market cap filter enabled, but CoinGeckoAPI not available.")
            return pairs_data

        logger.info("Applying Market Cap Filter...")
        filtered_list = []
        in_backtest_mode = self._config['runmode'] == RunMode.BACKTEST
        
        current_candle_date_for_mc: Optional[datetime] = None 
        if in_backtest_mode and self._fetch_historical_mc_backtest:
            latest_date_found: Optional[pd.Timestamp] = None
            if self._pairlistmanager and hasattr(self._pairlistmanager, '_dataprovider') and \
               self._pairlistmanager._dataprovider and \
               hasattr(self._pairlistmanager._dataprovider, '_data') and \
               self._pairlistmanager._dataprovider._data: 
                for df_pair_key in self._pairlistmanager._dataprovider._data: 
                    df_pair = self._pairlistmanager._dataprovider._data[df_pair_key] 
                    if df_pair is not None and not df_pair.empty and 'date' in df_pair.columns:
                        last_date_in_df = df_pair.iloc[-1]['date']
                        if latest_date_found is None or last_date_in_df > latest_date_found:
                            latest_date_found = last_date_in_df
            if latest_date_found:
                current_candle_date_for_mc = pd.to_datetime(latest_date_found, utc=True).to_pydatetime()
            else: 
                logger.warning("Could not determine current backtest date for HISTORICAL market cap.")

        for p_data in pairs_data: 
            pair_str = p_data['pair']
            base_currency_symbol = self._exchange.get_pair_base_currency(pair_str)
            asset_id = self._get_asset_id_from_symbol(base_currency_symbol)

            if not asset_id:
                logger.debug(f"No CoinGecko ID for {base_currency_symbol} ({pair_str}). Skipping MC check.")
                if self._min_market_cap_usd == 0 : filtered_list.append(p_data)
                continue 
            
            market_cap_usd: Optional[float] = None
            if in_backtest_mode:
                if self._use_live_mc_backtest_fallback: 
                    market_cap_usd = self._live_fallback_market_caps_data.get(asset_id)
                    if market_cap_usd is None:
                         logger.debug(f"Asset ID {asset_id} ({base_currency_symbol}) not in pre-fetched top N MC data.")
                elif self._fetch_historical_mc_backtest and current_candle_date_for_mc: 
                    market_cap_usd = self._get_historical_market_cap_usd(asset_id, current_candle_date_for_mc)
                else: 
                    if self._min_market_cap_usd == 0 : filtered_list.append(p_data) 
                    continue
            else: 
                market_cap_usd = self._get_market_cap_usd(asset_id, in_backtest_mode=False)

            if market_cap_usd is None: 
                if self._min_market_cap_usd == 0:
                    p_data['market_cap_usd'] = None 
                    filtered_list.append(p_data)
                continue 
            
            p_data['market_cap_usd'] = market_cap_usd
            if self._min_market_cap_usd > 0 and market_cap_usd < self._min_market_cap_usd:
                continue
            if self._max_market_cap_usd is not None and market_cap_usd > self._max_market_cap_usd:
                continue
            filtered_list.append(p_data)
            
        logger.info(f"Market Cap Filter: {len(filtered_list)} pairs left from {len(pairs_data)}.")
        return filtered_list

    def _filter_by_leverage(self, pairs_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._leverage_filter_enabled: return pairs_data
        
        if self._config['runmode'] == RunMode.BACKTEST:
            logger.debug("Bybit leverage filter: In backtest mode, using current market leverage data.")

        if not (self._exchange.name.lower() == 'bybit' and 
                self._config.get('trading_mode', 'spot') == 'futures'):
            return pairs_data

        logger.info("Applying Bybit Leverage Filter...")
        filtered_list = []
        markets = self._exchange.markets 
        if not markets: 
            logger.warning("Markets not available for leverage filter.")
            return pairs_data 

        for p_data in pairs_data: 
            pair_str = p_data['pair']
            market = markets.get(pair_str)
            if not market: 
                filtered_list.append(p_data) 
                continue
            
            max_leverage = None
            if market.get('info') and isinstance(market['info'], dict):
                leverage_filter_info = market['info'].get('leverage_filter')
                if leverage_filter_info and 'max_leverage' in leverage_filter_info:
                    try: max_leverage = float(leverage_filter_info['max_leverage'])
                    except ValueError: pass
                elif 'maxLeverage' in market['info']: 
                     try: max_leverage = float(market['info']['maxLeverage'])
                     except ValueError: pass

            if max_leverage is None and market.get('limits', {}).get('leverage', {}).get('max') is not None:
                 try: max_leverage = float(market['limits']['leverage']['max'])
                 except ValueError: pass
            
            p_data['max_leverage'] = max_leverage 
            if max_leverage is None:
                filtered_list.append(p_data)
                continue

            if max_leverage < self._min_leverage:
                continue
            
            filtered_list.append(p_data)
        logger.info(f"Bybit Leverage Filter: {len(filtered_list)} pairs left from {len(pairs_data)}.")
        return filtered_list

    def _calculate_single_pair_percentage_change(self, pair: str) -> Optional[float]:
        candles_df: Optional[DataFrame] = None
        try:
            num_candles_needed = self._pct_lookback + 1
            dp = self._pairlistmanager._dataprovider if (self._pairlistmanager and hasattr(self._pairlistmanager, '_dataprovider')) else None

            if dp is None: return None

            candle_type_str = self._config.get('candle_type_def')
            if not candle_type_str: 
                candle_type_str = CandleType.FUTURES.value if self._config.get('trading_mode') == 'futures' else CandleType.SPOT.value
            
            if self._config['runmode'] in (RunMode.LIVE, RunMode.DRY_RUN):
                candles_df = dp.ohlcv(pair, self._pct_timeframe, candle_type=candle_type_str, copy=False) if dp else None
            elif self._config['runmode'] == RunMode.BACKTEST:
                candles_df = dp.get_pair_dataframe(pair, self._pct_timeframe, candle_type=candle_type_str) if dp else None
            else: return None
            
            if candles_df is None or len(candles_df) < num_candles_needed: return None
            
            candles_df = candles_df.tail(num_candles_needed)
            if len(candles_df) < num_candles_needed: return None

            close_earliest = candles_df['close'].iloc[0]
            close_latest = candles_df['close'].iloc[-1]
            if close_earliest == 0: return 0.0
            return ((close_latest - close_earliest) / close_earliest) * 100
        except Exception as e:
            logger.warning(f"Could not calculate pct change for {pair} ({self._pct_timeframe}): {type(e).__name__}, {str(e)}")
            import traceback
            logger.error(f"Full traceback for error calculating pct change for {pair}:\n{traceback.format_exc()}")
            return None

    def gen_pairlist(self, tickers: Dict[str, Any], cached_pairlist: Optional[List[str]] = None) -> List[str]:
        stake_currency = self._config['stake_currency']

        # Invalidate internal step-caches if DataProvider's max date has changed
        # This is crucial for backtesting's _precalculate_pairlist_timeline loop
        dp = self._pairlistmanager._dataprovider if (self._pairlistmanager and hasattr(self._pairlistmanager, '_dataprovider')) else None
        current_dp_max_date: Optional[datetime] = None
        if dp and hasattr(dp, '_dataframe_max_date'): # Check if _dataframe_max_date attribute exists
            current_dp_max_date = dp._dataframe_max_date

        if self._dp_max_date_cache_validity != current_dp_max_date:
            logger.info(
                f"DataProvider max date changed (or first run with new date context) "
                f"from '{self._dp_max_date_cache_validity}' to '{current_dp_max_date}'. "
                f"Invalidating TopMoversPairList internal step caches."
            )
            self._cached_initial_candidate_pairs_data = None
            self._cached_mc_filtered_pairs_data = None
            self._cached_leverage_filtered_pairs_data = None
            self._dp_max_date_cache_validity = current_dp_max_date
        
        # Use cached pre-filtered lists if available (for subsequent calls in _precalculate_pairlist_timeline)
        if self._cached_leverage_filtered_pairs_data is not None:
            candidate_pairs_data = copy.deepcopy(self._cached_leverage_filtered_pairs_data)
            logger.info(f"Using cached leverage-filtered list of {len(candidate_pairs_data)} pairs.")
        elif self._cached_mc_filtered_pairs_data is not None and self._leverage_filter_enabled:
            # Leverage filter not run yet, or was disabled, but MC was.
            temp_candidate_data = copy.deepcopy(self._cached_mc_filtered_pairs_data)
            candidate_pairs_data = self._filter_by_leverage(temp_candidate_data)
            self._cached_leverage_filtered_pairs_data = copy.deepcopy(candidate_pairs_data) # Cache it now
            logger.info(f"{len(candidate_pairs_data)} pairs remaining after Leverage filter (applied to cached MC list).")
            if not candidate_pairs_data: return []
        elif self._cached_initial_candidate_pairs_data is not None:
            # Only initial candidates cached, run both filters
            temp_candidate_data = copy.deepcopy(self._cached_initial_candidate_pairs_data)
            if self._market_cap_enabled:
                temp_candidate_data = self._filter_by_market_cap(temp_candidate_data)
                self._cached_mc_filtered_pairs_data = copy.deepcopy(temp_candidate_data)
            if self._leverage_filter_enabled:
                temp_candidate_data = self._filter_by_leverage(temp_candidate_data)
                self._cached_leverage_filtered_pairs_data = copy.deepcopy(temp_candidate_data)
            candidate_pairs_data = temp_candidate_data
            logger.info(f"{len(candidate_pairs_data)} pairs after initial filtering sequence.")
            if not candidate_pairs_data: return []
        else: # First run, no caches yet
            initial_candidate_pairs_str = list(self._exchange.get_markets(quote_currencies=[stake_currency], active_only=True).keys())
            if not initial_candidate_pairs_str:
                logger.warning(f"No pairs for stake {stake_currency}."); return []
            logger.info(f"{len(initial_candidate_pairs_str)} initial candidate pairs for {stake_currency}.")
            
            candidate_pairs_data = [{'pair': pair_str} for pair_str in initial_candidate_pairs_str]
            self._cached_initial_candidate_pairs_data = copy.deepcopy(candidate_pairs_data) # Cache initial

            if self._market_cap_enabled:
                candidate_pairs_data = self._filter_by_market_cap(candidate_pairs_data)
                self._cached_mc_filtered_pairs_data = copy.deepcopy(candidate_pairs_data) # Cache after MC
                logger.info(f"{len(candidate_pairs_data)} pairs remaining after Market Cap filter.")
                if not candidate_pairs_data: return []
            
            if self._leverage_filter_enabled:
                candidate_pairs_data = self._filter_by_leverage(candidate_pairs_data)
                self._cached_leverage_filtered_pairs_data = copy.deepcopy(candidate_pairs_data) # Cache after Leverage
                logger.info(f"{len(candidate_pairs_data)} pairs remaining after Leverage filter.")
                if not candidate_pairs_data: return []
        
        logger.info(f"Calculating % change for {len(candidate_pairs_data)} pre-filtered pairs...")
        
        pairs_with_pct_change: List[Dict[str, Any]] = []
        for p_data_dict in candidate_pairs_data: 
            pair_str = p_data_dict['pair']
            pct_change = self._calculate_single_pair_percentage_change(pair_str)
            if pct_change is not None:
                p_data_dict['pct_change'] = pct_change 
                pairs_with_pct_change.append(p_data_dict)

        if not pairs_with_pct_change:
            logger.warning("No pairs with valid pct change data found after calculations on pre-filtered list.")
            return []
        
        try:
            sorted_pairs = sorted(pairs_with_pct_change, key=lambda x: x['pct_change'], reverse=(self._sort_direction == 'desc'))
        except KeyError as e: 
            logger.error(f"KeyError 'pct_change' during sorting. Data: {pairs_with_pct_change}. Error: {e}")
            return []
        except TypeError as e: 
            logger.error(f"TypeError during sorting pairs: {e}. Data: {pairs_with_pct_change}")
            return []

        top_n_pairs_data = sorted_pairs[:self._number_assets]
        final_pairs_list = [p_data['pair'] for p_data in top_n_pairs_data]

        self._log_selection_to_csv(top_n_pairs_data)
        
        log_msg_details = []
        for p_data_log_item in top_n_pairs_data: 
            detail = f"{p_data_log_item['pair']} ({p_data_log_item.get('pct_change', 0.0):.2f}%)"
            if self._market_cap_enabled and 'market_cap_usd' in p_data_log_item and p_data_log_item['market_cap_usd'] is not None:
                detail += f" (MCap: ${p_data_log_item['market_cap_usd']:,.0f})"
            if self._leverage_filter_enabled and 'max_leverage' in p_data_log_item and p_data_log_item['max_leverage'] is not None:
                detail += f" (Lev: {p_data_log_item['max_leverage']}x)"
            log_msg_details.append(detail)
        logger.info(f"Selected Top {len(final_pairs_list)} pairs: {'; '.join(log_msg_details)}")
        
        return final_pairs_list