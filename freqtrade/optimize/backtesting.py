# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""

import logging
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import ccxt # Add import for ccxt
from ccxt.base.errors import ExchangeError # Import ExchangeError directly
import json
import hashlib
from pathlib import Path

from numpy import isnan, nan
from pandas import DataFrame, Series

from freqtrade import constants
from freqtrade.configuration import TimeRange, validate_config_consistency
from freqtrade.constants import DATETIME_PRINT_FORMAT, Config, IntOrInf, LongShort
from freqtrade.data import history
from freqtrade.data.btanalysis import (
    find_existing_backtest_stats,
    get_tick_size_over_time,
    trade_list_to_dataframe,
)
from freqtrade.data.converter import trim_dataframe, trim_dataframes
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.metrics import combined_dataframes_with_rel_mean
from freqtrade.enums import (
    BacktestState,
    CandleType,
    ExitCheckTuple,
    ExitType,
    MarginMode,
    RunMode,
    TradingMode,
)
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.exchange import (
    amount_to_contract_precision,
    price_to_precision,
    timeframe_to_seconds,
)
from freqtrade.exchange.exchange import TICK_SIZE, Exchange
from freqtrade.ft_types import (
    BacktestContentType,
    BacktestContentTypeIcomplete,
    BacktestResultType,
    get_BacktestResultType_default,
)
from freqtrade.leverage.liquidation_price import update_liquidation_prices
from freqtrade.mixins import LoggingMixin
from freqtrade.optimize.backtest_caching import get_strategy_run_id
from freqtrade.optimize.bt_progress import BTProgress
from freqtrade.optimize.optimize_reports import (
    generate_backtest_stats,
    generate_rejected_signals,
    generate_trade_signal_candles,
    show_backtest_results,
    store_backtest_results,
)
from freqtrade.persistence import (
    CustomDataWrapper,
    LocalTrade,
    Order,
    PairLocks,
    Trade,
    disable_database_use,
    enable_database_use,
)
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.plugins.protectionmanager import ProtectionManager
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.util import FtPrecise, dt_now
from freqtrade.util.migrations import migrate_data
from freqtrade.wallets import Wallets


logger = logging.getLogger(__name__)

PAIRLIST_TIMELINE_CACHE_DIR = "pairlist_cache"

# Indexes for backtest tuples
DATE_IDX = 0
OPEN_IDX = 1
HIGH_IDX = 2
LOW_IDX = 3
CLOSE_IDX = 4
LONG_IDX = 5
ELONG_IDX = 6  # Exit long
SHORT_IDX = 7
ESHORT_IDX = 8  # Exit short
ENTER_TAG_IDX = 9
EXIT_TAG_IDX = 10

# Every change to this headers list must evaluate further usages of the resulting tuple
# and eventually change the constants for indexes at the top
HEADERS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "enter_long",
    "exit_long",
    "enter_short",
    "exit_short",
    "enter_tag",
    "exit_tag",
]


class Backtesting:
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """

    def __init__(self, config: Config, exchange: Exchange | None = None) -> None:
        LoggingMixin.show_output = False
        self.config = config
        self.results: BacktestResultType = get_BacktestResultType_default()
        self.trade_id_counter: int = 0
        self.order_id_counter: int = 0

        config["dry_run"] = True
        self.price_pair_prec: dict[str, Series] = {}
        self.run_ids: dict[str, str] = {}
        self.strategylist: list[IStrategy] = []
        self.all_bt_content: dict[str, BacktestContentType] = {}
        self.analysis_results: dict[str, dict[str, DataFrame]] = {
            "signals": {},
            "rejected": {},
            "exited": {},
        }
        self.rejected_dict: dict[str, list] = {}
        self.pairlist_timeline: Dict[datetime, List[str]] = {}
        self.precalculation_interval_str: str = self.config.get("pairlist_precalc_interval", "1d")


        self._exchange_name = self.config["exchange"]["name"]
        if not exchange:
            exchange = ExchangeResolver.load_exchange(self.config, load_leverage_tiers=True)
        self.exchange = exchange

        self.dataprovider = DataProvider(self.config, self.exchange)

        if self.config.get("strategy_list"):
            if self.config.get("freqai", {}).get("enabled", False):
                logger.warning(
                    "Using --strategy-list with FreqAI REQUIRES all strategies "
                    "to have identical feature_engineering_* functions."
                )
            for strat in list(self.config["strategy_list"]):
                stratconf = deepcopy(self.config)
                stratconf["strategy"] = strat
                self.strategylist.append(StrategyResolver.load_strategy(stratconf))
                validate_config_consistency(stratconf)

        else:
            # No strategy list specified, only one strategy
            self.strategylist.append(StrategyResolver.load_strategy(self.config))
            validate_config_consistency(self.config)

        if "timeframe" not in self.config:
            raise OperationalException(
                "Timeframe needs to be set in either "
                "configuration or as cli argument `--timeframe 5m`"
            )
        self.timeframe = str(self.config.get("timeframe"))
        self.timeframe_secs = timeframe_to_seconds(self.timeframe)
        self.timeframe_min = self.timeframe_secs // 60
        self.timeframe_td = timedelta(seconds=self.timeframe_secs)
        self.disable_database_use()
        self.init_backtest_detail()
        self.pairlists = PairListManager(self.exchange, self.config, self.dataprovider)
        self._validate_pairlists_for_backtesting()

        self.dataprovider.add_pairlisthandler(self.pairlists)
        self.pairlists.refresh_pairlist()

        if len(self.pairlists.whitelist) == 0:
            raise OperationalException("No pair in whitelist.")

        if config.get("fee", None) is not None:
            self.fee = config["fee"]
            logger.info(f"Using fee {self.fee:.4%} from config.")
        else:
            fees = [
                self.exchange.get_fee(
                    symbol=self.pairlists.whitelist[0],
                    taker_or_maker=mt,  # type: ignore
                )
                for mt in ("taker", "maker")
            ]
            self.fee = max(fee for fee in fees if fee is not None)
            logger.info(f"Using fee {self.fee:.4%} - worst case fee from exchange (lowest tier).")
        self.precision_mode = self.exchange.precisionMode
        self.precision_mode_price = self.exchange.precision_mode_price

        if self.config.get("freqai_backtest_live_models", False):
            from freqtrade.freqai.utils import get_timerange_backtest_live_models

            self.config["timerange"] = get_timerange_backtest_live_models(self.config)

        self.timerange = TimeRange.parse_timerange(
            None if self.config.get("timerange") is None else str(self.config.get("timerange"))
        )

        # Get maximum required startup period
        self.required_startup = max([strat.startup_candle_count for strat in self.strategylist])
        self.exchange.validate_required_startup_candles(self.required_startup, self.timeframe)

        # Add maximum startup candle count to configuration for informative pairs support
        self.config["startup_candle_count"] = self.required_startup

        if self.config.get("freqai", {}).get("enabled", False):
            # For FreqAI, increase the required_startup to includes the training data
            # This value should NOT be written to startup_candle_count
            self.required_startup = self.dataprovider.get_required_startup(self.timeframe)

        self.trading_mode: TradingMode = config.get("trading_mode", TradingMode.SPOT)
        self.margin_mode: MarginMode = config.get("margin_mode", MarginMode.ISOLATED)
        # strategies which define "can_short=True" will fail to load in Spot mode.
        self._can_short = self.trading_mode != TradingMode.SPOT
        self._position_stacking: bool = self.config.get("position_stacking", False)
        self.enable_protections: bool = self.config.get("enable_protections", False)
        migrate_data(config, self.exchange)

        self.init_backtest()

    def _validate_pairlists_for_backtesting(self):
        if "VolumePairList" in self.pairlists.name_list:
            raise OperationalException(
                "VolumePairList not allowed for backtesting. Please use StaticPairList instead."
            )

        if len(self.strategylist) > 1 and "PrecisionFilter" in self.pairlists.name_list:
            raise OperationalException(
                "PrecisionFilter not allowed for backtesting multiple strategies."
            )

    @staticmethod
    def cleanup():
        LoggingMixin.show_output = True
        enable_database_use()

    def init_backtest_detail(self) -> None:
        # Load detail timeframe if specified
        self.timeframe_detail = str(self.config.get("timeframe_detail", ""))
        if self.timeframe_detail:
            timeframe_detail_secs = timeframe_to_seconds(self.timeframe_detail)
            self.timeframe_detail_td = timedelta(seconds=timeframe_detail_secs)
            if self.timeframe_secs <= timeframe_detail_secs:
                raise OperationalException(
                    "Detail timeframe must be smaller than strategy timeframe."
                )

        else:
            self.timeframe_detail_td = timedelta(seconds=0)
        self.detail_data: dict[str, DataFrame] = {}
        self.futures_data: dict[str, DataFrame] = {}

    def init_backtest(self):
        self.prepare_backtest(False)

        self.wallets = Wallets(self.config, self.exchange, is_backtest=True)

        self.progress = BTProgress()
        self.abort = False

    def _set_strategy(self, strategy: IStrategy):
        """
        Load strategy into backtesting
        """
        self.strategy: IStrategy = strategy
        strategy.dp = self.dataprovider
        # Attach Wallets to Strategy baseclass
        strategy.wallets = self.wallets
        # Set stoploss_on_exchange to false for backtesting,
        # since a "perfect" stoploss-exit is assumed anyway
        # And the regular "stoploss" function would not apply to that case
        self.strategy.order_types["stoploss_on_exchange"] = False
        # Update can_short flag
        self._can_short = self.trading_mode != TradingMode.SPOT and strategy.can_short

        self.strategy.ft_bot_start()

    def _get_precalculation_interval_td(self) -> timedelta:
        """
        Converts the precalculation_interval_str to a timedelta.
        Supports 'daily', 'weekly', 'monthly', or a number of candles (e.g., '100c').
        """
        interval_str = self.precalculation_interval_str.lower()
        if interval_str == "daily":
            return timedelta(days=1)
        elif interval_str == "weekly":
            return timedelta(weeks=1)
        elif interval_str == "monthly":
            # Approximate, actual month length varies. For pre-calculation, this is acceptable.
            return timedelta(days=30)
        elif interval_str.endswith("c"):
            try:
                num_candles = int(interval_str[:-1])
                if num_candles <= 0:
                    raise ValueError("Candle interval must be positive.")
                return timedelta(seconds=self.timeframe_secs * num_candles)
            except ValueError as e:
                raise OperationalException(
                    f"Invalid pairlist_precalc_interval candle format: '{self.precalculation_interval_str}'. "
                    f"Expected format like '100c'. Error: {e}"
                )
        else:
            # Try to parse as a timeframe string (e.g., '1h', '4h')
            try:
                return timedelta(seconds=timeframe_to_seconds(interval_str))
            except ExchangeError as e:  # Catch ccxt's NotSupported and other exchange errors
                raise OperationalException(
                    f"Invalid pairlist_precalc_interval timeframe string: '{self.precalculation_interval_str}'. "
                    f"Error: {e}"
                ) from e
            except ValueError: # General catch for other unexpected format issues
                raise OperationalException(
                    f"Invalid pairlist_precalc_interval: '{self.precalculation_interval_str}'. "
                    "Supported values: 'daily', 'weekly', 'monthly', '<N>c' (e.g., '100c'), "
                    "or a valid timeframe string (e.g., '1h', '4d')."
                )

    def _precalculate_pairlist_timeline(self, min_date: datetime, max_date: datetime):
        """
        Pre-calculates the pairlist timeline for the entire backtest range.
        Uses a file-based cache to store/retrieve timelines for specific pairlist configurations.
        """
        logger.info("Attempting to load or pre-calculate pairlist timeline...")
        self.progress.init_step(BacktestState.PAIRLIST_PRECALC, 0)  # Will be re-initialized later
        self.pairlist_timeline = {}

        # 1. Determine Cache Filename
        cache_file: Path | None = None
        try:
            active_pairlist_name = self.pairlists.name_list[0] if self.pairlists.name_list else "UnknownPairlist"
            logger.debug(f"Attempting to find config for active_pairlist_name: '{active_pairlist_name}'")
            pairlist_cfg_for_hash = {}
            
            pairlists_config_list = self.config.get('pairlists', [])
            logger.debug(f"Full pairlists config from self.config.get('pairlists', []): {pairlists_config_list}")

            if active_pairlist_name != "UnknownPairlist":
                for idx, pairlist_entry in enumerate(pairlists_config_list):
                    logger.debug(f"Checking pairlist_entry #{idx}: {pairlist_entry}")
                    if isinstance(pairlist_entry, dict):
                        entry_method_name = pairlist_entry.get('method')
                        logger.debug(f"  Entry method name: '{entry_method_name}'")
                        # Standard format: {"method": "MyPairList", "other_config": "value"}
                        if entry_method_name == active_pairlist_name:
                            logger.debug(f"  Match found for '{active_pairlist_name}' using 'method' key.")
                            pairlist_cfg_for_hash = {
                                k: v for k, v in pairlist_entry.items() if k != 'method'
                            }
                            break
                        # Older/alternative format (less likely for named pairlists from list): {"MyPairList": {"other_config": "value"}}
                        elif active_pairlist_name in pairlist_entry and isinstance(pairlist_entry[active_pairlist_name], dict):
                            logger.debug(f"  Match found for '{active_pairlist_name}' as a direct key.")
                            pairlist_cfg_for_hash = pairlist_entry[active_pairlist_name]
                            break
                        else:
                            logger.debug(f"  No match in pairlist_entry #{idx} for '{active_pairlist_name}'.")
                    else:
                        logger.debug(f"  pairlist_entry #{idx} is not a dict: {type(pairlist_entry)}")
            
            logger.debug(f"Final pairlist_cfg_for_hash for '{active_pairlist_name}': {pairlist_cfg_for_hash}")
            if not pairlist_cfg_for_hash and active_pairlist_name not in ('StaticPairList', "UnknownPairlist"):
                logger.warning(
                    f"Could not find specific configuration for pairlist '{active_pairlist_name}' "
                    f"in config['pairlists'] for cache key generation. Using empty dict."
                )

            hash_content_str = (
                f"{self.config['exchange']['name']}:"
                f"{self.config['stake_currency']}:"
                f"{active_pairlist_name}:"
                f"{json.dumps(pairlist_cfg_for_hash, sort_keys=True)}:"
                f"{self.timeframe}"
            )
            cache_filename_hash = hashlib.sha256(hash_content_str.encode('utf-8')).hexdigest()
            cache_dir = Path(self.config['user_data_dir']) / PAIRLIST_TIMELINE_CACHE_DIR
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{cache_filename_hash}.json"
            logger.debug(f"Pairlist timeline cache file: {cache_file}")

        except Exception as e:
            logger.error(f"Error generating cache filename for pairlist timeline: {e}. Skipping cache.")
            # cache_file remains None

        master_cached_timeline_str_keys: Dict[str, List[str]] = {}
        if cache_file:
            try:
                if cache_file.exists():
                    with cache_file.open('r') as f:
                        master_cached_timeline_str_keys = json.load(f)
                    logger.info(f"Loaded master pairlist timeline from cache: {cache_file}")
            except Exception as e:
                logger.warning(
                    f"Could not load pairlist timeline cache from {cache_file}: {e}. Will recompute."
                )
                master_cached_timeline_str_keys = {}

        interval_td = self._get_precalculation_interval_td()
        if interval_td.total_seconds() == 0:
            logger.warning("Pairlist pre-calculation interval is zero, skipping.")
            self.progress.init_step(BacktestState.PAIRLIST_PRECALC, 1)
            self.progress.set_new_value(1)
            return

        current_run_timeline_dt_keys: Dict[datetime, List[str]] = {}
        original_slice_date = self.dataprovider._DataProvider__slice_date  # type: ignore

        num_steps = 0
        temp_eval_time = min_date
        while temp_eval_time <= max_date:
            num_steps += 1
            temp_eval_time += interval_td
        if num_steps == 0 and min_date <= max_date:
            num_steps = 1

        self.progress.init_step(BacktestState.PAIRLIST_PRECALC, num_steps if num_steps > 0 else 1)
        step_count = 0
        needs_save = False
        current_eval_time = min_date

        while current_eval_time <= max_date:
            self.check_abort()
            iso_key = current_eval_time.isoformat()

            if iso_key in master_cached_timeline_str_keys:
                current_whitelist = master_cached_timeline_str_keys[iso_key]
                logger.debug(
                    f"Using cached pairlist for {current_eval_time.strftime(DATETIME_PRINT_FORMAT)}: "
                    f"{len(current_whitelist)} pairs"
                )
            else:
                logger.debug(
                    f"Computing pairlist for {current_eval_time.strftime(DATETIME_PRINT_FORMAT)}"
                )
                self.dataprovider._set_dataframe_max_date(current_eval_time)
                self.pairlists.refresh_pairlist()
                current_whitelist = self.pairlists.whitelist.copy()
                master_cached_timeline_str_keys[iso_key] = current_whitelist
                needs_save = True
                logger.debug(
                    f"Computed whitelist for {current_eval_time.strftime(DATETIME_PRINT_FORMAT)}: "
                    f"{len(current_whitelist)} pairs"
                )

            current_run_timeline_dt_keys[current_eval_time] = current_whitelist

            current_eval_time += interval_td
            step_count += 1
            self.progress.set_new_value(step_count)

            # Handle final point if interval_td doesn't align perfectly with max_date
            # and max_date has not been processed yet.
            if current_eval_time > max_date and max_date not in current_run_timeline_dt_keys:
                final_eval_time = max_date
                final_iso_key = final_eval_time.isoformat()
                if final_iso_key in master_cached_timeline_str_keys:
                    final_whitelist = master_cached_timeline_str_keys[final_iso_key]
                    logger.debug(
                        f"Using cached pairlist for final point "
                        f"{final_eval_time.strftime(DATETIME_PRINT_FORMAT)}: {len(final_whitelist)} pairs"
                    )
                else:
                    logger.debug(
                        f"Computing pairlist for final point "
                        f"{final_eval_time.strftime(DATETIME_PRINT_FORMAT)}"
                    )
                    self.dataprovider._set_dataframe_max_date(final_eval_time)
                    self.pairlists.refresh_pairlist()
                    final_whitelist = self.pairlists.whitelist.copy()
                    master_cached_timeline_str_keys[final_iso_key] = final_whitelist
                    needs_save = True
                    logger.debug(
                        f"Computed whitelist for final point "
                        f"{final_eval_time.strftime(DATETIME_PRINT_FORMAT)}: {len(final_whitelist)} pairs"
                    )
                current_run_timeline_dt_keys[final_eval_time] = final_whitelist
                # Ensure progress reflects this potential extra step if it wasn't the last one counted
                if step_count <= num_steps : # Check if this step was already counted or is an "extra" one
                    self.progress.set_new_value(min(step_count +1, num_steps if num_steps > 0 else 1))


        self.dataprovider._set_dataframe_max_date(original_slice_date)

        if cache_file and needs_save:
            try:
                with cache_file.open('w') as f:
                    json.dump(master_cached_timeline_str_keys, f, indent=2) # indent for readability
                logger.info(f"Saved updated pairlist timeline to cache: {cache_file}")
            except Exception as e:
                logger.error(f"Could not save pairlist timeline cache to {cache_file}: {e}")

        self.pairlist_timeline = current_run_timeline_dt_keys
        logger.info(
            f"Pairlist timeline pre-calculation complete. "
            f"{len(self.pairlist_timeline)} timeline entries for current run."
        )
        if num_steps == 0: # Should ideally not happen if min_date <= max_date
            self.progress.init_step(BacktestState.PAIRLIST_PRECALC, 1) # Ensure init
            self.progress.set_new_value(1) # Mark as complete
        elif self.progress._max_steps and step_count < self.progress._max_steps:
             self.progress.set_new_value(self.progress._max_steps) # Ensure 100%


    def _load_protections(self, strategy: IStrategy):
        if self.config.get("enable_protections", False):
            self.protections = ProtectionManager(self.config, strategy.protections)

    def load_bt_data(self, pairs_to_load: Optional[List[str]] = None) -> tuple[dict[str, DataFrame], TimeRange]:
        """
        Loads backtest data and returns the data combined with the timerange
        as tuple.
        :param pairs_to_load: Optional list of pairs to load. If None, uses self.pairlists.whitelist.
        """
        effective_pairs = pairs_to_load if pairs_to_load is not None else self.pairlists.whitelist
        if not effective_pairs:
            logger.warning("load_bt_data called with no effective pairs to load. Returning empty data.")
            # Return a copy of timerange to avoid modification issues if it's used later
            return {}, deepcopy(self.timerange)

        self.progress.init_step(BacktestState.DATALOAD, 1)

        data = history.load_data(
            datadir=self.config["datadir"],
            pairs=effective_pairs, # Use effective_pairs
            timeframe=self.timeframe,
            timerange=self.timerange, # self.timerange is used by history.load_data
            startup_candles=self.required_startup,
            fail_without_data=False, # Set to False to allow loading partial data if some superset pairs are missing
            data_format=self.config["dataformat_ohlcv"],
            candle_type=self.config.get("candle_type_def", CandleType.SPOT),
        )

        if not data: # Check if history.load_data returned an empty dict
            logger.warning(f"No data loaded for any of the {len(effective_pairs)} effective pairs. "
                           f"Pairs attempted: {effective_pairs[:20]}")
            # Return a copy of timerange
            return {}, deepcopy(self.timerange)

        min_date, max_date = history.get_timerange(data) # Will be based on actually loaded data

        # Adjust startts forward if not enough data is available for the loaded set
        # This adjustment is critical and should use min_date from the *actually loaded data for these effective_pairs*
        current_timerange_to_adjust = deepcopy(self.timerange) # Use a copy for this specific load's adjustment context
        current_timerange_to_adjust.adjust_start_if_necessary(
            timeframe_to_seconds(self.timeframe), self.required_startup, min_date
        )

        self.progress.set_new_value(1)
        self._load_bt_data_detail(pairs_for_detail=list(data.keys())) # Pass actually loaded pairs
        self.price_pair_prec = {}
        for pair in data.keys(): # Iterate over keys of actually loaded data
            if pair in data: # Redundant check, but safe
                # Load price precision logic
                self.price_pair_prec[pair] = get_tick_size_over_time(data[pair])
        # Return the timerange that was potentially adjusted for *this specific call*
        return data, current_timerange_to_adjust

    def _load_bt_data_detail(self, pairs_for_detail: Optional[List[str]] = None) -> None:
        """
        Loads backtest detail data (smaller timeframe) if necessary.
        :param pairs_for_detail: Optional list of pairs for which to load detail data.
                               If None, defaults to self.pairlists.whitelist (legacy behavior).
        """
        effective_pairs = pairs_for_detail if pairs_for_detail is not None else self.pairlists.whitelist
        if not effective_pairs:
            logger.debug("_load_bt_data_detail called with no effective_pairs. Skipping detail/futures data load.")
            self.detail_data = {}
            self.futures_data = {}
            return

        if self.timeframe_detail:
            self.detail_data = history.load_data(
                datadir=self.config["datadir"],
                pairs=effective_pairs, # Use effective_pairs
                timeframe=self.timeframe_detail,
                timerange=self.timerange, # Uses the main backtest timerange
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config["dataformat_ohlcv"],
                candle_type=self.config.get("candle_type_def", CandleType.SPOT),
            )
        else:
            self.detail_data = {}
        if self.trading_mode == TradingMode.FUTURES:
            funding_fee_timeframe: str = self.exchange.get_option("funding_fee_timeframe")
            self.funding_fee_timeframe_secs: int = timeframe_to_seconds(funding_fee_timeframe)
            mark_timeframe: str = self.exchange.get_option("mark_ohlcv_timeframe")

            # Load additional futures data.
            funding_rates_dict = history.load_data(
                datadir=self.config["datadir"],
                pairs=effective_pairs, # Use effective_pairs
                timeframe=funding_fee_timeframe,
                timerange=self.timerange, # Uses the main backtest timerange
                startup_candles=0,
                fail_without_data=False, # Allow partial data for futures info
                data_format=self.config["dataformat_ohlcv"],
                candle_type=CandleType.FUNDING_RATE,
            )

            # For simplicity, assign to CandleType.Mark (might contain index candles!)
            mark_rates_dict = history.load_data(
                datadir=self.config["datadir"],
                pairs=effective_pairs, # Use effective_pairs
                timeframe=mark_timeframe,
                timerange=self.timerange, # Uses the main backtest timerange
                startup_candles=0,
                fail_without_data=False, # Allow partial data for futures info
                data_format=self.config["dataformat_ohlcv"],
                candle_type=CandleType.from_string(self.exchange.get_option("mark_ohlcv_price")),
            )
            # Combine data to avoid combining the data per trade.
            unavailable_pairs = []
            uses_leverage_tiers = self.exchange.get_option("uses_leverage_tiers", True)
            # Iterate over effective_pairs for which we attempted to load data
            for pair in effective_pairs:
                if pair not in funding_rates_dict or pair not in mark_rates_dict:
                    logger.debug(f"Funding or mark rate data missing for {pair}, cannot combine for futures_data.")
                    continue # Skip if essential data for combining is missing

                if uses_leverage_tiers and pair not in self.exchange._leverage_tiers:
                    unavailable_pairs.append(pair)
                    logger.warning(f"Leverage tiers not available for {pair}, cannot backtest in futures mode.")
                    continue

                self.futures_data[pair] = self.exchange.combine_funding_and_mark(
                    funding_rates=funding_rates_dict[pair],
                    mark_rates=mark_rates_dict[pair],
                    futures_funding_rate=self.config.get("futures_funding_rate", None),
                )

            if unavailable_pairs:
                raise OperationalException(
                    f"Pairs {', '.join(unavailable_pairs)} got no leverage tiers available. "
                    "It is therefore impossible to backtest with this pair at the moment."
                )
        else:
            self.futures_data = {}

    def get_pair_precision(self, pair: str, current_time: datetime) -> tuple[float | None, int]:
        """
        Get pair precision at that moment in time
        :param pair: Pair to get precision for
        :param current_time: Time to get precision for
        :return: tuple of price precision, precision_mode_price for the pair at that given time.
        """
        precision_series = self.price_pair_prec.get(pair)
        if precision_series is not None:
            precision = precision_series.asof(current_time)

            if not isnan(precision):
                # Force tick size if we define the precision
                return precision, TICK_SIZE
        return self.exchange.get_precision_price(pair), self.precision_mode_price

    def disable_database_use(self):
        disable_database_use(self.timeframe)

    def prepare_backtest(self, enable_protections):
        """
        Backtesting setup method - called once for every call to "backtest()".
        """
        self.disable_database_use()
        PairLocks.reset_locks()
        Trade.reset_trades()
        CustomDataWrapper.reset_custom_data()
        self.rejected_trades = 0
        self.timedout_entry_orders = 0
        self.timedout_exit_orders = 0
        self.canceled_trade_entries = 0
        self.canceled_entry_orders = 0
        self.replaced_entry_orders = 0
        self.canceled_exit_orders = 0
        self.replaced_exit_orders = 0
        self.dataprovider.clear_cache()
        if enable_protections:
            self._load_protections(self.strategy)

    def check_abort(self):
        """
        Check if abort was requested, raise DependencyException if that's the case
        Only applies to Interactive backtest mode (webserver mode)
        """
        if self.abort:
            self.abort = False
            raise DependencyException("Stop requested")

    def _get_ohlcv_as_lists(self, processed: dict[str, DataFrame]) -> dict[str, tuple]:
        """
        Helper function to convert a processed dataframes into lists for performance reasons.

        Used by backtest() - so keep this optimized for performance.

        :param processed: a processed dictionary with format {pair, data}, which gets cleared to
        optimize memory usage!
        """

        data: dict = {}
        self.progress.init_step(BacktestState.CONVERT, len(processed))

        # Create dict with data
        for pair in processed.keys():
            pair_data = processed[pair]
            self.check_abort()
            self.progress.increment()

            if not pair_data.empty:
                # Cleanup from prior runs
                pair_data.drop(HEADERS[5:] + ["buy", "sell"], axis=1, errors="ignore")
            df_analyzed = self.strategy.ft_advise_signals(pair_data, {"pair": pair})
            # Update dataprovider cache
            self.dataprovider._set_cached_df(
                pair, self.timeframe, df_analyzed, self.config["candle_type_def"]
            )

            # Trim startup period from analyzed dataframe
            df_trimmed_for_pair = trim_dataframe( # Use a new variable name for clarity
                df_analyzed, self.timerange, startup_candles=self.required_startup
            )

            df_analyzed = processed[pair] = pair_data = df_trimmed_for_pair # Assign back

            # Create a copy of the dataframe before shifting, that way the entry signal/tag
            # remains on the correct candle for callbacks.
            df_analyzed = df_analyzed.copy()

            # To avoid using data from future, we use entry/exit signals shifted
            # from the previous candle
            for col in HEADERS[5:]:
                tag_col = col in ("enter_tag", "exit_tag")
                if col in df_analyzed.columns:
                    df_analyzed[col] = (
                        df_analyzed.loc[:, col]
                        .replace([nan], [0 if not tag_col else None])
                        .shift(1)
                    )
                elif not df_analyzed.empty:
                    df_analyzed[col] = 0 if not tag_col else None

            df_analyzed = df_analyzed.drop(df_analyzed.head(1).index)

            # Convert from Pandas to list for performance reasons
            # (Looping Pandas is slow.)
            data[pair] = df_analyzed[HEADERS].values.tolist() if not df_analyzed.empty else []
        return data

    def _get_close_rate(
        self, row: tuple, trade: LocalTrade, exit_: ExitCheckTuple, trade_dur: int
    ) -> float:
        """
        Get close rate for backtesting result
        """
        # Special handling if high or low hit STOP_LOSS or ROI
        if exit_.exit_type in (
            ExitType.STOP_LOSS,
            ExitType.TRAILING_STOP_LOSS,
            ExitType.LIQUIDATION,
        ):
            return self._get_close_rate_for_stoploss(row, trade, exit_, trade_dur)
        elif exit_.exit_type == (ExitType.ROI):
            return self._get_close_rate_for_roi(row, trade, exit_, trade_dur)
        else:
            return row[OPEN_IDX]

    def _get_close_rate_for_stoploss(
        self, row: tuple, trade: LocalTrade, exit_: ExitCheckTuple, trade_dur: int
    ) -> float:
        # our stoploss was already lower than candle high,
        # possibly due to a cancelled trade exit.
        # exit at open price.
        is_short = trade.is_short or False
        leverage = trade.leverage or 1.0
        side_1 = -1 if is_short else 1
        if exit_.exit_type == ExitType.LIQUIDATION and trade.liquidation_price:
            stoploss_value = trade.liquidation_price
        else:
            stoploss_value = trade.stop_loss

        if is_short:
            if stoploss_value < row[LOW_IDX]:
                return row[OPEN_IDX]
        else:
            if stoploss_value > row[HIGH_IDX]:
                return row[OPEN_IDX]

        # Special case: trailing triggers within same candle as trade opened. Assume most
        # pessimistic price movement, which is moving just enough to arm stoploss and
        # immediately going down to stop price.
        if exit_.exit_type == ExitType.TRAILING_STOP_LOSS and trade_dur == 0:
            if (
                not self.strategy.use_custom_stoploss
                and self.strategy.trailing_stop
                and self.strategy.trailing_only_offset_is_reached
                and self.strategy.trailing_stop_positive_offset is not None
                and self.strategy.trailing_stop_positive
            ):
                # Worst case: price reaches stop_positive_offset and dives down.
                stop_rate = row[OPEN_IDX] * (
                    1
                    + side_1 * abs(self.strategy.trailing_stop_positive_offset)
                    - side_1 * abs(self.strategy.trailing_stop_positive / leverage)
                )
            else:
                # Worst case: price ticks tiny bit above open and dives down.
                stop_rate = row[OPEN_IDX] * (
                    1 - side_1 * abs((trade.stop_loss_pct or 0.0) / leverage)
                )

            # Limit lower-end to candle low to avoid exits below the low.
            # This still remains "worst case" - but "worst realistic case".
            if is_short:
                return min(row[HIGH_IDX], stop_rate)
            else:
                return max(row[LOW_IDX], stop_rate)

        # Set close_rate to stoploss
        return stoploss_value

    def _get_close_rate_for_roi(
        self, row: tuple, trade: LocalTrade, exit_: ExitCheckTuple, trade_dur: int
    ) -> float:
        is_short = trade.is_short or False
        leverage = trade.leverage or 1.0
        side_1 = -1 if is_short else 1
        roi_entry, roi = self.strategy.min_roi_reached_entry(trade_dur)
        if roi is not None and roi_entry is not None:
            if roi == -1 and roi_entry % self.timeframe_min == 0:
                # When force_exiting with ROI=-1, the roi time will always be equal to trade_dur.
                # If that entry is a multiple of the timeframe (so on candle open)
                # - we'll use open instead of close
                return row[OPEN_IDX]

            # - (Expected abs profit - open_rate - open_fee) / (fee_close -1)
            roi_rate = trade.open_rate * roi / leverage
            open_fee_rate = side_1 * trade.open_rate * (1 + side_1 * trade.fee_open)
            close_rate = -(roi_rate + open_fee_rate) / ((trade.fee_close or 0.0) - side_1 * 1)
            if is_short:
                is_new_roi = row[OPEN_IDX] < close_rate
            else:
                is_new_roi = row[OPEN_IDX] > close_rate
            if (
                trade_dur > 0
                and trade_dur == roi_entry
                and roi_entry % self.timeframe_min == 0
                and is_new_roi
            ):
                # new ROI entry came into effect.
                # use Open rate if open_rate > calculated exit rate
                return row[OPEN_IDX]

            if trade_dur == 0 and (
                (
                    is_short
                    # Red candle (for longs)
                    and row[OPEN_IDX] < row[CLOSE_IDX]  # Red candle
                    and trade.open_rate > row[OPEN_IDX]  # trade-open above open_rate
                    and close_rate < row[CLOSE_IDX]  # closes below close
                )
                or (
                    not is_short
                    # green candle (for shorts)
                    and row[OPEN_IDX] > row[CLOSE_IDX]  # green candle
                    and trade.open_rate < row[OPEN_IDX]  # trade-open below open_rate
                    and close_rate > row[CLOSE_IDX]  # closes above close
                )
            ):
                # ROI on opening candles with custom pricing can only
                # trigger if the entry was at Open or lower wick.
                # details: https: // github.com/freqtrade/freqtrade/issues/6261
                # If open_rate is < open, only allow exits below the close on red candles.
                raise ValueError("Opening candle ROI on red candles.")

            # Use the maximum between close_rate and low as we
            # cannot exit outside of a candle.
            # Applies when a new ROI setting comes in place and the whole candle is above that.
            return min(max(close_rate, row[LOW_IDX]), row[HIGH_IDX])

        else:
            # This should not be reached...
            return row[OPEN_IDX]

    def _check_adjust_trade_for_candle(
        self, trade: LocalTrade, row: tuple, current_time: datetime
    ) -> LocalTrade:
        current_rate: float = row[OPEN_IDX]
        current_profit = trade.calc_profit_ratio(current_rate)
        min_stake = self.exchange.get_min_pair_stake_amount(trade.pair, current_rate, -0.1)
        max_stake = self.exchange.get_max_pair_stake_amount(trade.pair, current_rate)
        stake_available = self.wallets.get_available_stake_amount()
        stake_amount, order_tag = self.strategy._adjust_trade_position_internal(
            trade=trade,  # type: ignore[arg-type]
            current_time=current_time,
            current_rate=current_rate,
            current_profit=current_profit,
            min_stake=min_stake,
            max_stake=min(max_stake, stake_available),
            current_entry_rate=current_rate,
            current_exit_rate=current_rate,
            current_entry_profit=current_profit,
            current_exit_profit=current_profit,
        )

        # Check if we should increase our position
        if stake_amount is not None and stake_amount > 0.0:
            check_adjust_entry = True
            if self.strategy.max_entry_position_adjustment > -1:
                entry_count = trade.nr_of_successful_entries
                check_adjust_entry = entry_count <= self.strategy.max_entry_position_adjustment
            if check_adjust_entry:
                pos_trade = self._enter_trade(
                    trade.pair,
                    row,
                    "short" if trade.is_short else "long",
                    stake_amount,
                    trade,
                    entry_tag1=order_tag,
                )
                if pos_trade is not None:
                    self.wallets.update()
                    return pos_trade

        if stake_amount is not None and stake_amount < 0.0:
            amount = amount_to_contract_precision(
                abs(
                    float(
                        FtPrecise(stake_amount)
                        * FtPrecise(trade.amount)
                        / FtPrecise(trade.stake_amount)
                    )
                ),
                trade.amount_precision,
                self.precision_mode,
                trade.contract_size,
            )
            if amount == 0.0:
                return trade
            remaining = (trade.amount - amount) * current_rate
            if min_stake and remaining != 0 and remaining < min_stake:
                # Remaining stake is too low to be sold.
                return trade
            exit_ = ExitCheckTuple(ExitType.PARTIAL_EXIT, order_tag)
            pos_trade = self._get_exit_for_signal(trade, row, exit_, current_time, amount)
            if pos_trade is not None:
                order = pos_trade.orders[-1]
                # If the order was filled and for the full trade amount, we need to close the trade.
                self._process_exit_order(order, pos_trade, current_time, row, trade.pair)
                return pos_trade

        return trade

    def _get_order_filled(self, rate: float, row: tuple) -> bool:
        """Rate is within candle, therefore filled"""
        return row[LOW_IDX] <= rate <= row[HIGH_IDX]

    def _call_adjust_stop(self, current_date: datetime, trade: LocalTrade, current_rate: float):
        profit = trade.calc_profit_ratio(current_rate)
        self.strategy.ft_stoploss_adjust(
            current_rate,
            trade,  # type: ignore
            current_date,
            profit,
            0,
            after_fill=True,
        )

    def _try_close_open_order(
        self, order: Order | None, trade: LocalTrade, current_date: datetime, row: tuple
    ) -> bool:
        """
        Check if an order is open and if it should've filled.
        :return:  True if the order filled.
        """
        if order and self._get_order_filled(order.ft_price, row):
            order.close_bt_order(current_date, trade)
            self._run_funding_fees(trade, current_date, force=True)
            strategy_safe_wrapper(self.strategy.order_filled, default_retval=None)(
                pair=trade.pair,
                trade=trade,  # type: ignore[arg-type]
                order=order,
                current_time=current_date,
            )

            if self.margin_mode == MarginMode.CROSS or not (
                order.ft_order_side == trade.exit_side and order.safe_amount == trade.amount
            ):
                # trade is still open or we are in cross margin mode and
                # must update all liquidation prices
                update_liquidation_prices(
                    trade,
                    exchange=self.exchange,
                    wallets=self.wallets,
                    stake_currency=self.config["stake_currency"],
                    dry_run=True,
                )
            if not (order.ft_order_side == trade.exit_side and order.safe_amount == trade.amount):
                self._call_adjust_stop(current_date, trade, order.ft_price)
            return True
        return False

    def _process_exit_order(
        self, order: Order, trade: LocalTrade, current_time: datetime, row: tuple, pair: str
    ):
        """
        Takes an exit order and processes it, potentially closing the trade.
        """
        if self._try_close_open_order(order, trade, current_time, row):
            sub_trade = order.safe_amount_after_fee != trade.amount
            if sub_trade:
                trade.recalc_trade_from_orders()
            else:
                trade.close_date = current_time
                trade.close(order.ft_price, show_msg=False)

                LocalTrade.close_bt_trade(trade)
            self.wallets.update()
            self.run_protections(pair, current_time, trade.trade_direction)

    def _get_exit_for_signal(
        self,
        trade: LocalTrade,
        row: tuple,
        exit_: ExitCheckTuple,
        current_time: datetime,
        amount: float | None = None,
    ) -> LocalTrade | None:
        if exit_.exit_flag:
            trade.close_date = current_time
            exit_reason = exit_.exit_reason
            amount_ = amount if amount is not None else trade.amount
            trade_dur = int((trade.close_date_utc - trade.open_date_utc).total_seconds() // 60)
            try:
                close_rate = self._get_close_rate(row, trade, exit_, trade_dur)
            except ValueError:
                return None
            # call the custom exit price,with default value as previous close_rate
            current_profit = trade.calc_profit_ratio(close_rate)
            order_type = self.strategy.order_types["exit"]
            if exit_.exit_type in (
                ExitType.EXIT_SIGNAL,
                ExitType.CUSTOM_EXIT,
                ExitType.PARTIAL_EXIT,
            ):
                # Checks and adds an exit tag, after checking that the length of the
                # row has the length for an exit tag column
                if (
                    len(row) > EXIT_TAG_IDX
                    and row[EXIT_TAG_IDX] is not None
                    and len(row[EXIT_TAG_IDX]) > 0
                    and exit_.exit_type in (ExitType.EXIT_SIGNAL,)
                ):
                    exit_reason = row[EXIT_TAG_IDX]
                # Custom exit pricing only for exit-signals
                if order_type == "limit":
                    rate = strategy_safe_wrapper(
                        self.strategy.custom_exit_price, default_retval=close_rate
                    )(
                        pair=trade.pair,
                        trade=trade,  # type: ignore[arg-type]
                        current_time=current_time,
                        proposed_rate=close_rate,
                        current_profit=current_profit,
                        exit_tag=exit_reason,
                    )
                    if rate is not None and rate != close_rate:
                        close_rate = price_to_precision(
                            rate, trade.price_precision, trade.precision_mode_price
                        )
                    # We can't place orders lower than current low.
                    # freqtrade does not support this in live, and the order would fill immediately
                    if trade.is_short:
                        close_rate = min(close_rate, row[HIGH_IDX])
                    else:
                        close_rate = max(close_rate, row[LOW_IDX])
            # Confirm trade exit:
            time_in_force = self.strategy.order_time_in_force["exit"]

            if exit_.exit_type not in (
                ExitType.LIQUIDATION,
                ExitType.PARTIAL_EXIT,
            ) and not strategy_safe_wrapper(self.strategy.confirm_trade_exit, default_retval=True)(
                pair=trade.pair,
                trade=trade,  # type: ignore[arg-type]
                order_type=order_type,
                amount=amount_,
                rate=close_rate,
                time_in_force=time_in_force,
                sell_reason=exit_reason,  # deprecated
                exit_reason=exit_reason,
                current_time=current_time,
            ):
                return None

            trade.exit_reason = exit_reason

            return self._exit_trade(trade, row, close_rate, amount_, exit_reason)
        return None

    def _exit_trade(
        self,
        trade: LocalTrade,
        sell_row: tuple,
        close_rate: float,
        amount: float,
        exit_reason: str | None,
    ) -> LocalTrade | None:
        self.order_id_counter += 1
        exit_candle_time = sell_row[DATE_IDX].to_pydatetime()
        order_type = self.strategy.order_types["exit"]
        # amount = amount or trade.amount
        amount = amount_to_contract_precision(
            amount or trade.amount, trade.amount_precision, self.precision_mode, trade.contract_size
        )

        if self.handle_similar_order(trade, close_rate, amount, trade.exit_side, exit_candle_time):
            return None

        order = Order(
            id=self.order_id_counter,
            ft_trade_id=trade.id,
            order_date=exit_candle_time,
            order_update_date=exit_candle_time,
            ft_is_open=True,
            ft_pair=trade.pair,
            order_id=str(self.order_id_counter),
            symbol=trade.pair,
            ft_order_side=trade.exit_side,
            side=trade.exit_side,
            order_type=order_type,
            status="open",
            ft_price=close_rate,
            price=close_rate,
            average=close_rate,
            amount=amount,
            filled=0,
            remaining=amount,
            cost=amount * close_rate * (1 + self.fee),
            ft_order_tag=exit_reason,
        )
        order._trade_bt = trade
        trade.orders.append(order)
        return trade

    def _check_trade_exit(
        self, trade: LocalTrade, row: tuple, current_time: datetime
    ) -> LocalTrade | None:
        self._run_funding_fees(trade, current_time)

        # Check if we need to adjust our current positions
        if self.strategy.position_adjustment_enable:
            trade = self._check_adjust_trade_for_candle(trade, row, current_time)

        if trade.is_open:
            enter_signal_val = row[SHORT_IDX] if trade.is_short else row[LONG_IDX]
            exit_signal_val = row[ESHORT_IDX] if trade.is_short else row[ELONG_IDX]
            candle_time = row[DATE_IDX].to_pydatetime()
            candle_open_price = row[OPEN_IDX]
            candle_low_price = row[LOW_IDX]
            candle_high_price = row[HIGH_IDX]

            exits: list[ExitCheckTuple] = self.strategy.should_exit(
                trade,  # type: ignore
                candle_open_price, # current_rate for should_exit
                candle_time,    # current_time for should_exit
                enter=enter_signal_val,
                exit_=exit_signal_val,
                low=candle_low_price,
                high=candle_high_price,
            )

            for exit_check_item in exits:
                # current_time here is the main loop's current_time, which should be same as candle_time
                processed_trade = self._get_exit_for_signal(trade, row, exit_check_item, current_time)
                if processed_trade:
                    # If _get_exit_for_signal returns a trade object, it means an exit order was created.
                    # The actual processing of that order (filling it, closing the trade) happens later
                    # in the main backtest_loop via _process_exit_order.
                    # This return 't' (now 'processed_trade') is used by backtest_loop to know an exit was initiated.
                    return processed_trade
        return None

    def _run_funding_fees(self, trade: LocalTrade, current_time: datetime, force: bool = False):
        """
        Calculate funding fees if necessary and add them to the trade.
        """
        if self.trading_mode == TradingMode.FUTURES:
            if force or (current_time.timestamp() % self.funding_fee_timeframe_secs) == 0:
                # Funding fee interval.
                trade.set_funding_fees(
                    self.exchange.calculate_funding_fees(
                        self.futures_data[trade.pair],
                        amount=trade.amount,
                        is_short=trade.is_short,
                        open_date=trade.date_last_filled_utc,
                        close_date=current_time,
                    )
                )

    def get_valid_price_and_stake(
        self,
        pair: str,
        row: tuple,
        propose_rate: float,
        stake_amount: float,
        direction: LongShort,
        current_time: datetime,
        entry_tag: str | None,
        trade: LocalTrade | None,
        order_type: str,
        price_precision: float | None,
        precision_mode_price: int,
    ) -> tuple[float, float, float, float]:
        if order_type == "limit":
            new_rate = strategy_safe_wrapper(
                self.strategy.custom_entry_price, default_retval=propose_rate
            )(
                pair=pair,
                trade=trade,  # type: ignore[arg-type]
                current_time=current_time,
                proposed_rate=propose_rate,
                entry_tag=entry_tag,
                side=direction,
            )  # default value is the open rate
            # We can't place orders higher than current high (otherwise it'd be a stop limit entry)
            # which freqtrade does not support in live.
            if new_rate is not None and new_rate != propose_rate:
                propose_rate = price_to_precision(new_rate, price_precision, precision_mode_price)
            if direction == "short":
                propose_rate = max(propose_rate, row[LOW_IDX])
            else:
                propose_rate = min(propose_rate, row[HIGH_IDX])

        pos_adjust = trade is not None
        leverage = trade.leverage if trade else 1.0
        if not pos_adjust:
            try:
                stake_amount = self.wallets.get_trade_stake_amount(
                    pair, self.strategy.max_open_trades, update=False
                )
            except DependencyException:
                return 0, 0, 0, 0

            max_leverage = self.exchange.get_max_leverage(pair, stake_amount)
            leverage = (
                strategy_safe_wrapper(self.strategy.leverage, default_retval=1.0)(
                    pair=pair,
                    current_time=current_time,
                    current_rate=row[OPEN_IDX],
                    proposed_leverage=1.0,
                    max_leverage=max_leverage,
                    side=direction,
                    entry_tag=entry_tag,
                )
                if self.trading_mode != TradingMode.SPOT
                else 1.0
            )
            # Cap leverage between 1.0 and max_leverage.
            leverage = min(max(leverage, 1.0), max_leverage)

        min_stake_amount = (
            self.exchange.get_min_pair_stake_amount(
                pair, propose_rate, -0.05 if not pos_adjust else 0.0, leverage=leverage
            )
            or 0
        )
        max_stake_amount = self.exchange.get_max_pair_stake_amount(
            pair, propose_rate, leverage=leverage
        )
        stake_available = self.wallets.get_available_stake_amount()

        if not pos_adjust:
            stake_amount = strategy_safe_wrapper(
                self.strategy.custom_stake_amount, default_retval=stake_amount
            )(
                pair=pair,
                current_time=current_time,
                current_rate=propose_rate,
                proposed_stake=stake_amount,
                min_stake=min_stake_amount,
                max_stake=min(stake_available, max_stake_amount),
                leverage=leverage,
                entry_tag=entry_tag,
                side=direction,
            )

        stake_amount_val = self.wallets.validate_stake_amount(
            pair=pair,
            stake_amount=stake_amount,
            min_stake_amount=min_stake_amount,
            max_stake_amount=max_stake_amount,
            trade_amount=trade.stake_amount if trade else None,
        )

        return propose_rate, stake_amount_val, leverage, min_stake_amount

    def _enter_trade(
        self,
        pair: str,
        row: tuple,
        direction: LongShort,
        stake_amount: float | None = None,
        trade: LocalTrade | None = None,
        requested_rate: float | None = None,
        requested_stake: float | None = None,
        entry_tag1: str | None = None,
    ) -> LocalTrade | None:
        """
        :param trade: Trade to adjust - initial entry if None
        :param requested_rate: Adjusted entry rate
        :param requested_stake: Stake amount for adjusted orders (`adjust_entry_price`).
        """

        current_time = row[DATE_IDX].to_pydatetime()
        entry_tag = entry_tag1 or (row[ENTER_TAG_IDX] if len(row) >= ENTER_TAG_IDX + 1 else None)
        # let's call the custom entry price, using the open price as default price
        order_type = self.strategy.order_types["entry"]
        pos_adjust = trade is not None and requested_rate is None

        stake_amount_ = stake_amount or (trade.stake_amount if trade else 0.0)
        precision_price, precision_mode_price = self.get_pair_precision(pair, current_time)

        propose_rate, stake_amount, leverage, min_stake_amount = self.get_valid_price_and_stake(
            pair,
            row,
            row[OPEN_IDX],
            stake_amount_,
            direction,
            current_time,
            entry_tag,
            trade,
            order_type,
            precision_price,
            precision_mode_price,
        )

        # replace proposed rate if another rate was requested
        propose_rate = requested_rate if requested_rate else propose_rate
        stake_amount = requested_stake if requested_stake else stake_amount

        if not stake_amount:
            # In case of pos adjust, still return the original trade
            # If not pos adjust, trade is None
            return trade
        time_in_force = self.strategy.order_time_in_force["entry"]

        if stake_amount and (not min_stake_amount or stake_amount >= min_stake_amount):
            self.order_id_counter += 1
            base_currency = self.exchange.get_pair_base_currency(pair)
            amount_p = (stake_amount / propose_rate) * leverage

            contract_size = self.exchange.get_contract_size(pair)
            precision_amount = self.exchange.get_precision_amount(pair)
            amount = amount_to_contract_precision(
                amount_p, precision_amount, self.precision_mode, contract_size
            )
            if not amount:
                # No amount left after truncating to precision.
                return trade
            # Backcalculate actual stake amount.
            stake_amount = amount * propose_rate / leverage

            if not pos_adjust:
                # Confirm trade entry:
                if not strategy_safe_wrapper(
                    self.strategy.confirm_trade_entry, default_retval=True
                )(
                    pair=pair,
                    order_type=order_type,
                    amount=amount,
                    rate=propose_rate,
                    time_in_force=time_in_force,
                    current_time=current_time,
                    entry_tag=entry_tag,
                    side=direction,
                ):
                    return trade

            is_short = direction == "short"
            # Necessary for Margin trading. Disabled until support is enabled.
            # interest_rate = self.exchange.get_interest_rate()

            if trade is None:
                # Enter trade
                self.trade_id_counter += 1
                trade = LocalTrade(
                    id=self.trade_id_counter,
                    pair=pair,
                    base_currency=base_currency,
                    stake_currency=self.config["stake_currency"],
                    open_rate=propose_rate,
                    open_rate_requested=propose_rate,
                    open_date=current_time,
                    stake_amount=stake_amount,
                    amount=0,
                    amount_requested=amount,
                    fee_open=self.fee,
                    fee_close=self.fee,
                    is_open=True,
                    enter_tag=entry_tag,
                    timeframe=self.timeframe_min,
                    exchange=self._exchange_name,
                    is_short=is_short,
                    trading_mode=self.trading_mode,
                    leverage=leverage,
                    # interest_rate=interest_rate,
                    amount_precision=precision_amount,
                    price_precision=precision_price,
                    precision_mode=self.precision_mode,
                    precision_mode_price=precision_mode_price,
                    contract_size=contract_size,
                    orders=[],
                )
                LocalTrade.add_bt_trade(trade)
            elif self.handle_similar_order(
                trade, propose_rate, amount, trade.entry_side, current_time
            ):
                return None

            trade.adjust_stop_loss(trade.open_rate, self.strategy.stoploss, initial=True)

            order = Order(
                id=self.order_id_counter,
                ft_trade_id=trade.id,
                ft_is_open=True,
                ft_pair=trade.pair,
                order_id=str(self.order_id_counter),
                symbol=trade.pair,
                ft_order_side=trade.entry_side,
                side=trade.entry_side,
                order_type=order_type,
                status="open",
                order_date=current_time,
                order_filled_date=current_time,
                order_update_date=current_time,
                ft_price=propose_rate,
                price=propose_rate,
                average=propose_rate,
                amount=amount,
                filled=0,
                remaining=amount,
                cost=amount * propose_rate * (1 + self.fee),
                ft_order_tag=entry_tag,
            )
            order._trade_bt = trade
            trade.orders.append(order)
            self._try_close_open_order(order, trade, current_time, row)
            trade.recalc_trade_from_orders()

        return trade

    def handle_left_open(
        self, open_trades: dict[str, list[LocalTrade]], data: dict[str, list[tuple]]
    ) -> None:
        """
        Handling of left open trades at the end of backtesting
        """
        for pair in open_trades.keys():
            for trade in list(open_trades[pair]):
                if (
                    trade.has_open_orders and trade.nr_of_successful_entries == 0
                ) or not trade.has_open_position:
                    # Ignore trade if entry-order did not fill yet
                    LocalTrade.remove_bt_trade(trade)
                    continue

                exit_row = data[pair][-1]
                self._exit_trade(
                    trade, exit_row, exit_row[OPEN_IDX], trade.amount, ExitType.FORCE_EXIT.value
                )
                trade.exit_reason = ExitType.FORCE_EXIT.value
                self._process_exit_order(
                    trade.orders[-1], trade, exit_row[DATE_IDX].to_pydatetime(), exit_row, pair
                )

    def trade_slot_available(self, open_trade_count: int) -> bool:
        # Always allow trades when max_open_trades is enabled.
        max_open_trades: IntOrInf = self.strategy.max_open_trades
        if max_open_trades <= 0 or open_trade_count < max_open_trades:
            return True
        # Rejected trade
        self.rejected_trades += 1
        return False

    def check_for_trade_entry(self, row) -> LongShort | None:
        enter_long = row[LONG_IDX] == 1
        exit_long = row[ELONG_IDX] == 1
        # Ensure SHORT_IDX is valid before accessing
        enter_short_signal = row[SHORT_IDX] if len(row) > SHORT_IDX else 0
        enter_short = self._can_short and enter_short_signal == 1
        # Ensure ESHORT_IDX is valid
        exit_short_signal = row[ESHORT_IDX] if len(row) > ESHORT_IDX else 0
        exit_short = self._can_short and exit_short_signal == 1

        if enter_long == 1 and not any([exit_long, enter_short]):
            # Long
            return "long"
        if enter_short == 1 and not any([exit_short, enter_long]):
            # Short
            return "short"
        return None

    def run_protections(self, pair: str, current_time: datetime, side: LongShort):
        if self.enable_protections:
            self.protections.stop_per_pair(pair, current_time, side)
            self.protections.global_stop(current_time, side)

    def manage_open_orders(self, trade: LocalTrade, current_time: datetime, row: tuple) -> bool:
        """
        Check if any open order needs to be cancelled or replaced.
        Returns True if the trade should be deleted.
        """
        for order in [o for o in trade.orders if o.ft_is_open]:
            oc = self.check_order_cancel(trade, order, current_time)
            if oc:
                # delete trade due to order timeout
                return True
            elif oc is None and self.check_order_replace(trade, order, current_time, row):
                # delete trade due to user request
                self.canceled_trade_entries += 1
                return True
        # default maintain trade
        return False

    def cancel_open_orders(self, trade: LocalTrade, current_time: datetime):
        """
        Cancel all open orders for the given trade.
        """
        for order in [o for o in trade.orders if o.ft_is_open]:
            if order.side == trade.entry_side:
                self.canceled_entry_orders += 1
            elif order.side == trade.exit_side:
                self.canceled_exit_orders += 1
            # canceled orders are removed from the trade
            del trade.orders[trade.orders.index(order)]

    def handle_similar_order(
        self, trade: LocalTrade, price: float, amount: float, side: str, current_time: datetime
    ) -> bool:
        """
        Handle similar order for the given trade.
        """
        if trade.has_open_orders:
            oo = trade.select_order(side, True)
            if oo:
                if (price == oo.price) and (side == oo.side) and (amount == oo.amount):
                    return True
            self.cancel_open_orders(trade, current_time)

        return False

    def check_order_cancel(
        self, trade: LocalTrade, order: Order, current_time: datetime
    ) -> bool | None:
        """
        Check if current analyzed order has to be canceled.
        Returns True if the trade should be Deleted (initial order was canceled),
                False if it's Canceled
                None if the order is still active.
        """
        timedout = self.strategy.ft_check_timed_out(
            trade,  # type: ignore[arg-type]
            order,
            current_time,
        )
        if timedout:
            if order.side == trade.entry_side:
                self.timedout_entry_orders += 1
                if trade.nr_of_successful_entries == 0:
                    # Remove trade due to entry timeout expiration.
                    return True
                else:
                    # Close additional entry order
                    del trade.orders[trade.orders.index(order)]
                    return False
            if order.side == trade.exit_side:
                self.timedout_exit_orders += 1
                # Close exit order and retry exiting on next signal.
                del trade.orders[trade.orders.index(order)]
                return False
        return None

    def check_order_replace(
        self, trade: LocalTrade, order: Order, current_time, row: tuple
    ) -> bool:
        """
        Check if current analyzed entry order has to be replaced and do so.
        If user requested cancellation and there are no filled orders in the trade will
        instruct caller to delete the trade.
        Returns True if the trade should be deleted.
        """
        # only check on new candles for open entry orders
        if current_time > order.order_date_utc:
            is_entry = order.side == trade.entry_side
            requested_rate = strategy_safe_wrapper(
                self.strategy.adjust_order_price, default_retval=order.ft_price
            )(
                trade=trade,  # type: ignore[arg-type]
                order=order,
                pair=trade.pair,
                current_time=current_time,
                proposed_rate=row[OPEN_IDX],
                current_order_rate=order.ft_price,
                entry_tag=trade.enter_tag,
                side=trade.trade_direction,
                is_entry=is_entry,
            )  # default value is current order price

            # cancel existing order whenever a new rate is requested (or None)
            if requested_rate == order.ft_price:
                # assumption: there can't be multiple open entry orders at any given time
                return False
            else:
                del trade.orders[trade.orders.index(order)]
                if is_entry:
                    self.canceled_entry_orders += 1
                else:
                    self.canceled_exit_orders += 1

            # place new order if result was not None
            if requested_rate:
                if is_entry:
                    self._enter_trade(
                        pair=trade.pair,
                        row=row,
                        trade=trade,
                        requested_rate=requested_rate,
                        requested_stake=(order.safe_remaining * order.ft_price / trade.leverage),
                        direction="short" if trade.is_short else "long",
                    )
                    self.replaced_entry_orders += 1
                else:
                    self._exit_trade(
                        trade=trade,
                        sell_row=row,
                        close_rate=requested_rate,
                        amount=order.safe_remaining,
                        exit_reason=order.ft_order_tag,
                    )
                    self.replaced_exit_orders += 1
                # Delete trade if no successful entries happened (if placing the new order failed)
                if not trade.has_open_orders and is_entry and trade.nr_of_successful_entries == 0:
                    return True
            else:
                # assumption: there can't be multiple open entry orders at any given time
                return trade.nr_of_successful_entries == 0
        return False

    def validate_row(
        self, data: dict, pair: str, row_index: int, current_time: datetime
    ) -> tuple | None:
        try:
            # Row is treated as "current incomplete candle".
            # entry / exit signals are shifted by 1 to compensate for this.
            row = data[pair][row_index]
        except IndexError:
            # missing Data for one pair at the end.
            # Warnings for this are shown during data loading
            return None

        # Waits until the time-counter reaches the start of the data for this pair.
        if row[DATE_IDX] > current_time:
            return None
        return row

    def _collate_rejected(self, pair, row):
        """
        Temporarily store rejected signal information for downstream use in backtesting_analysis
        """
        # It could be fun to enable hyperopt mode to write
        # a loss function to reduce rejected signals
        if (
            self.config.get("export", "none") == "signals"
            and self.dataprovider.runmode == RunMode.BACKTEST
        ):
            if pair not in self.rejected_dict:
                self.rejected_dict[pair] = []
            self.rejected_dict[pair].append([row[DATE_IDX], row[ENTER_TAG_IDX]])

    def backtest_loop(
        self,
        row: tuple,
        pair: str,
        current_time: datetime,
        trade_dir: LongShort | None,
        can_enter: bool,
        is_active_for_new_entry: bool, # ADDED for dynamic pairlist entry gating
    ) -> LongShort | None:
        """
        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.

        Backtesting processing for one candle/pair.
        """
        exiting_dir: LongShort | None = None
        if not self._position_stacking and len(LocalTrade.bt_trades_open_pp[pair]) > 0:
            # position_stacking not supported for now.
            exiting_dir = "short" if LocalTrade.bt_trades_open_pp[pair][0].is_short else "long"

        for t in list(LocalTrade.bt_trades_open_pp[pair]):
            # 1. Manage currently open orders of active trades
            if self.manage_open_orders(t, current_time, row):
                # Remove trade (initial open order never filled)
                LocalTrade.remove_bt_trade(t)
                self.wallets.update()

        # 2. Process entries.
        # without positionstacking, we can only have one open trade per pair.
        # max_open_trades must be respected
        # don't open on the last row
        # We only open trades on the main candle, not on detail candles
        # Corrected if condition block with logging
        log_can_enter_val = can_enter
        log_trade_dir_not_none_val = trade_dir is not None
        log_position_cond_val = (self._position_stacking or len(LocalTrade.bt_trades_open_pp[pair]) == 0)
        log_pair_locked_val = PairLocks.is_pair_locked(pair, row[DATE_IDX], trade_dir)
        
        if (
            log_can_enter_val  # can_enter
            and log_trade_dir_not_none_val  # trade_dir is not None
            and is_active_for_new_entry  # Check if pair is in current dynamic whitelist
            and log_position_cond_val  # (self._position_stacking or len(LocalTrade.bt_trades_open_pp[pair]) == 0)
            and not log_pair_locked_val  # not PairLocks.is_pair_locked(...)
        ):
            if self.trade_slot_available(LocalTrade.bt_open_open_trade_count):
                trade = self._enter_trade(pair, row, trade_dir)
                if trade:
                    self.wallets.update()
                else:
                    self._collate_rejected(pair, row)
            else:
                self._collate_rejected(pair, row)

        for trade_to_check_exit in list(LocalTrade.bt_trades_open_pp[pair]):
            # 3. Process entry orders.
            order = trade_to_check_exit.select_order(trade_to_check_exit.entry_side, is_open=True)
            if self._try_close_open_order(order, trade_to_check_exit, current_time, row): # Changed trade to trade_to_check_exit
                self.wallets.update()

            # 4. Create exit orders (if any)
            if trade_to_check_exit.has_open_position: # Changed trade to trade_to_check_exit
                self._check_trade_exit(trade_to_check_exit, row, current_time)  # Changed trade to trade_to_check_exit

            # 5. Process exit orders.
            order = trade_to_check_exit.select_order(trade_to_check_exit.exit_side, is_open=True) # Changed trade to trade_to_check_exit
            if order:
                self._process_exit_order(order, trade_to_check_exit, current_time, row, pair) # Changed trade to trade_to_check_exit

        if exiting_dir and len(LocalTrade.bt_trades_open_pp[pair]) == 0:
            return exiting_dir
        return None

    def get_detail_data(self, pair: str, row: tuple) -> list[tuple] | None:
        """
        Spread into detail data
        """
        current_detail_time: datetime = row[DATE_IDX].to_pydatetime()
        exit_candle_end = current_detail_time + self.timeframe_td
        detail_data = self.detail_data[pair]
        detail_data = detail_data.loc[
            (detail_data["date"] >= current_detail_time) & (detail_data["date"] < exit_candle_end)
        ].copy()

        if len(detail_data) == 0:
            return None
        detail_data.loc[:, "enter_long"] = row[LONG_IDX]
        detail_data.loc[:, "exit_long"] = row[ELONG_IDX]
        detail_data.loc[:, "enter_short"] = row[SHORT_IDX]
        detail_data.loc[:, "exit_short"] = row[ESHORT_IDX]
        detail_data.loc[:, "enter_tag"] = row[ENTER_TAG_IDX]
        detail_data.loc[:, "exit_tag"] = row[EXIT_TAG_IDX]
        return detail_data[HEADERS].values.tolist()

    def _time_generator(self, start_date: datetime, end_date: datetime):
        current_time = start_date + self.timeframe_td
        while current_time <= end_date:
            yield current_time
            current_time += self.timeframe_td

    def _time_generator_det(self, start_date: datetime, end_date: datetime):
        """
        Loop for each detail candle.
        Yields only the start date if no detail timeframe is set.
        """
        if not self.timeframe_detail_td:
            yield start_date, True, False, 0
            return

        current_time = start_date
        i = 0
        while current_time <= end_date:
            yield current_time, i == 0, True, i
            i += 1
            current_time += self.timeframe_detail_td

    def _time_pair_generator_det(self, current_time: datetime, pairs: list[str]):
        for current_time_det, is_first, has_detail, idx in self._time_generator_det(
            current_time, current_time + self.timeframe_td
        ):
            # Pairs that have open trades should be processed first
            new_pairlist = list(dict.fromkeys([t.pair for t in LocalTrade.bt_trades_open] + pairs))
            for pair in new_pairlist:
                yield current_time_det, is_first, has_detail, idx, pair

    def time_pair_generator(
        self,
        start_date: datetime,
        end_date: datetime,
        pairs: list[str],  # Initial whitelist, potentially a superset from all timeline entries
        data: dict[str, list[tuple]],
    ):
        """
        Backtest time and pair generator
        :returns: generator of (current_time, pair, row, is_last_row, trade_dir)
            where is_last_row is a boolean indicating if this is the data end date.
        """
        current_time = start_date + self.timeframe_td
        self.progress.init_step(
            BacktestState.BACKTEST, int((end_date - start_date) / self.timeframe_td)
        )
        # Indexes per pair, so some pairs are allowed to have a missing start.
        indexes: dict = defaultdict(int)
        # Keep track of the last used whitelist to avoid redundant lookups if timeline is sparse
        last_active_whitelist: List[str] = []

        for current_time in self._time_generator(start_date, end_date):
            # Loop for each main candle.
            self.check_abort()
            strategy_safe_wrapper(self.strategy.bot_loop_start, supress_error=True)(
                current_time=current_time
            )
            pair_detail_cache: dict[str, list[tuple]] = {}
            pair_tradedir_cache: dict[str, LongShort | None] = {}
            # Get pairs with open trades at the START of this main candle processing.
            pairs_with_open_trades_at_main_candle_start = [t.pair for t in LocalTrade.bt_trades_open]

            # Determine active whitelist for this current_time from pre-calculated timeline
            active_whitelist_for_main_candle: List[str] = []
            if self.pairlist_timeline:
                relevant_timeline_keys = [
                    ts for ts in self.pairlist_timeline if ts <= current_time
                ]
                if relevant_timeline_keys:
                    latest_ts = max(relevant_timeline_keys)
                    active_whitelist_for_main_candle = self.pairlist_timeline[latest_ts]
                    last_active_whitelist = active_whitelist_for_main_candle
                elif last_active_whitelist: # Fallback to last known if current_time is before any timeline entry
                    active_whitelist_for_main_candle = last_active_whitelist
                else: # Fallback if timeline is empty or current_time is before any entry
                    active_whitelist_for_main_candle = pairs # Original static list from load_bt_data
            else:
                active_whitelist_for_main_candle = pairs # Use initial pairs if no timeline (e.g. StaticPairList)

            # Pairs to process in the detail loop: active list + any pair that had an open trade at the start of this main candle
            pairs_to_process_in_detail = list(dict.fromkeys(
                active_whitelist_for_main_candle + pairs_with_open_trades_at_main_candle_start
            ))

            for current_time_det, is_first, has_detail, idx, pair in self._time_pair_generator_det(
                current_time, pairs_to_process_in_detail
            ):
                is_in_active_whitelist = pair in active_whitelist_for_main_candle
                is_in_open_trades_list = pair in pairs_with_open_trades_at_main_candle_start
                
                # Ensure the pair is either in the active whitelist for this main candle period
                # OR it's a pair that had an open trade at the start of this main candle.
                if not (is_in_active_whitelist or is_in_open_trades_list):
                    continue
                
                trade_dir: LongShort | None = None
                if is_first:
                    # Main candle processing for this pair
                    target_time = current_time # This is the global time step
                    
                    search_start_index = indexes[pair] # Start searching from last known good index for this pair
                    row_index = -1 # Default to -1 if not found

                    # Iterate forwards from search_start_index to find the row matching target_time
                    if pair in data and len(data[pair]) > 0: # Ensure data exists for the pair
                        for i in range(search_start_index, len(data[pair])):
                            candidate_row_date = data[pair][i][DATE_IDX]

                            if candidate_row_date == target_time:
                                row_index = i
                                break
                            elif candidate_row_date > target_time:
                                # We've passed the target_time for this pair's data. No exact match.
                                # The pair might not have a candle at this exact global time.
                                # Set row_index to -1 to indicate no suitable row found for this global time.
                                row_index = -1
                                break
                        # If loop finishes without break and row_index is still -1 (or initial search_start_index was past end)
                        # it means target_time was not found or is beyond the available data for this pair.
                        if row_index != -1 and data[pair][row_index][DATE_IDX] != target_time: # Double check if found row is correct
                             row_index = -1

                    if row_index == -1: # No suitable row found for this pair at current global time
                        # Original logic to advance index if skipped:
                        if pair in data and len(data[pair]) > 0:
                            next_search_idx = search_start_index
                            for i in range(search_start_index, len(data[pair])):
                                if data[pair][i][DATE_IDX] >= target_time: # Note: using target_time here
                                    next_search_idx = i
                                    break
                            else:
                                next_search_idx = len(data[pair])
                            indexes[pair] = next_search_idx
                        continue

                    # Now, row_index should point to the correct candle in data[pair] for current_time
                    # The original validate_row is still useful for its IndexError check,
                    # but the date check `row[DATE_IDX] > current_time` should ideally not be hit
                    # if our search logic is correct and found an exact match.
                    # If `row[DATE_IDX] < current_time` it means our search was flawed.
                    row = self.validate_row(data, pair, row_index, current_time)

                    if not row:
                        # This case should be rare if row_index was found correctly,
                        # unless validate_row has other reasons to fail (e.g. internal IndexError if data[pair] is empty, though checked above)
                        indexes[pair] = row_index + 1 # Still advance index to avoid getting stuck
                        continue
                    
                    # Critical check: Ensure the fetched row's date actually matches the global current_time
                    if row[DATE_IDX] != current_time:
                        indexes[pair] = row_index + 1 # Advance to avoid getting stuck on this mismatched row
                        continue

                    # Update indexes[pair] to point to the *next* candle for this pair, for the next global time step.
                    indexes[pair] = row_index + 1
                    
                    self.dataprovider._set_dataframe_max_index(
                        pair, self.required_startup + row_index
                    )
                    trade_dir = self.check_for_trade_entry(row)
                    pair_tradedir_cache[pair] = trade_dir
                else:
                    # Detail candle - from cache.
                    detail_data_list = pair_detail_cache.get(pair)
                    if detail_data_list is None or len(detail_data_list) <= idx:
                        continue
                    row = detail_data_list[idx]
                    trade_dir = pair_tradedir_cache.get(pair)

                    if self.strategy.ignore_expired_candle(
                        current_time - self.timeframe_td,
                        current_time_det,
                        self.timeframe_secs,
                        trade_dir is not None,
                    ):
                        trade_dir = None

                self.dataprovider._set_dataframe_max_date(current_time_det)

                # Check current open trades for this pair to manage ongoing trades
                current_open_trades_for_pair = LocalTrade.bt_trades_open_pp.get(pair, [])
                pair_has_open_trades_now = len(current_open_trades_for_pair) > 0

                if pair in pairs_with_open_trades_at_main_candle_start and not pair_has_open_trades_now:
                    # This pair had a trade at the start of the main candle, but it closed during detail processing.
                    # It should continue to be processed for this main candle's detail loop if it was in the active_whitelist.
                    # If it wasn't in active_whitelist, it was only processed due to the open trade, so can stop if trade closed.
                    if pair not in active_whitelist_for_main_candle:
                        pass


                if (
                    is_first
                    and (trade_dir is not None or pair_has_open_trades_now)
                    and has_detail
                    and pair not in pair_detail_cache
                    and pair in self.detail_data # Ensure detail data exists for this pair
                    and row # Ensure main row was valid
                ):
                    pair_detail = self.get_detail_data(pair, row)
                    if pair_detail is not None and len(pair_detail) > 0 :
                        pair_detail_cache[pair] = pair_detail
                        if idx < len(pair_detail): # Ensure idx is valid for the new detail data
                           row = pair_detail[idx]
                        else:
                            continue # Should not happen if get_detail_data is correct
                    else: # No detail data for this specific slice or empty
                        if pair_has_open_trades_now: # If trade is open, we must use main candle row
                            pass # Using main candle data
                        else: # No trade open, no detail data, skip
                           continue


                is_last_row = current_time_det == end_date
                yield current_time_det, pair, row, is_last_row, trade_dir, is_in_active_whitelist
            self.progress.increment()

    def backtest(
        self, processed: dict, start_date: datetime, end_date: datetime
    ) -> BacktestContentTypeIcomplete:
        """
        Implement backtesting functionality

        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.
        Of course try to not have ugly code. By some accessor are sometime slower than functions.
        Avoid extensive logging in this method and functions it calls.

        :param processed: a processed dictionary with format {pair, data}, which gets cleared to
        optimize memory usage!
        :param start_date: backtesting timerange start datetime
        :param end_date: backtesting timerange end datetime
        :return: DataFrame with trades (results of backtesting)
        """
        self.prepare_backtest(self.enable_protections)
        # Ensure wallets are up-to-date (important for --strategy-list)
        self.wallets.update()
        # Use dict of lists with data for performance
        # (looping lists is a lot faster than pandas DataFrames)
        data: dict = self._get_ohlcv_as_lists(processed)

        # Loop timerange and get candle for each pair at that point in time
        for (
            current_time,
            pair,
            row,
            is_last_row,
            trade_dir,
            is_active_for_new_entry, # Unpack the new flag
        ) in self.time_pair_generator(start_date, end_date, list(data.keys()), data):
            if not self._can_short or trade_dir is None:
                # No need to reverse position if shorting is disabled or there's no new signal
                self.backtest_loop(row, pair, current_time, trade_dir, not is_last_row, is_active_for_new_entry)
            else:
                # Conditionally call backtest_loop a 2nd time if shorting is enabled,
                # a position closed and a new signal in the other direction is available.

                for _ in (0, 1):
                    a = self.backtest_loop(row, pair, current_time, trade_dir, not is_last_row, is_active_for_new_entry)
                    if not a or a == trade_dir:
                        # the trade didn't close or position change is in the same direction
                        break

        self.handle_left_open(LocalTrade.bt_trades_open_pp, data=data)
        self.wallets.update()

        results = trade_list_to_dataframe(LocalTrade.bt_trades)
        return {
            "results": results,
            "config": self.strategy.config,
            "locks": PairLocks.get_all_locks(),
            "rejected_signals": self.rejected_trades,
            "timedout_entry_orders": self.timedout_entry_orders,
            "timedout_exit_orders": self.timedout_exit_orders,
            "canceled_trade_entries": self.canceled_trade_entries,
            "canceled_entry_orders": self.canceled_entry_orders,
            "replaced_entry_orders": self.replaced_entry_orders,
            "final_balance": self.wallets.get_total(self.strategy.config["stake_currency"]),
        }

    def backtest_one_strategy(
        self, strat: IStrategy, data: dict[str, DataFrame], timerange: TimeRange
    ):
        self.progress.init_step(BacktestState.ANALYZE, 0)
        strategy_name = strat.get_strategy_name()
        logger.info(f"Running backtesting for Strategy {strategy_name}")
        backtest_start_time = dt_now()
        self._set_strategy(strat)

        # need to reprocess data every time to populate signals
        preprocessed = self.strategy.advise_all_indicators(data)

        # Trim startup period from analyzed dataframe
        # This only used to determine if trimming would result in an empty dataframe
        preprocessed_tmp = trim_dataframes(preprocessed, timerange, self.required_startup)

        if not preprocessed_tmp:
            raise OperationalException("No data left after adjusting for startup candles.")

        # Use preprocessed_tmp for date generation (the trimmed dataframe).
        # Backtesting will re-trim the dataframes after entry/exit signal generation.
        min_date, max_date = history.get_timerange(preprocessed_tmp)
        logger.info(
            f"Backtesting with data from {min_date.strftime(DATETIME_PRINT_FORMAT)} "
            f"up to {max_date.strftime(DATETIME_PRINT_FORMAT)} "
            f"({(max_date - min_date).days} days)."
        )
        # Execute backtest and store results
        results = self.backtest(
            processed=preprocessed,
            start_date=min_date,
            end_date=max_date,
        )
        backtest_end_time = dt_now()
        results.update(
            {
                "run_id": self.run_ids.get(strategy_name, ""),
                "backtest_start_time": int(backtest_start_time.timestamp()),
                "backtest_end_time": int(backtest_end_time.timestamp()),
            }
        )
        self.all_bt_content[strategy_name] = results

        if (
            self.config.get("export", "none") == "signals"
            and self.dataprovider.runmode == RunMode.BACKTEST
        ):
            signals = generate_trade_signal_candles(preprocessed_tmp, results, "open_date")
            rejected = generate_rejected_signals(preprocessed_tmp, self.rejected_dict)
            exited = generate_trade_signal_candles(preprocessed_tmp, results, "close_date")

            self.analysis_results["signals"][strategy_name] = signals
            self.analysis_results["rejected"][strategy_name] = rejected
            self.analysis_results["exited"][strategy_name] = exited

        return min_date, max_date

    def _get_min_cached_backtest_date(self):
        min_backtest_date = None
        backtest_cache_age = self.config.get("backtest_cache", constants.BACKTEST_CACHE_DEFAULT)
        if self.timerange.stopts == 0 or self.timerange.stopdt > dt_now():
            logger.warning("Backtest result caching disabled due to use of open-ended timerange.")
        elif backtest_cache_age == "day":
            min_backtest_date = dt_now() - timedelta(days=1)
        elif backtest_cache_age == "week":
            min_backtest_date = dt_now() - timedelta(weeks=1)
        elif backtest_cache_age == "month":
            min_backtest_date = dt_now() - timedelta(weeks=4)
        return min_backtest_date

    def load_prior_backtest(self):
        self.run_ids = {
            strategy.get_strategy_name(): get_strategy_run_id(strategy)
            for strategy in self.strategylist
        }

        # Load previous result that will be updated incrementally.
        # This can be circumvented in certain instances in combination with downloading more data
        min_backtest_date = self._get_min_cached_backtest_date()
        if min_backtest_date is not None:
            self.results = find_existing_backtest_stats(
                self.config["user_data_dir"] / "backtest_results", self.run_ids, min_backtest_date
            )

    def start(self) -> None:
        """
        Run backtesting end-to-end
        """
        data: dict[str, DataFrame] = {}
        # config_timerange will be determined by the final load_bt_data call.
        # self.timerange is initialized in __init__ and holds the overall configured range.
        # It's used by load_bt_data internally.

        # Step 1: Determine the true operational date range.
        # This requires a preliminary data load using the initial whitelist to find data boundaries.
        logger.info("Performing preliminary data load to determine actual date boundaries for pairlist timeline...")
        
        # Store original whitelist to ensure the preliminary load uses it, and it can be restored if needed.
        # self.pairlists is initialized in __init__ and refresh_pairlist is called there.
        initial_whitelist_for_daterange = self.pairlists.whitelist[:]
        
        # Use a temporary PairListManager if we need to absolutely ensure no side-effects
        # on self.pairlists from this preliminary load, though load_bt_data uses the passed pairs.
        # For this step, we primarily need load_bt_data to respect self.timerange and give us date boundaries.
        # The pairs used for this initial load are just to get any data to establish the range.
        
        # Explicitly pass the initial_whitelist_for_daterange for the preliminary load.
        # The returned timerange from this call is specific to this load.
        temp_data_for_daterange, _ = self.load_bt_data(pairs_to_load=initial_whitelist_for_daterange)

        if not temp_data_for_daterange:
            # If initial whitelist is dynamic and empty, this might happen.
            # Or if no data for any initial pairs in the configured range.
            # Try to use self.timerange directly for timeline calculation if possible.
            if not initial_whitelist_for_daterange and self.config.get('pairlists', [{}])[0].get('method') != 'StaticPairList':
                 logger.warning("Initial whitelist was empty. Using configured timerange for pairlist timeline calculation.")
                 min_date_for_timeline, max_date_for_timeline = self.timerange.startdt, self.timerange.stopdt
                 if max_date_for_timeline == datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc) and self.timerange.stopts == 0: # Open-ended
                     raise OperationalException(
                         "Open-ended timerange with an empty initial whitelist is not supported "
                         "for dynamic pairlist pre-calculation without a data-derived max_date."
                     )
            else:
                raise OperationalException(
                    "No data found for any pair in initial whitelist. "
                    "Cannot determine date range for dynamic pairlist pre-calculation."
                )
        else:
            min_date_for_timeline, max_date_for_timeline = history.get_timerange(temp_data_for_daterange)
        
        del temp_data_for_daterange # Free memory
        logger.info(
            f"Effective data range for pairlist pre-calculation: "
            f"{min_date_for_timeline.strftime(DATETIME_PRINT_FORMAT)} to "
            f"{max_date_for_timeline.strftime(DATETIME_PRINT_FORMAT)}"
        )

        # Step 2: Pre-calculate the pairlist timeline using these determined boundaries
        self._precalculate_pairlist_timeline(min_date_for_timeline, max_date_for_timeline)

        # Step 3: Construct the superset of all pairs from the timeline
        super_set_of_all_pairs = set()
        if self.pairlist_timeline:
            for pairs_in_slot in self.pairlist_timeline.values():
                super_set_of_all_pairs.update(pairs_in_slot)
        
        # Fallback if timeline is empty or yields no pairs (e.g., StaticPairList was used)
        if not super_set_of_all_pairs:
            logger.warning("Pairlist timeline did not yield any pairs. "
                           "Using initial whitelist for the main data load.")
            super_set_of_all_pairs.update(initial_whitelist_for_daterange)

        if not super_set_of_all_pairs:
            raise OperationalException(
                "No pairs found to backtest after pairlist timeline generation and fallback."
            )

        logger.info(f"Superset of all pairs for main backtest data load: {len(super_set_of_all_pairs)}.")

        # Step 4: Load all necessary data using the superset.
        # The PairListManager's whitelist property has no setter.
        # Instead, we pass the super_set_of_all_pairs directly to load_bt_data
        # via the 'pairs_list' argument, which load_bt_data is modified to handle.
        # self.pairlists.whitelist = list(super_set_of_all_pairs) # This line caused the AttributeError
        
        # The logger below would also fail if the above line was active and failed.
        # It's kept for context but relies on load_bt_data using the superset correctly.
        logger.info(f"Loading main backtest data for the superset of {len(super_set_of_all_pairs)} pairs...")
        # This call to load_bt_data() is the definitive one for the backtest.
        # It must use the super_set_of_all_pairs.
        # The returned config_timerange will be based on the data actually loaded for the superset.
        logger.info(f"Attempting main data load for super_set_of_all_pairs ({len(super_set_of_all_pairs)} pairs)...")
        data, config_timerange = self.load_bt_data(pairs_to_load=list(super_set_of_all_pairs))
        
        if not data:
            # This means no data could be loaded for ANY pair in the super_set_of_all_pairs.
            raise OperationalException(
                f"No data successfully loaded for any pair in the superset: {super_set_of_all_pairs}. "
                "Check data availability for these pairs and timeframe within the configured range."
            )

        # These are now the definitive dates for the overall backtest based on the superset and timerange.
        actual_min_date, actual_max_date = history.get_timerange(data)
        logger.info(
            f"Main dataload for superset complete. Final data range for backtest: "
            f"{actual_min_date.strftime(DATETIME_PRINT_FORMAT)} to "
            f"{actual_max_date.strftime(DATETIME_PRINT_FORMAT)}"
        )
        
        self.load_prior_backtest()
        # config_timerange from the final load_bt_data is the one to use.
 
        for strat in self.strategylist:
            if self.results and strat.get_strategy_name() in self.results["strategy"]:
                # When previous result hash matches - reuse that result and skip backtesting.
                logger.info(f"Reusing result of previous backtest for {strat.get_strategy_name()}")
                continue
            # backtest_one_strategy uses its own min_date, max_date from processed data
            # and the config_timerange for other purposes.
            # actual_min_date and actual_max_date are now correctly set from the superset load.
            # These will be used by generate_backtest_stats later.
            self.backtest_one_strategy(strat, data, config_timerange)

        # Update old results with new ones.
        # generate_backtest_stats needs the min_date, max_date from the superset data load.
        if len(self.all_bt_content) > 0: # Check if any strategy was actually backtested
            min_date, max_date = actual_min_date, actual_max_date # Use the definitive dates
        if len(self.all_bt_content) > 0:
            results = generate_backtest_stats(
                data, self.all_bt_content, min_date=min_date, max_date=max_date
            )
            if self.results:
                self.results["metadata"].update(results["metadata"])
                self.results["strategy"].update(results["strategy"])
                self.results["strategy_comparison"].extend(results["strategy_comparison"])
            else:
                self.results = results
            dt_appendix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if self.config.get("export", "none") in ("trades", "signals"):
                combined_res = combined_dataframes_with_rel_mean(data, min_date, max_date)
                store_backtest_results(
                    self.config,
                    self.results,
                    dt_appendix,
                    market_change_data=combined_res,
                    analysis_results=self.analysis_results,
                    strategy_files={s.get_strategy_name(): s.__file__ for s in self.strategylist},
                )

        # Results may be mixed up now. Sort them so they follow --strategy-list order.
        if "strategy_list" in self.config and len(self.results) > 0:
            self.results["strategy_comparison"] = sorted(
                self.results["strategy_comparison"],
                key=lambda c: self.config["strategy_list"].index(c["key"]),
            )
            self.results["strategy"] = dict(
                sorted(
                    self.results["strategy"].items(),
                    key=lambda kv: self.config["strategy_list"].index(kv[0]),
                )
            )

        if len(self.strategylist) > 0:
            # Show backtest results
            show_backtest_results(self.config, self.results)
