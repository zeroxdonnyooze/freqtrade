# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from pandas import DataFrame
# --------------------------------

# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from freqtrade.persistence import Trade, Order
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_minutes # Corrected import path
from datetime import datetime, timedelta # Keep timedelta if used for time-based logic
from freqtrade.strategy import (IStrategy, CategoricalParameter, DecimalParameter,
                                 IntParameter, RealParameter, informative)
from freqtrade.enums import ExitType, ExitCheckTuple # Added for custom stoploss override
import logging # For custom logging if needed
import pandas as pd # For DataFrame type hinting and operations
from typing import Union, Optional, Dict, Any # Add Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

class RSIPNRBounce(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - The prototypes for the methods:
        - populate_indicators
        - populate_entry_trend
        - populate_exit_trend
        - populate_buy_trend (deprecated, use populate_entry_trend)
        - populate_sell_trend (deprecated, use populate_exit_trend)
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    def __init__(self, config: dict):
        super().__init__(config)
        self.so_custom_stakes: Dict[int, Dict[str, float]] = {}
        # self.so_custom_stakes is initialized here
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -1
    # Trailing stoploss
    trailing_stop = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.02
    # trailing_only_offset_is_reached = False

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    can_short = True # Enable short selling
    position_adjustment_enable = True # Enable Safety Orders

    # Number of candles the strategy requires before producing valid signals
    startup_trend_candlestick_count: int = 1400 # Adjusted for 14-day ATR on 15m timeframe (14 * 96 = 1344)

    # --- Strategy Specific Variables ---
    # RSI
    rsi_period = IntParameter(7, 21, default=14, space="buy")

    # PNR
    pnr_lookback = IntParameter(100, 500, default=150, space="buy", optimize=True, load=True) # Shared

    # Long PNR Thresholds
    long_pnr_lower_threshold = IntParameter(1, 30, default=10, space="buy", optimize=True, load=True)
    long_pnr_upper_threshold = IntParameter(70, 99, default=90, space="sell", optimize=True, load=True) # For long exits

    # Short PNR Thresholds
    short_pnr_lower_threshold = IntParameter(1, 30, default=10, space="sell", optimize=True, load=True) # For short exits
    short_pnr_upper_threshold = IntParameter(70, 99, default=90, space="sell", optimize=True, load=True) # For short entries

    # PNR Crossing Direction Parameters
    long_entry_pnr_cross_direction = CategoricalParameter(['below', 'above'], default='above', space='buy', optimize=True, load=True)
    short_entry_pnr_cross_direction = CategoricalParameter(['below', 'above'], default='above', space='sell', optimize=True, load=True)

    long_exit_pnr_cross_direction = CategoricalParameter(['above', 'below'], default='below', space='sell', optimize=True, load=True)
    short_exit_pnr_cross_direction = CategoricalParameter(['above', 'below'], default='below', space='buy', optimize=True, load=True)

    # ATR for Safety Orders (SO) - Period is shared
    atr_period = IntParameter(5, 20, default=14, space="buy", optimize=True, load=True) # This is for the informative ATR

    # Max number of safety orders
    long_max_safety_orders = IntParameter(0, 10, default=3, space="buy", optimize=True, load=True)
    short_max_safety_orders = IntParameter(0, 10, default=3, space="sell", optimize=True, load=True)

    # Desired leverage for trades
    desired_leverage = RealParameter(1.0, 100.0, default=10.0, space="buy", optimize=True, load=True) # Changed optimize to False

    # Base order sizing: target position value as a percentage of total equity
    base_order_position_value_percentage = RealParameter(0.05, 1.0, default=0.1, space="buy", optimize=True, load=True) # e.g. 0.1 means 10% of equity as position value

    # Safety Order (SO) trigger parameters
    long_so_atr_distance_multiplier = RealParameter(0.5, 3.0, default=0.5, space="buy", optimize=True, load=True)
    short_so_atr_distance_multiplier = RealParameter(0.5, 3.0, default=0.5, space="sell", optimize=True, load=True)
    
    # SO1 sizing: target position value as a percentage of total equity
    so1_position_value_percentage = RealParameter(0.05, 1.0, default=0.1, space="buy", optimize=True, load=True)

    # SO2-N sizing: Multiplier for SO(N)_cost / SO(N-1)_cost
    so_subsequent_size_multiplier = RealParameter(0.5, 3.0, default=1.5, space="buy", optimize=True, load=True)

    allow_long_trades = CategoricalParameter([True, False], default=True, space="buy", optimize=True, load=True)
    allow_short_trades = CategoricalParameter([True, False], default=True, space="buy", optimize=True, load=True) # Changed optimize to False

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = {
        'main_plot': {},
        'subplots': {
            "RSI_Bands": {
                'rsi': {'color': 'blue'},
                'rsi_lower_band': {'color': 'green', 'linestyle': '--'},
                'rsi_upper_band': {'color': 'red', 'linestyle': '--'}
            }
            # Note: 'atr' was previously in the "RSI" subplot but is not calculated in the current populate_indicators.
            # If ATR is needed for plotting, it should be calculated in populate_indicators and added here.
        }
    }

    # Removed informative daily ATR method. ATR for SOs will be calculated in populate_indicators.

    def informative_pairs(self):
        # Define an informative pair for 1d data to calculate a stable long-period ATR
        # This will be used for SO placement.
        # We will use the same pair as the traded pair.
        # The decorator @informative on populate_indicators_1d handles merging this back.
        # informative_pairs can return an empty list if all informatives are handled by decorators.
        return []

    # Method to populate indicators for the 1d informative timeframe
    # The decorator will automatically merge the result back to the strategy's main timeframe
    # The result will be prefixed with e.g. "1d_"
    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate ATR on the 1d timeframe.
        # 14 days * (24 hours / 24 hours_per_candle) = 14 * 1 = 14 periods for 1d candles
        # The result will be named e.g. 'atr_1d' when merged.
        df_len_1d = len(dataframe)
        df_start_1d = dataframe['date'].iloc[0] if df_len_1d > 0 else "N/A"
        df_end_1d = dataframe['date'].iloc[-1] if df_len_1d > 0 else "N/A"
        required_periods = 14 # For 14-day ATR on 1d timeframe
        
        if len(dataframe) < required_periods:
            # Optionally, ensure 'atr_1d' column exists with NaNs if other parts of the strategy expect it
            # dataframe['atr_inf'] = pd.NA # Or np.nan, depending on desired handling downstream
            return dataframe # Return without calculating ATR if not enough data

        # Calculate ATR. ta.ATR requires high, low, close.
        atr_series = ta.ATR(dataframe, timeperiod=required_periods)
        
        # Create a new DataFrame with only 'date' and the new 'atr_inf'
        # This makes it explicit to the @informative decorator what new columns are being provided.
        
        # Ensure the input dataframe's index is not causing issues if it's not a simple range index
        temp_df = dataframe.reset_index(drop=True)
        
        informative_df = pd.DataFrame({'date': temp_df['date']})
        # Corrected: ta.ATR needs a DataFrame with high, low, close columns, or those series explicitly.
        # Assuming temp_df (which is derived from the input 'dataframe') has these.
        informative_df['atr_inf'] = ta.ATR(temp_df[['high', 'low', 'close']], timeperiod=required_periods)
        
        # Explicitly drop rows where atr_inf might be NaN initially due to ATR period,
        # though merge_asof should handle this with forward fill. This is more for clean debugging.
        # informative_df.dropna(subset=['atr_inf'], inplace=True) # Optional: consider if this affects merge
        
        if not informative_df.empty and 'atr_inf' in informative_df.columns:
            first_valid_atr_index = informative_df['atr_inf'].first_valid_index()
            if first_valid_atr_index is not None:
                first_valid_atr_date = informative_df.loc[first_valid_atr_index, 'date']
        elif informative_df.empty: # Should not happen if input dataframe wasn't empty
            pass
        else:
            pass
        return informative_df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # Shared lookback period
        lookback_period = self.pnr_lookback.value

        # Calculate bands on a shifted RSI series to avoid using current candle's RSI
        # in band calculation for that same candle. This makes cross signals more responsive.
        shifted_rsi = dataframe['rsi'].shift(1)

        if lookback_period > 0 and len(dataframe) >= lookback_period + 1: # +1 needed because of .shift(1)
            # Long PNR Bands
            long_lower_quantile = self.long_pnr_lower_threshold.value / 100.0
            long_upper_quantile = self.long_pnr_upper_threshold.value / 100.0

            dataframe['long_rsi_lower_band'] = shifted_rsi.rolling(
                window=lookback_period,
                min_periods=lookback_period
            ).quantile(long_lower_quantile)
            dataframe['long_rsi_upper_band'] = shifted_rsi.rolling(
                window=lookback_period,
                min_periods=lookback_period
            ).quantile(long_upper_quantile)

            # Short PNR Bands
            short_lower_quantile = self.short_pnr_lower_threshold.value / 100.0
            short_upper_quantile = self.short_pnr_upper_threshold.value / 100.0

            dataframe['short_rsi_lower_band'] = shifted_rsi.rolling(
                window=lookback_period,
                min_periods=lookback_period
            ).quantile(short_lower_quantile)
            dataframe['short_rsi_upper_band'] = shifted_rsi.rolling(
                window=lookback_period,
                min_periods=lookback_period
            ).quantile(short_upper_quantile)
            
        else:
            # Not enough data, fill all with NaN
            dataframe['long_rsi_lower_band'] = np.nan
            dataframe['long_rsi_upper_band'] = np.nan
            dataframe['short_rsi_lower_band'] = np.nan
            dataframe['short_rsi_upper_band'] = np.nan
            
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if 'enter_long' not in dataframe.columns: dataframe['enter_long'] = 0
        if 'enter_short' not in dataframe.columns: dataframe['enter_short'] = 0
        if 'enter_tag' not in dataframe.columns: dataframe['enter_tag'] = pd.NA

        rsi_series = dataframe['rsi']
        
        # Get side-specific bands
        long_rsi_lower_band = dataframe.get('long_rsi_lower_band')
        # long_rsi_upper_band = dataframe.get('long_rsi_upper_band') # Not used for long entry
        # short_rsi_lower_band = dataframe.get('short_rsi_lower_band') # Not used for short entry
        short_rsi_upper_band = dataframe.get('short_rsi_upper_band')

        dataframe['enter_long'] = 0  # Reset signals each pass
        dataframe['enter_short'] = 0

        # --- LONG ENTRIES ---
        if self.allow_long_trades.value:
            if long_rsi_lower_band is not None:
                long_cross_condition_active = False
                current_long_entry_direction = self.long_entry_pnr_cross_direction.value
                if current_long_entry_direction == 'below':
                    long_cross_condition = qtpylib.crossed_below(rsi_series, long_rsi_lower_band)
                    long_tag_direction = 'below'
                    long_cross_condition_active = True
                elif current_long_entry_direction == 'above':
                    long_cross_condition = qtpylib.crossed_above(rsi_series, long_rsi_lower_band)
                    long_tag_direction = 'above'
                    long_cross_condition_active = True
                
                if long_cross_condition_active:
                    dataframe.loc[
                        (rsi_series.notna()) & (long_rsi_lower_band.notna()) & long_cross_condition,
                        ['enter_long', 'enter_tag']
                    ] = (1, f'long_rsi_vs_lower_band({self.long_pnr_lower_threshold.value}%)_crossed_{long_tag_direction}')
            else:
                # long_rsi_lower_band is None, cannot evaluate long entries
                pass

        # --- SHORT ENTRIES ---
        if self.can_short and self.allow_short_trades.value:
            if short_rsi_upper_band is not None:
                short_cross_condition_active = False
                current_short_entry_direction = self.short_entry_pnr_cross_direction.value
                if current_short_entry_direction == 'below':
                    short_cross_condition = qtpylib.crossed_below(rsi_series, short_rsi_upper_band)
                    short_tag_direction = 'below'
                    short_cross_condition_active = True
                elif current_short_entry_direction == 'above':
                    short_cross_condition = qtpylib.crossed_above(rsi_series, short_rsi_upper_band)
                    short_tag_direction = 'above'
                    short_cross_condition_active = True

                if short_cross_condition_active:
                    dataframe.loc[
                        (rsi_series.notna()) & (short_rsi_upper_band.notna()) & short_cross_condition & (dataframe['enter_long'] == 0),
                        ['enter_short', 'enter_tag']
                    ] = (1, f'short_rsi_vs_upper_band({self.short_pnr_upper_threshold.value}%)_crossed_{short_tag_direction}')
            else:
                # short_rsi_upper_band is None, cannot evaluate short entries
                pass
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if 'exit_long' not in dataframe.columns: dataframe['exit_long'] = 0
        if 'exit_short' not in dataframe.columns: dataframe['exit_short'] = 0
        if 'exit_tag' not in dataframe.columns: dataframe['exit_tag'] = pd.NA

        pair = metadata['pair']
        # Ensure dataframe is not empty before trying to access iloc[-1]
        if dataframe.empty:
            return dataframe
        current_time = dataframe['date'].iloc[-1]

        rsi_series = dataframe['rsi']
        
        # Get side-specific bands
        long_rsi_upper_band = dataframe.get('long_rsi_upper_band') # For long exits
        short_rsi_lower_band = dataframe.get('short_rsi_lower_band') # For short exits

        dataframe['exit_long'] = 0 # Reset signals each pass
        dataframe['exit_short'] = 0
        
        # --- LONG EXITS ---
        if self.allow_long_trades.value:
            if long_rsi_upper_band is not None and not rsi_series.empty and not long_rsi_upper_band.empty:
                effective_long_exit_direction = self.long_exit_pnr_cross_direction.value
                crossed_series_long = pd.Series([False] * len(dataframe), index=dataframe.index)

                if effective_long_exit_direction == 'above':
                    crossed_series_long = qtpylib.crossed_above(rsi_series, long_rsi_upper_band)
                elif effective_long_exit_direction == 'below':
                    crossed_series_long = qtpylib.crossed_below(rsi_series, long_rsi_upper_band)
                
                if crossed_series_long.any():
                    exit_tag_val = f'long_rsi_vs_upper_band({self.long_pnr_upper_threshold.value}%)_exit_crossed_{effective_long_exit_direction}'
                    dataframe.loc[crossed_series_long, 'exit_long'] = 1
                    dataframe.loc[crossed_series_long, 'exit_tag'] = exit_tag_val
            else:
                # long_rsi_upper_band is None or series empty, cannot evaluate long exits
                pass

        # --- SHORT EXITS ---
        if self.can_short and self.allow_short_trades.value:
            if short_rsi_lower_band is not None and not rsi_series.empty and not short_rsi_lower_band.empty:
                effective_short_exit_direction = self.short_exit_pnr_cross_direction.value
                crossed_series_short = pd.Series([False] * len(dataframe), index=dataframe.index)

                if effective_short_exit_direction == 'above':
                    crossed_series_short = qtpylib.crossed_above(rsi_series, short_rsi_lower_band)
                elif effective_short_exit_direction == 'below':
                    crossed_series_short = qtpylib.crossed_below(rsi_series, short_rsi_lower_band)
                
                if crossed_series_short.any():
                    exit_tag_val = f'short_rsi_vs_lower_band({self.short_pnr_lower_threshold.value}%)_exit_crossed_{effective_short_exit_direction}'
                    # Apply only if no conflicting long exit is set on the same candles
                    condition_short_apply = crossed_series_short & (dataframe['exit_long'] == 0)
                    dataframe.loc[condition_short_apply, 'exit_short'] = 1
                    dataframe.loc[condition_short_apply, 'exit_tag'] = exit_tag_val
            else:
                # short_rsi_lower_band is None or series empty, cannot evaluate short exits
                pass
        
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.
        Called by Freqtrade core.
        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot (usually from exchange default).
        :param max_leverage: Max leverage allowed on this pair by the exchange.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        # Ensure leverage is not zero to prevent division by zero errors later
        chosen_leverage = self.desired_leverage.value
        if chosen_leverage < 1.0:
            logger.warning(f"Desired leverage {chosen_leverage} is less than 1.0. Setting to 1.0.")
            chosen_leverage = 1.0
        
        final_leverage = min(chosen_leverage, max_leverage)
        if final_leverage < 1.0: # Should not happen if max_leverage is always >= 1
             final_leverage = 1.0
        return final_leverage

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            entry_tag: Optional[str], side: str,
                            leverage: float, # Added leverage parameter
                            **kwargs) -> float:
        if self.wallets is None:
            logger.warning("Wallets object is not available. Cannot calculate equity-based stake.")
            return proposed_stake
        
        if leverage == 0: # Should be caught by leverage() callback returning >= 1
            logger.error(f"Leverage is zero for pair {pair}. Cannot calculate stake for target position value. Returning proposed_stake.")
            return proposed_stake

        tradable_balance_ratio = self.config.get('tradable_balance_ratio', 1.0)
        if tradable_balance_ratio == 0:
            logger.warning("tradable_balance_ratio is 0. Cannot calculate equity-based stake.")
            return proposed_stake
        
        # Calculate total equity available for the bot
        # Ensure self.wallets.get_total_stake_amount() is available and valid
        try:
            current_balance = self.wallets.get_total_stake_amount()
        except Exception as e:
            logger.warning(f"Could not get total stake amount from wallets: {e}. Using dry_run_wallet if available.")
            # Fallback for backtesting or if wallet info is somehow unavailable
            current_balance = self.config.get('dry_run_wallet', proposed_stake * leverage * 5) # Estimate from proposed_stake

        total_equity_for_bot = current_balance / tradable_balance_ratio

        # Target position value based on equity percentage
        target_position_value = total_equity_for_bot * self.base_order_position_value_percentage.value

        # Calculate the required cost (stake) to achieve this position value with the given leverage
        required_cost = target_position_value / leverage

        if required_cost <= 0:
            logger.warning(f"Calculated required_cost is {required_cost:.8f} for {pair} with target position value {target_position_value:.2f} and leverage {leverage:.2f}. Using proposed_stake.")
            # Fallback to proposed_stake or a fraction of max_stake if proposed_stake is too large or zero
            if proposed_stake > 0 and (max_stake is None or proposed_stake <= max_stake):
                 return proposed_stake
            elif min_stake is not None:
                 return min_stake
            else: # Absolute fallback if everything else fails
                 return max_stake / 10 if max_stake else 10 # Arbitrary small stake

        # Apply exchange limits (min_stake and max_stake)
        # max_stake here refers to the maximum cost the bot is allowed to use for one trade from its balance
        # min_stake is the minimum cost required by the exchange
        
        final_cost = required_cost
        if min_stake is not None:
            final_cost = max(final_cost, min_stake)
        
        # max_stake is the upper limit of what we can afford or are allowed by config (e.g. if it's not 'unlimited')
        # It's the cost, not the position value.
        final_cost = min(final_cost, max_stake)
        
        return final_cost

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float,
                            min_stake: Optional[float], max_stake: float,
                            current_entry_rate: float, current_exit_rate: float,
                            current_entry_profit: float, current_exit_profit: float,
                            **kwargs) -> Optional[Union[float, Dict[str, Any]]]:
        """
        Custom trade adjustment logic, used for placing safety orders.
        SO1: Sized based on so1_position_value_percentage of total account equity. The parameter value (e.g. 0.1 for 10%) directly multiplies equity to get target position value.
        SO2-N: Sized based on so_subsequent_size_multiplier * margin_cost_of_previous_SO.
        Triggers: Evenly spaced from initial entry price using ATR.
        """

        if trade.id not in self.so_custom_stakes:
            self.so_custom_stakes[trade.id] = {}
        trade_so_stakes = self.so_custom_stakes[trade.id]

        if current_profit > 0.005:  # Don't add to winning trades
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            logger.warning(f"Dataframe not available for {trade.pair} to adjust position.")
            return None
        

        latest_candle = dataframe.iloc[-1]
        
        # Use 'atr_4h' from the informative pair calculated in populate_indicators_4h
        # This column should be prefixed by the informative timeframe, e.g., '4h_atr'
        # However, the @informative decorator by default renames the column to match the target if it's unique.
        # Let's assume it will be 'atr_4h' or check common prefixes if needed.
        # For safety, let's check for common prefixed names if 'atr_4h' isn't directly there.
        
        # The @informative('1d') decorator on populate_indicators_1d (where 'atr_inf' is set)
        # by default renames columns to {column}_{timeframe}, so 'atr_inf' becomes 'atr_inf_1d'.
        atr_column_name = 'atr_inf_1d' # Corrected based on default fmt
        
        if atr_column_name not in latest_candle or pd.isna(latest_candle[atr_column_name]):
            return None

        current_atr = latest_candle[atr_column_name]
        if current_atr == 0:
            logger.warning(f"ATR is zero for {trade.pair} at {current_time}. Cannot place SO.")
            return None

        num_successful_entries = trade.nr_of_successful_entries
        num_existing_sos = num_successful_entries - 1

        # Determine side-specific SO parameters
        if not trade.is_short:
            current_max_safety_orders = self.long_max_safety_orders.value
            current_so_atr_distance_multiplier = self.long_so_atr_distance_multiplier.value
        else:
            current_max_safety_orders = self.short_max_safety_orders.value
            current_so_atr_distance_multiplier = self.short_so_atr_distance_multiplier.value

        if num_existing_sos >= current_max_safety_orders:
            return None  # Max safety orders reached for this side

        next_so_number = num_existing_sos + 1
        
        current_trade_leverage = trade.leverage
        if not current_trade_leverage or current_trade_leverage < 1.0:
            logger.error(f"Trade leverage ({current_trade_leverage}) is invalid for trade {trade.id} ({trade.pair}). Cannot calculate SO cost.")
            current_trade_leverage = self.leverage(trade.pair, current_time, current_rate, 1.0, trade.max_leverage or self.desired_leverage.value, trade.enter_tag, trade.trade_direction)
            if not current_trade_leverage or current_trade_leverage < 1.0:
                 logger.error(f"Fallback leverage from callback is also invalid ({current_trade_leverage}). Aborting SO for trade {trade.id}.")
                 return None

        filled_entry_orders = sorted(
            [o for o in trade.orders if o.status == 'closed' and o.filled == o.amount and o.ft_order_side == trade.entry_side],
            key=lambda o: o.order_filled_date or datetime.min
        )

        if not filled_entry_orders:
            logger.warning(f"No filled entry orders found for trade {trade.id} to base SO #{next_so_number} upon.")
            return None
        
        initial_entry_order = filled_entry_orders[0]
        initial_entry_fill_price = initial_entry_order.average or trade.open_rate
        initial_position_value = initial_entry_order.cost

        if initial_entry_fill_price is None or initial_entry_fill_price <=0:
            logger.warning(f"Initial entry order fill price is invalid for {trade.pair} for trade {trade.id}.")
            return None
        if initial_position_value is None or initial_position_value <=0:
            logger.warning(f"Initial position value (from initial_entry_order.cost) is invalid for {trade.pair} for trade {trade.id}.")
            return None
        
        calculated_so_cost: float = 0.0

        if next_so_number == 1:
            if self.wallets is None:
                logger.warning("Wallets object not available for SO1. Cannot calculate equity-based stake.")
                return None
            
            tradable_balance_ratio = self.config.get('tradable_balance_ratio', 1.0)
            if tradable_balance_ratio == 0:
                logger.warning("tradable_balance_ratio is 0 for SO1. Cannot calculate equity-based stake.")
                return None

            try:
                current_balance = self.wallets.get_total_stake_amount()
            except Exception as e:
                logger.warning(f"Could not get total stake amount from wallets for SO1: {e}. Using dry_run_wallet if available.")
                initial_order_cost_for_fallback = initial_entry_order.cost or initial_entry_order.stake_amount or 100
                # Use current_max_safety_orders for the fallback estimation
                current_balance = self.config.get('dry_run_wallet', initial_order_cost_for_fallback * current_trade_leverage * (current_max_safety_orders +1) * 2 )

            total_equity_for_bot = current_balance / tradable_balance_ratio
            target_so1_position_value = total_equity_for_bot * self.so1_position_value_percentage.value
            
            calculated_so_cost = target_so1_position_value / current_trade_leverage

        else: # SO2, SO3, ...
            prev_so_number = next_so_number - 1
            reference_prev_so_cost = trade_so_stakes.get(f'actual_stake_of_so_{prev_so_number}')

            if reference_prev_so_cost is None or reference_prev_so_cost <= 0:
                logger.warning(f"Could not find valid 'actual_stake_of_so_{prev_so_number}' (cost) for trade {trade.id} "
                               f"to base SO#{next_so_number} upon. Value: {reference_prev_so_cost}. Skipping SO.")
                return None
            
            calculated_so_cost = reference_prev_so_cost * self.so_subsequent_size_multiplier.value # so_subsequent_size_multiplier is shared

        if calculated_so_cost <= 0:
            logger.warning(f"Calculated SO cost is {calculated_so_cost:.2f} for {trade.pair} (SO#{next_so_number}). Skipping SO.")
            return None
        
        final_so_cost = calculated_so_cost
        if min_stake is not None:
            final_so_cost = max(final_so_cost, min_stake)
        
        final_so_cost = min(final_so_cost, max_stake)

        if final_so_cost <= 0:
            logger.warning(f"Final SO cost is {final_so_cost:.2f} after clamping for {trade.pair} (SO#{next_so_number}). Skipping SO.")
            return None
        
        if min_stake is not None and final_so_cost < min_stake:
            return None

        price_offset = next_so_number * current_so_atr_distance_multiplier * current_atr # Use side-specific multiplier
        
        target_so_price: float
        trigger_condition: bool
        so_side_tag: str

        if not trade.is_short:
            target_so_price = initial_entry_fill_price - price_offset
            trigger_condition = current_rate < target_so_price
            so_side_tag = "long"
        else:
            target_so_price = initial_entry_fill_price + price_offset
            trigger_condition = current_rate > target_so_price
            so_side_tag = "short"
            
        if trigger_condition:
            trade_so_stakes[f'so_{next_so_number}_initial_entry_price'] = initial_entry_fill_price
            trade_so_stakes[f'so_{next_so_number}_target_price'] = target_so_price
            trade_so_stakes[f'so_{next_so_number}_atr_at_trigger'] = current_atr
            trade_so_stakes[f'actual_stake_of_so_{next_so_number}'] = final_so_cost
            
            return float(final_so_cost)

        return None

    # Override ft_stoploss_reached to disable percentage-based stoploss
    def ft_stoploss_reached(
        self,
        current_rate: float,
        trade: Trade,
        current_time: datetime,
        current_profit: float,
        force_stoploss: float, # This is the stoploss value from config/strategy attribute
        low: float | None = None,
        high: float | None = None,
    ) -> ExitCheckTuple:
        """
        Override to disable percentage-based stoploss.
        Only liquidation or other exit signals should close the trade.
        """
        super().ft_stoploss_adjust(
            current_rate, trade, current_time, current_profit, force_stoploss, low, high, after_fill=False
        )

        liq_price_valid = trade.liquidation_price is not None and trade.liquidation_price > 0
        
        if liq_price_valid:
            if not trade.is_short: # Long position
                if low is not None and low <= trade.liquidation_price: # Backtesting with low
                    return ExitCheckTuple(exit_type=ExitType.LIQUIDATION)
                elif low is None and current_rate <= trade.liquidation_price: # Live/Dry-run
                    return ExitCheckTuple(exit_type=ExitType.LIQUIDATION)
            else: # Short position
                if high is not None and high >= trade.liquidation_price: # Backtesting with high
                    return ExitCheckTuple(exit_type=ExitType.LIQUIDATION)
                elif high is None and current_rate >= trade.liquidation_price: # Live/Dry-run
                    return ExitCheckTuple(exit_type=ExitType.LIQUIDATION)

        return ExitCheckTuple(exit_type=ExitType.NONE)


    def ft_check_timed_out(self, trade: Trade, order: Order, current_time: datetime) -> bool:
        """
        Custom timeout logic.
        Checks if the associated trade is closed and the order is still open.
        If so, signals to cancel the order.
        This method should also include any other regular timeout logic the strategy might need.
        :param trade: The trade associated with this order.
        :param order: The order object itself.
        :param current_time: The current time.
        :return: True if the order should be cancelled, False otherwise.
        """
        if not trade.is_open and order.ft_is_open:
            return True
        return False