import logging

from pandas import DataFrame, read_feather, to_datetime

from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS
from freqtrade.enums import CandleType, TradingMode

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class FeatherDataHandler(IDataHandler):
    _columns = DEFAULT_DATAFRAME_COLUMNS

    def ohlcv_store(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        """
        Store data in json format "values".
            format looks as follows:
            [[<date>,<open>,<high>,<low>,<close>]]
        :param pair: Pair - used to generate filename
        :param timeframe: Timeframe - used to generate filename
        :param data: Dataframe containing OHLCV data
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: None
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        # logger.info(f"FDH_STORE: Storing data for {pair} {timeframe} {candle_type} to {filename}")
        self.create_dir_if_needed(filename)

        data.reset_index(drop=True).loc[:, self._columns].to_feather(
            filename, compression_level=9, compression="lz4"
        )

    def _ohlcv_load(
        self, pair: str, timeframe: str, timerange: TimeRange | None, candle_type: CandleType
    ) -> DataFrame:
        """
        Internal method used to load data for one pair from disk.
        Implements the loading and conversion to a Pandas dataframe.
        Timerange trimming and dataframe validation happens outside of this method.
        :param pair: Pair to load data
        :param timeframe: Timeframe (e.g. "5m")
        :param timerange: Limit data to be loaded to this timerange.
                        Optionally implemented by subclasses to avoid loading
                        all data where possible.
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: DataFrame with ohlcv data, or empty DataFrame
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type=candle_type)
        # logger.info(f"FDH_LOAD: Initial attempt for {pair} {timeframe} {candle_type} using file: {filename}")
        if not filename.exists():
            # logger.info(f"FDH_LOAD: Primary file {filename} does not exist. Attempting fallback.")
            # Fallback mode for 1M files
            filename_fallback_path = self._pair_data_filename( # Use a temp name for clarity
                self._datadir, pair, timeframe, candle_type=candle_type, no_timeframe_modify=True
            )
            # logger.info(f"FDH_LOAD: Fallback filename is {filename_fallback_path}")
            if not filename_fallback_path.exists():
                # logger.info(f"FDH_LOAD: Fallback file {filename_fallback_path} also does not exist. Returning empty DataFrame for {pair} {timeframe} {candle_type}.")
                return DataFrame(columns=self._columns)
            else:
                # logger.info(f"FDH_LOAD: Fallback file {filename_fallback_path} exists. Using this file for {pair} {timeframe} {candle_type}.")
                filename = filename_fallback_path # Assign to filename to be used by try block
        else:
            pass
            # logger.info(f"FDH_LOAD: Primary file {filename} exists for {pair} {timeframe} {candle_type}.")

        try:
            pairdata = read_feather(filename)
            # logger.info(f"FDH_LOAD: Successfully read {len(pairdata)} rows from {filename} for {pair} {timeframe} {candle_type}.")
            pairdata.columns = self._columns
            pairdata = pairdata.astype(
                dtype={
                    "open": "float",
                    "high": "float",
                    "low": "float",
                    "close": "float",
                    "volume": "float",
                }
            )
            pairdata["date"] = to_datetime(pairdata["date"], unit="ms", utc=True)
            return pairdata
        except Exception as e:
            # logger.exception(
            #     f"FDH_LOAD: Error loading data from {filename} for {pair} {timeframe} {candle_type}. Exception: {e}. Returning empty dataframe."
            # )
            return DataFrame(columns=self._columns)

    def ohlcv_append(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        """
        Append data to existing data structures
        :param pair: Pair
        :param timeframe: Timeframe this ohlcv data is for
        :param data: Data to append.
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """
        raise NotImplementedError()

    def _trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        :param trading_mode: Trading mode to use (used to determine the filename)
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        self.create_dir_if_needed(filename)
        data.reset_index(drop=True).to_feather(filename, compression_level=9, compression="lz4")

    def trades_append(self, pair: str, data: DataFrame):
        """
        Append data to existing files
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        raise NotImplementedError()

    def _trades_load(
        self, pair: str, trading_mode: TradingMode, timerange: TimeRange | None = None
    ) -> DataFrame:
        """
        Load a pair from file, either .json.gz or .json
        # TODO: respect timerange ...
        :param pair: Load trades for this pair
        :param trading_mode: Trading mode to use (used to determine the filename)
        :param timerange: Timerange to load trades for - currently not implemented
        :return: Dataframe containing trades
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        if not filename.exists():
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

        tradesdata = read_feather(filename)

        return tradesdata

    @classmethod
    def _get_file_extension(cls):
        return "feather"
