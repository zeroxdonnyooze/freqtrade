from enum import Enum


class BacktestState(Enum):
    """
    Bot application states
    """

    STARTUP = 1
    DATALOAD = 2
    ANALYZE = 3
    CONVERT = 4
    PAIRLIST_PRECALC = 5
    BACKTEST = 6

    def __str__(self):
        return f"{self.name.lower()}"
