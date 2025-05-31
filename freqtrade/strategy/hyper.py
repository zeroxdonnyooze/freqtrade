"""
IHyperStrategy interface, hyperoptable Parameter class.
This module defines a base class for auto-hyperoptable strategies.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.misc import deep_merge_dicts
from freqtrade.optimize.hyperopt_tools import HyperoptTools
from freqtrade.strategy.parameters import BaseParameter


logger = logging.getLogger(__name__)


class HyperStrategyMixin:
    """
    A helper base class which allows HyperOptAuto class to reuse implementations of buy/sell
     strategy logic.
    """

    def __init__(self, config: Config, *args, **kwargs):
        """
        Initialize hyperoptable strategy mixin.
        """
        self.config = config
        self.ft_buy_params: list[BaseParameter] = []
        self.ft_sell_params: list[BaseParameter] = []
        self.ft_protection_params: list[BaseParameter] = []

        params = self.load_params_from_file()
        params = params.get("params", {})
        self._ft_params_from_file = params
        # Init/loading of parameters is done as part of ft_bot_start().

    def enumerate_parameters(
        self, category: str | None = None
    ) -> Iterator[tuple[str, BaseParameter]]:
        """
        Find all optimizable parameters and return (name, attr) iterator.
        :param category:
        :return:
        """
        if category not in ("buy", "sell", "protection", None):
            raise OperationalException(
                'Category must be one of: "buy", "sell", "protection", None.'
            )

        if category is None:
            params = self.ft_buy_params + self.ft_sell_params + self.ft_protection_params
        else:
            params = getattr(self, f"ft_{category}_params")

        for par in params:
            yield par.name, par

    @classmethod
    def detect_all_parameters(cls) -> dict:
        """Detect all parameters and return them as a list"""
        params: dict[str, Any] = {
            "buy": list(detect_parameters(cls, "buy")),
            "sell": list(detect_parameters(cls, "sell")),
            "protection": list(detect_parameters(cls, "protection")),
        }
        params.update({"count": len(params["buy"] + params["sell"] + params["protection"])})

        return params

    def ft_load_params_from_file(self) -> None:
        """
        Load Parameters from parameter file
        Should/must run before config values are loaded in strategy_resolver.
        """
        if self._ft_params_from_file:
            # Set parameters from Hyperopt results file
            params = self._ft_params_from_file
            self.minimal_roi = params.get("roi", getattr(self, "minimal_roi", {}))

            self.stoploss = params.get("stoploss", {}).get(
                "stoploss", getattr(self, "stoploss", -0.1)
            )
            self.max_open_trades = params.get("max_open_trades", {}).get(
                "max_open_trades", getattr(self, "max_open_trades", -1)
            )
            trailing = params.get("trailing", {})
            self.trailing_stop = trailing.get(
                "trailing_stop", getattr(self, "trailing_stop", False)
            )
            self.trailing_stop_positive = trailing.get(
                "trailing_stop_positive", getattr(self, "trailing_stop_positive", None)
            )
            self.trailing_stop_positive_offset = trailing.get(
                "trailing_stop_positive_offset", getattr(self, "trailing_stop_positive_offset", 0)
            )
            self.trailing_only_offset_is_reached = trailing.get(
                "trailing_only_offset_is_reached",
                getattr(self, "trailing_only_offset_is_reached", 0.0),
            )

    def ft_load_hyper_params(self, hyperopt: bool = False) -> None:
        """
        Load Hyperoptable parameters
        Prevalence:
        * Parameters from parameter file
        * Parameters defined in parameters objects (buy_params, sell_params, ...)
        * Parameter defaults
        """
        # Get the strategy-specific block directly using the strategy's name
        # self is an instance of IStrategy (or a subclass) due to HyperStrategyMixin
        strategy_name = self.get_strategy_name() # Assuming get_strategy_name() is available
        strategy_params_from_config = self.config.get(strategy_name, {})
        


        # Parameters from strategy class attribute (e.g., self.buy_params = {...})
        buy_params_from_strategy_attr = getattr(self, "buy_params", {})
        # Parameters from <strategy>.json file
        buy_params_from_file = self._ft_params_from_file.get("buy", {})
        # Parameters from main config.json strategy_parameters.buy section
        buy_params_from_main_config = strategy_params_from_config.get("buy", {})

        # Merge: main_config overrides file, which overrides strategy attribute
        buy_params = deep_merge_dicts(buy_params_from_strategy_attr, buy_params_from_file)
        buy_params = deep_merge_dicts(buy_params, buy_params_from_main_config)


        sell_params_from_strategy_attr = getattr(self, "sell_params", {})
        sell_params_from_file = self._ft_params_from_file.get("sell", {})
        sell_params_from_main_config = strategy_params_from_config.get("sell", {})
        sell_params = deep_merge_dicts(sell_params_from_strategy_attr, sell_params_from_file)
        sell_params = deep_merge_dicts(sell_params, sell_params_from_main_config)


        protection_params_from_strategy_attr = getattr(self, "protection_params", {})
        protection_params_from_file = self._ft_params_from_file.get("protection", {})
        protection_params_from_main_config = strategy_params_from_config.get("protection", {})
        protection_params = deep_merge_dicts(protection_params_from_strategy_attr, protection_params_from_file)
        protection_params = deep_merge_dicts(protection_params, protection_params_from_main_config)

        self._ft_load_params(buy_params, "buy", hyperopt)
        self._ft_load_params(sell_params, "sell", hyperopt)
        self._ft_load_params(protection_params, "protection", hyperopt)

    def load_params_from_file(self) -> dict:
        filename_str = getattr(self, "__file__", "")
        if not filename_str:
            return {}
        filename = Path(filename_str).with_suffix(".json")

        if filename.is_file():
            logger.info(f"Loading parameters from file {filename}")
            try:
                params = HyperoptTools.load_params(filename)
                if params.get("strategy_name") != self.__class__.__name__:
                    raise OperationalException("Invalid parameter file provided.")
                return params
            except ValueError:
                logger.warning("Invalid parameter file format.")
                return {}
        logger.info("Found no parameter file.")

        return {}

    def _ft_load_params(self, params: dict, space: str, hyperopt: bool = False) -> None:
        """
        Set optimizable parameter values.
        :param params: Dictionary with new parameter values.
        """
        if not params:
            logger.info(f"No params for {space} found, using default values.")
        param_container: list[BaseParameter] = getattr(self, f"ft_{space}_params")

        for attr_name, attr in detect_parameters(self, space):
            attr.name = attr_name
            attr.in_space = hyperopt and HyperoptTools.has_space(self.config, space)
            if not attr.category:
                attr.category = space

            param_container.append(attr)

            if attr.load:
                loaded_value = None
                source_description = "default" # Default source description

                # 1. Check in space-specific params dictionary (e.g., buy_params derived from config's strategy_parameters.buy)
                if params and attr_name in params:
                    loaded_value = params[attr_name]
                    source_description = f"space-specific params dict for space '{space}'"
                else:
                    # 2. If not in space-specific, check the strategy-specific block from the main config.
                    # This was already loaded into strategy_params_from_config in ft_load_hyper_params
                    # and then into `params` for the current space.
                    # The original `params` dictionary (e.g., buy_params) should contain the correct values if the
                    # strategy_params_from_config was loaded correctly.
                    # This fallback might be redundant if the initial `params` construction is correct.
                    # However, to be safe and align with the original intent of checking a broader scope:
                    strategy_block_from_config = self.config.get(self.get_strategy_name(), {})
                    if attr_name in strategy_block_from_config:
                        if not isinstance(strategy_block_from_config[attr_name], dict): # Ensure it's a simple value
                            loaded_value = strategy_block_from_config[attr_name]
                            source_description = f"config's strategy block '{self.get_strategy_name()}' (param for space '{space}')"
                
                if loaded_value is not None:
                    attr.value = loaded_value
                    logger.info(f"Strategy Parameter ({source_description}): {attr_name} = {attr.value}")
                else: # Not found in specific or suitable top-level, use default
                    logger.info(f"Strategy Parameter(default): {attr_name} = {attr.value}")
            else: # attr.load is False
                logger.warning(
                    f'Parameter "{attr_name}" exists, but is disabled (load=False). '
                    f'Default value "{attr.value}" used.'
                )

    def get_no_optimize_params(self) -> dict[str, dict]:
        """
        Returns list of Parameters that are not part of the current optimize job
        """
        params: dict[str, dict] = {
            "buy": {},
            "sell": {},
            "protection": {},
        }
        for name, p in self.enumerate_parameters():
            if p.category and (not p.optimize or not p.in_space):
                params[p.category][name] = p.value
        return params


def detect_parameters(
    obj: HyperStrategyMixin | type[HyperStrategyMixin], category: str
) -> Iterator[tuple[str, BaseParameter]]:
    """
    Detect all parameters for 'category' for "obj"
    :param obj: Strategy object or class
    :param category: category - usually `'buy', 'sell', 'protection',...
    """
    for attr_name in dir(obj):
        if not attr_name.startswith("__"):  # Ignore internals, not strictly necessary.
            attr = getattr(obj, attr_name)
            if issubclass(attr.__class__, BaseParameter):
                if (
                    attr_name.startswith(category + "_")
                    and attr.category is not None
                    and attr.category != category
                ):
                    raise OperationalException(
                        f"Inconclusive parameter name {attr_name}, category: {attr.category}."
                    )

                if category == attr.category or (
                    attr_name.startswith(category + "_") and attr.category is None
                ):
                    yield attr_name, attr
