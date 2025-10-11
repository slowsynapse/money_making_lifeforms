"""
Trading Evolution System

This module contains the standalone trading strategy evolution system,
extracted from the SICA agent framework for cleaner architecture.
"""

from .trading_config import TradingConfig
from .trading_evolution import (
    run_trading_demo,
    run_trading_test,
    run_trading_evolve,
    run_trading_learn,
)

__all__ = [
    'TradingConfig',
    'run_trading_demo',
    'run_trading_test',
    'run_trading_evolve',
    'run_trading_learn',
]
