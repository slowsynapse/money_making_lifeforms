"""
Trading Evolution Configuration

Centralized configuration for the trading evolution system.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TradingConfig:
    """Configuration for trading evolution experiments"""

    # Market configuration
    DEFAULT_SYMBOL: str = "PURR"
    DEFAULT_TIMEFRAMES: List[str] = None

    # Capital and fees
    DEFAULT_INITIAL_CAPITAL: float = 100.0
    DEFAULT_FEE_RATE: float = 0.00045  # HyperLiquid taker fee

    # Evolution parameters
    LENIENT_CELL_COUNT: int = 100  # Birth any survivor for first N cells
    STAGNATION_LIMIT: int = 100  # Terminate after N gens without improvement
    DEFAULT_FITNESS_GOAL: float = 200.0

    # LLM analysis parameters
    PATTERN_ANALYSIS_BATCH_SIZE: int = 30  # Cells per LLM batch (for 8K context)
    MAX_CELLS_TO_ANALYZE: int = 100  # Top N cells for pattern discovery
    MIN_TRADES_FOR_ANALYSIS: int = 1  # Filter out zero-trade strategies

    # Database configuration
    DEFAULT_DB_NAME: str = "cells.db"

    def __post_init__(self):
        if self.DEFAULT_TIMEFRAMES is None:
            self.DEFAULT_TIMEFRAMES = ["1H", "4H", "1D"]


# Global config instance
config = TradingConfig()
