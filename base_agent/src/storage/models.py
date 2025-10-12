"""
Data models for the cell-based evolution system.

These dataclasses represent the core entities stored in the database.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class Cell:
    """
    A successful trading strategy (birthed cell).

    Represents a strategy that survived natural selection with:
    - Genome: DSL string (the strategy code)
    - Fitness: Economic performance
    - Lineage: Parent and generation tracking
    - Semantics: LLM's interpretation (optional)
    - Identity: Unique cell_name and dish_name for organization
    """
    cell_id: int
    generation: int
    parent_cell_id: Optional[int]
    dsl_genome: str
    fitness: float
    status: str  # 'online', 'deprecated', 'archived', 'extinct'
    created_at: datetime

    # Dish identity (for multi-experiment organization)
    cell_name: Optional[str] = None  # e.g., "baseline_purr_g114_c001"
    dish_name: Optional[str] = None  # e.g., "baseline_purr"

    # LLM semantics (optional, populated by trading-learn mode)
    llm_name: Optional[str] = None
    llm_category: Optional[str] = None
    llm_hypothesis: Optional[str] = None
    llm_confidence: Optional[float] = None
    llm_analyzed_at: Optional[datetime] = None

    # Deprecation info
    deprecated_reason: Optional[str] = None
    superseded_by_cell_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'cell_id': self.cell_id,
            'cell_name': self.cell_name,
            'dish_name': self.dish_name,
            'generation': self.generation,
            'parent_cell_id': self.parent_cell_id,
            'dsl_genome': self.dsl_genome,
            'fitness': self.fitness,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'llm_name': self.llm_name,
            'llm_category': self.llm_category,
            'llm_hypothesis': self.llm_hypothesis,
            'llm_confidence': self.llm_confidence,
            'llm_analyzed_at': self.llm_analyzed_at.isoformat() if self.llm_analyzed_at else None,
            'deprecated_reason': self.deprecated_reason,
            'superseded_by_cell_id': self.superseded_by_cell_id,
        }


@dataclass
class CellPhenotype:
    """
    Market behavior of a cell (how the genome expresses itself).

    Each cell can have multiple phenotypes (one per timeframe or symbol tested).
    """
    phenotype_id: int
    cell_id: int
    symbol: str  # e.g., 'PURR'
    timeframe: str  # e.g., '1H', '4H', '1D'

    # Test context
    data_start_date: Optional[str] = None
    data_end_date: Optional[str] = None

    # Trading metrics
    total_trades: int = 0
    profitable_trades: int = 0
    losing_trades: int = 0

    # Financial metrics
    total_profit: float = 0.0
    total_fees: float = 0.0
    max_drawdown: Optional[float] = None
    max_runup: Optional[float] = None

    # Risk metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None

    # Trade characteristics
    avg_trade_duration_hours: Optional[float] = None
    avg_profit_per_trade: Optional[float] = None
    avg_loss_per_trade: Optional[float] = None
    longest_winning_streak: Optional[int] = None
    longest_losing_streak: Optional[int] = None

    # Trigger analysis (JSON string)
    trigger_conditions: Optional[str] = None

    tested_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'phenotype_id': self.phenotype_id,
            'cell_id': self.cell_id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'data_start_date': self.data_start_date,
            'data_end_date': self.data_end_date,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'losing_trades': self.losing_trades,
            'total_profit': self.total_profit,
            'total_fees': self.total_fees,
            'max_drawdown': self.max_drawdown,
            'max_runup': self.max_runup,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_duration_hours': self.avg_trade_duration_hours,
            'avg_profit_per_trade': self.avg_profit_per_trade,
            'avg_loss_per_trade': self.avg_loss_per_trade,
            'longest_winning_streak': self.longest_winning_streak,
            'longest_losing_streak': self.longest_losing_streak,
            'trigger_conditions': json.loads(self.trigger_conditions) if self.trigger_conditions else None,
            'tested_at': self.tested_at.isoformat() if self.tested_at else None,
        }


@dataclass
class DiscoveredPattern:
    """
    An LLM-named trading pattern discovered through evolution.

    Patterns are emergent - they're discovered after evolution finds profitable
    structures, not pre-defined by humans.
    """
    pattern_id: int
    pattern_name: str  # e.g., "Volume Spike Reversal"
    category: str  # e.g., "Volume Analysis"
    description: str  # LLM's explanation

    # Pattern characteristics
    typical_dsl_structure: Optional[str] = None
    required_indicators: Optional[str] = None  # JSON array like ["EPSILON", "DELTA"]
    typical_parameters: Optional[str] = None  # JSON object

    # Discovery info
    discovered_at: datetime = field(default_factory=datetime.now)
    discovered_by_cell_id: Optional[int] = None

    # Performance tracking
    cells_using_pattern: int = 0
    avg_fitness: Optional[float] = None
    best_fitness: Optional[float] = None
    worst_fitness: Optional[float] = None
    best_cell_id: Optional[int] = None

    # Market applicability
    works_on_symbols: Optional[str] = None  # JSON array
    fails_on_symbols: Optional[str] = None  # JSON array

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'category': self.category,
            'description': self.description,
            'typical_dsl_structure': self.typical_dsl_structure,
            'required_indicators': json.loads(self.required_indicators) if self.required_indicators else None,
            'typical_parameters': json.loads(self.typical_parameters) if self.typical_parameters else None,
            'discovered_at': self.discovered_at.isoformat() if self.discovered_at else None,
            'discovered_by_cell_id': self.discovered_by_cell_id,
            'cells_using_pattern': self.cells_using_pattern,
            'avg_fitness': self.avg_fitness,
            'best_fitness': self.best_fitness,
            'worst_fitness': self.worst_fitness,
            'best_cell_id': self.best_cell_id,
            'works_on_symbols': json.loads(self.works_on_symbols) if self.works_on_symbols else None,
            'fails_on_symbols': json.loads(self.fails_on_symbols) if self.fails_on_symbols else None,
        }


@dataclass
class EvolutionRun:
    """
    Metadata for a complete evolution session.

    Tracks configuration, results, and termination reason for an evolution run.
    """
    run_id: int
    run_type: str  # 'evolution' or 'trading-learn'

    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Configuration
    max_generations: Optional[int] = None
    fitness_goal: Optional[float] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    initial_capital: Optional[float] = None
    transaction_fee_rate: Optional[float] = None

    # Results
    best_cell_id: Optional[int] = None
    total_cells_birthed: int = 0
    total_mutations_failed: int = 0
    final_best_fitness: Optional[float] = None

    # Termination
    termination_reason: Optional[str] = None
    generations_without_improvement: Optional[int] = None

    # Costs (for trading-learn mode)
    total_llm_cost: float = 0.0
    total_tokens_used: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'run_id': self.run_id,
            'run_type': self.run_type,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'max_generations': self.max_generations,
            'fitness_goal': self.fitness_goal,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'initial_capital': self.initial_capital,
            'transaction_fee_rate': self.transaction_fee_rate,
            'best_cell_id': self.best_cell_id,
            'total_cells_birthed': self.total_cells_birthed,
            'total_mutations_failed': self.total_mutations_failed,
            'final_best_fitness': self.final_best_fitness,
            'termination_reason': self.termination_reason,
            'generations_without_improvement': self.generations_without_improvement,
            'total_llm_cost': self.total_llm_cost,
            'total_tokens_used': self.total_tokens_used,
        }
