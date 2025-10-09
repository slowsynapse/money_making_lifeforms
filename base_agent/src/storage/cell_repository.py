"""
Cell repository for persistent storage of trading strategies.

Provides CRUD operations for cells, phenotypes, patterns, and evolution runs.
Uses SQLite for local storage with full transaction support.
"""

import sqlite3
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from contextlib import contextmanager

from .models import Cell, CellPhenotype, DiscoveredPattern, EvolutionRun


class CellRepository:
    """
    Repository for managing cell storage and retrieval.

    This class handles all database operations for the evolution system,
    including cell birth, lineage tracking, phenotype storage, and pattern discovery.
    """

    def __init__(self, db_path: Path):
        """
        Initialize repository with database path.

        Args:
            db_path: Path to SQLite database file (will be created if doesn't exist)
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with transaction support."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self):
        """
        Create database schema if it doesn't exist.

        Creates all 8 tables from DATABASE_SCHEMA.md:
        - cells: Successful strategies with lineage
        - cell_phenotypes: Market behavior per timeframe/symbol
        - failed_mutations: Statistics on unsuccessful mutations
        - evolution_runs: Metadata for evolution sessions
        - discovered_patterns: LLM-named patterns
        - cell_patterns: Many-to-many relationship
        - cell_metadata: Key-value storage for cells
        - cell_mutation_proposals: LLM-suggested mutations
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Table 1: cells
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cells (
                    cell_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation INTEGER NOT NULL,
                    parent_cell_id INTEGER,
                    dsl_genome TEXT NOT NULL,
                    fitness REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'online',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    llm_name TEXT,
                    llm_category TEXT,
                    llm_hypothesis TEXT,
                    llm_confidence REAL,
                    llm_analyzed_at TIMESTAMP,

                    deprecated_reason TEXT,
                    superseded_by_cell_id INTEGER,

                    FOREIGN KEY (parent_cell_id) REFERENCES cells(cell_id),
                    FOREIGN KEY (superseded_by_cell_id) REFERENCES cells(cell_id),
                    CHECK (status IN ('online', 'deprecated', 'archived', 'extinct'))
                )
            """)

            # Table 2: cell_phenotypes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cell_phenotypes (
                    phenotype_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cell_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,

                    data_start_date TEXT,
                    data_end_date TEXT,

                    total_trades INTEGER DEFAULT 0,
                    profitable_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,

                    total_profit REAL DEFAULT 0.0,
                    total_fees REAL DEFAULT 0.0,
                    max_drawdown REAL,
                    max_runup REAL,

                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    win_rate REAL,
                    profit_factor REAL,

                    avg_trade_duration_hours REAL,
                    avg_profit_per_trade REAL,
                    avg_loss_per_trade REAL,
                    longest_winning_streak INTEGER,
                    longest_losing_streak INTEGER,

                    trigger_conditions TEXT,
                    tested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (cell_id) REFERENCES cells(cell_id) ON DELETE CASCADE
                )
            """)

            # Table 3: failed_mutations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS failed_mutations (
                    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_cell_id INTEGER NOT NULL,
                    attempted_genome TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    failure_reason TEXT NOT NULL,
                    fitness_achieved REAL,
                    failed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (parent_cell_id) REFERENCES cells(cell_id),
                    CHECK (failure_reason IN ('negative_fitness', 'lower_than_parent', 'execution_error', 'invalid_dsl'))
                )
            """)

            # Table 4: evolution_runs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evolution_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_type TEXT NOT NULL,

                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    duration_seconds REAL,

                    max_generations INTEGER,
                    fitness_goal REAL,
                    symbol TEXT,
                    timeframe TEXT,
                    initial_capital REAL,
                    transaction_fee_rate REAL,

                    best_cell_id INTEGER,
                    total_cells_birthed INTEGER DEFAULT 0,
                    total_mutations_failed INTEGER DEFAULT 0,
                    final_best_fitness REAL,

                    termination_reason TEXT,
                    generations_without_improvement INTEGER,

                    total_llm_cost REAL DEFAULT 0.0,
                    total_tokens_used INTEGER DEFAULT 0,

                    FOREIGN KEY (best_cell_id) REFERENCES cells(cell_id),
                    CHECK (run_type IN ('evolution', 'trading-learn'))
                )
            """)

            # Table 5: discovered_patterns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discovered_patterns (
                    pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,

                    typical_dsl_structure TEXT,
                    required_indicators TEXT,
                    typical_parameters TEXT,

                    discovered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    discovered_by_cell_id INTEGER,

                    cells_using_pattern INTEGER DEFAULT 0,
                    avg_fitness REAL,
                    best_fitness REAL,
                    worst_fitness REAL,
                    best_cell_id INTEGER,

                    works_on_symbols TEXT,
                    fails_on_symbols TEXT,

                    FOREIGN KEY (discovered_by_cell_id) REFERENCES cells(cell_id),
                    FOREIGN KEY (best_cell_id) REFERENCES cells(cell_id)
                )
            """)

            # Table 6: cell_patterns (many-to-many)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cell_patterns (
                    cell_id INTEGER NOT NULL,
                    pattern_id INTEGER NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    assigned_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    PRIMARY KEY (cell_id, pattern_id),
                    FOREIGN KEY (cell_id) REFERENCES cells(cell_id) ON DELETE CASCADE,
                    FOREIGN KEY (pattern_id) REFERENCES discovered_patterns(pattern_id) ON DELETE CASCADE
                )
            """)

            # Table 7: cell_metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cell_metadata (
                    cell_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,

                    PRIMARY KEY (cell_id, key),
                    FOREIGN KEY (cell_id) REFERENCES cells(cell_id) ON DELETE CASCADE
                )
            """)

            # Table 8: cell_mutation_proposals
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cell_mutation_proposals (
                    proposal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cell_id INTEGER NOT NULL,
                    proposed_genome TEXT NOT NULL,
                    mutation_rationale TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'pending',
                    proposed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    tested_at TIMESTAMP,
                    result_cell_id INTEGER,

                    FOREIGN KEY (cell_id) REFERENCES cells(cell_id) ON DELETE CASCADE,
                    FOREIGN KEY (result_cell_id) REFERENCES cells(cell_id),
                    CHECK (status IN ('pending', 'tested', 'abandoned'))
                )
            """)

            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cells_fitness
                ON cells(fitness DESC) WHERE status = 'online'
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cells_generation
                ON cells(generation)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cells_parent
                ON cells(parent_cell_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cells_unanalyzed
                ON cells(fitness) WHERE llm_name IS NULL AND status = 'online'
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_phenotypes_cell
                ON cell_phenotypes(cell_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_failed_mutations_parent
                ON failed_mutations(parent_cell_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_category
                ON discovered_patterns(category)
            """)

            conn.commit()

    def _row_to_cell(self, row: sqlite3.Row) -> Cell:
        """Convert database row to Cell dataclass."""
        return Cell(
            cell_id=row['cell_id'],
            generation=row['generation'],
            parent_cell_id=row['parent_cell_id'],
            dsl_genome=row['dsl_genome'],
            fitness=row['fitness'],
            status=row['status'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            llm_name=row['llm_name'],
            llm_category=row['llm_category'],
            llm_hypothesis=row['llm_hypothesis'],
            llm_confidence=row['llm_confidence'],
            llm_analyzed_at=datetime.fromisoformat(row['llm_analyzed_at']) if row['llm_analyzed_at'] else None,
            deprecated_reason=row['deprecated_reason'],
            superseded_by_cell_id=row['superseded_by_cell_id'],
        )

    def _row_to_phenotype(self, row: sqlite3.Row) -> CellPhenotype:
        """Convert database row to CellPhenotype dataclass."""
        return CellPhenotype(
            phenotype_id=row['phenotype_id'],
            cell_id=row['cell_id'],
            symbol=row['symbol'],
            timeframe=row['timeframe'],
            data_start_date=row['data_start_date'],
            data_end_date=row['data_end_date'],
            total_trades=row['total_trades'],
            profitable_trades=row['profitable_trades'],
            losing_trades=row['losing_trades'],
            total_profit=row['total_profit'],
            total_fees=row['total_fees'],
            max_drawdown=row['max_drawdown'],
            max_runup=row['max_runup'],
            sharpe_ratio=row['sharpe_ratio'],
            sortino_ratio=row['sortino_ratio'],
            win_rate=row['win_rate'],
            profit_factor=row['profit_factor'],
            avg_trade_duration_hours=row['avg_trade_duration_hours'],
            avg_profit_per_trade=row['avg_profit_per_trade'],
            avg_loss_per_trade=row['avg_loss_per_trade'],
            longest_winning_streak=row['longest_winning_streak'],
            longest_losing_streak=row['longest_losing_streak'],
            trigger_conditions=row['trigger_conditions'],
            tested_at=datetime.fromisoformat(row['tested_at']) if row['tested_at'] else None,
        )

    def _row_to_pattern(self, row: sqlite3.Row) -> DiscoveredPattern:
        """Convert database row to DiscoveredPattern dataclass."""
        return DiscoveredPattern(
            pattern_id=row['pattern_id'],
            pattern_name=row['pattern_name'],
            category=row['category'],
            description=row['description'],
            typical_dsl_structure=row['typical_dsl_structure'],
            required_indicators=row['required_indicators'],
            typical_parameters=row['typical_parameters'],
            discovered_at=datetime.fromisoformat(row['discovered_at']) if row['discovered_at'] else None,
            discovered_by_cell_id=row['discovered_by_cell_id'],
            cells_using_pattern=row['cells_using_pattern'],
            avg_fitness=row['avg_fitness'],
            best_fitness=row['best_fitness'],
            worst_fitness=row['worst_fitness'],
            best_cell_id=row['best_cell_id'],
            works_on_symbols=row['works_on_symbols'],
            fails_on_symbols=row['fails_on_symbols'],
        )

    def _row_to_evolution_run(self, row: sqlite3.Row) -> EvolutionRun:
        """Convert database row to EvolutionRun dataclass."""
        return EvolutionRun(
            run_id=row['run_id'],
            run_type=row['run_type'],
            started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            duration_seconds=row['duration_seconds'],
            max_generations=row['max_generations'],
            fitness_goal=row['fitness_goal'],
            symbol=row['symbol'],
            timeframe=row['timeframe'],
            initial_capital=row['initial_capital'],
            transaction_fee_rate=row['transaction_fee_rate'],
            best_cell_id=row['best_cell_id'],
            total_cells_birthed=row['total_cells_birthed'],
            total_mutations_failed=row['total_mutations_failed'],
            final_best_fitness=row['final_best_fitness'],
            termination_reason=row['termination_reason'],
            generations_without_improvement=row['generations_without_improvement'],
            total_llm_cost=row['total_llm_cost'],
            total_tokens_used=row['total_tokens_used'],
        )

    # CRUD Methods for Cells
    def birth_cell(
        self,
        generation: int,
        parent_cell_id: Optional[int],
        dsl_genome: str,
        fitness: float,
        status: str = 'online',
    ) -> int:
        """
        Birth a new cell (insert into database).

        Args:
            generation: Generation number (0 for seed)
            parent_cell_id: Parent cell ID (None for seed)
            dsl_genome: DSL strategy code
            fitness: Economic performance (profit - fees - LLM costs)
            status: Cell status ('online' by default)

        Returns:
            cell_id: ID of the newly birthed cell
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cells (generation, parent_cell_id, dsl_genome, fitness, status)
                VALUES (?, ?, ?, ?, ?)
            """, (generation, parent_cell_id, dsl_genome, fitness, status))
            return cursor.lastrowid

    def get_cell(self, cell_id: int) -> Optional[Cell]:
        """
        Retrieve a cell by ID.

        Args:
            cell_id: ID of the cell to retrieve

        Returns:
            Cell object or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cells WHERE cell_id = ?", (cell_id,))
            row = cursor.fetchone()
            return self._row_to_cell(row) if row else None

    def get_top_cells(self, limit: int = 10, status: str = 'online', min_trades: int = 0) -> List[Cell]:
        """
        Get top cells by fitness.

        Args:
            limit: Maximum number of cells to return
            status: Filter by status ('online', 'deprecated', 'archived', 'extinct')
            min_trades: Minimum number of trades required (default 0). Use min_trades=1 to filter
                       out zero-trade strategies for LLM analysis.

        Returns:
            List of Cell objects sorted by fitness descending
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if min_trades > 0:
                # Filter cells that have at least one phenotype with min_trades
                cursor.execute("""
                    SELECT DISTINCT c.* FROM cells c
                    INNER JOIN cell_phenotypes cp ON c.cell_id = cp.cell_id
                    WHERE c.status = ?
                      AND cp.total_trades >= ?
                    ORDER BY c.fitness DESC
                    LIMIT ?
                """, (status, min_trades, limit))
            else:
                cursor.execute("""
                    SELECT * FROM cells
                    WHERE status = ?
                    ORDER BY fitness DESC
                    LIMIT ?
                """, (status, limit))
            return [self._row_to_cell(row) for row in cursor.fetchall()]

    def get_lineage(self, cell_id: int) -> List[Cell]:
        """
        Get complete ancestry of a cell (from seed to this cell).

        Uses recursive query to trace parent_cell_id back to root.

        Args:
            cell_id: ID of the cell to trace

        Returns:
            List of Cell objects from oldest ancestor to specified cell
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                WITH RECURSIVE lineage AS (
                    SELECT * FROM cells WHERE cell_id = ?
                    UNION ALL
                    SELECT c.* FROM cells c
                    INNER JOIN lineage l ON c.cell_id = l.parent_cell_id
                )
                SELECT * FROM lineage ORDER BY generation ASC
            """, (cell_id,))
            return [self._row_to_cell(row) for row in cursor.fetchall()]

    def find_unanalyzed_cells(self, limit: int = 50, min_fitness: float = 0.0) -> List[Cell]:
        """
        Find cells that need LLM analysis.

        Args:
            limit: Maximum number of cells to return
            min_fitness: Minimum fitness threshold for analysis

        Returns:
            List of Cell objects without LLM analysis, sorted by fitness descending
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM cells
                WHERE llm_name IS NULL
                  AND status = 'online'
                  AND fitness >= ?
                ORDER BY fitness DESC
                LIMIT ?
            """, (min_fitness, limit))
            return [self._row_to_cell(row) for row in cursor.fetchall()]

    def update_cell_llm_analysis(
        self,
        cell_id: int,
        llm_name: str,
        llm_category: str,
        llm_hypothesis: str,
        llm_confidence: float,
    ) -> None:
        """
        Store LLM analysis results for a cell.

        Args:
            cell_id: ID of the cell
            llm_name: LLM-assigned pattern name (e.g., "Volume Spike Reversal")
            llm_category: Pattern category (e.g., "Volume Analysis")
            llm_hypothesis: LLM's interpretation of why it works
            llm_confidence: Confidence score (0.0 to 1.0)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE cells
                SET llm_name = ?,
                    llm_category = ?,
                    llm_hypothesis = ?,
                    llm_confidence = ?,
                    llm_analyzed_at = CURRENT_TIMESTAMP
                WHERE cell_id = ?
            """, (llm_name, llm_category, llm_hypothesis, llm_confidence, cell_id))

    def deprecate_cell(
        self,
        cell_id: int,
        reason: str,
        superseded_by_cell_id: Optional[int] = None,
    ) -> None:
        """
        Mark a cell as deprecated.

        Args:
            cell_id: ID of the cell to deprecate
            reason: Reason for deprecation
            superseded_by_cell_id: ID of the cell that replaced this one (if any)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE cells
                SET status = 'deprecated',
                    deprecated_reason = ?,
                    superseded_by_cell_id = ?
                WHERE cell_id = ?
            """, (reason, superseded_by_cell_id, cell_id))

    # CRUD Methods for Phenotypes
    def store_phenotype(self, phenotype: CellPhenotype) -> int:
        """
        Store market behavior data for a cell.

        Args:
            phenotype: CellPhenotype object with test results

        Returns:
            phenotype_id: ID of the stored phenotype
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cell_phenotypes (
                    cell_id, symbol, timeframe,
                    data_start_date, data_end_date,
                    total_trades, profitable_trades, losing_trades,
                    total_profit, total_fees, max_drawdown, max_runup,
                    sharpe_ratio, sortino_ratio, win_rate, profit_factor,
                    avg_trade_duration_hours, avg_profit_per_trade, avg_loss_per_trade,
                    longest_winning_streak, longest_losing_streak,
                    trigger_conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                phenotype.cell_id, phenotype.symbol, phenotype.timeframe,
                phenotype.data_start_date, phenotype.data_end_date,
                phenotype.total_trades, phenotype.profitable_trades, phenotype.losing_trades,
                phenotype.total_profit, phenotype.total_fees, phenotype.max_drawdown, phenotype.max_runup,
                phenotype.sharpe_ratio, phenotype.sortino_ratio, phenotype.win_rate, phenotype.profit_factor,
                phenotype.avg_trade_duration_hours, phenotype.avg_profit_per_trade, phenotype.avg_loss_per_trade,
                phenotype.longest_winning_streak, phenotype.longest_losing_streak,
                phenotype.trigger_conditions,
            ))
            return cursor.lastrowid

    def get_phenotypes(self, cell_id: int) -> List[CellPhenotype]:
        """
        Get all phenotypes for a cell.

        Args:
            cell_id: ID of the cell

        Returns:
            List of CellPhenotype objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM cell_phenotypes
                WHERE cell_id = ?
                ORDER BY timeframe
            """, (cell_id,))
            return [self._row_to_phenotype(row) for row in cursor.fetchall()]

    # CRUD Methods for Failed Mutations
    def record_failed_mutation(
        self,
        parent_cell_id: int,
        attempted_genome: str,
        generation: int,
        failure_reason: str,
        fitness_achieved: Optional[float] = None,
    ) -> int:
        """
        Record a failed mutation attempt.

        Args:
            parent_cell_id: ID of the parent cell
            attempted_genome: DSL genome that failed
            generation: Generation number
            failure_reason: Reason for failure ('negative_fitness', 'lower_than_parent', etc.)
            fitness_achieved: Fitness score if calculable

        Returns:
            failure_id: ID of the failure record
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO failed_mutations (
                    parent_cell_id, attempted_genome, generation, failure_reason, fitness_achieved
                ) VALUES (?, ?, ?, ?, ?)
            """, (parent_cell_id, attempted_genome, generation, failure_reason, fitness_achieved))
            return cursor.lastrowid

    def get_failure_statistics(self, parent_cell_id: Optional[int] = None) -> dict:
        """
        Get failure statistics.

        Args:
            parent_cell_id: Optional parent cell ID to filter by

        Returns:
            Dictionary with failure counts by reason
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if parent_cell_id:
                cursor.execute("""
                    SELECT failure_reason, COUNT(*) as count
                    FROM failed_mutations
                    WHERE parent_cell_id = ?
                    GROUP BY failure_reason
                """, (parent_cell_id,))
            else:
                cursor.execute("""
                    SELECT failure_reason, COUNT(*) as count
                    FROM failed_mutations
                    GROUP BY failure_reason
                """)
            return {row['failure_reason']: row['count'] for row in cursor.fetchall()}

    # CRUD Methods for Evolution Runs
    def start_evolution_run(
        self,
        run_type: str,
        max_generations: Optional[int] = None,
        fitness_goal: Optional[float] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        initial_capital: Optional[float] = None,
        transaction_fee_rate: Optional[float] = None,
    ) -> int:
        """
        Start a new evolution run.

        Args:
            run_type: 'evolution' or 'trading-learn'
            max_generations: Maximum generations to run
            fitness_goal: Target fitness to achieve
            symbol: Trading symbol (e.g., 'PURR')
            timeframe: Primary timeframe (e.g., '1H')
            initial_capital: Starting capital
            transaction_fee_rate: Transaction fee rate

        Returns:
            run_id: ID of the evolution run
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evolution_runs (
                    run_type, started_at,
                    max_generations, fitness_goal, symbol, timeframe,
                    initial_capital, transaction_fee_rate
                ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
            """, (run_type, max_generations, fitness_goal, symbol, timeframe, initial_capital, transaction_fee_rate))
            return cursor.lastrowid

    def complete_evolution_run(
        self,
        run_id: int,
        best_cell_id: int,
        total_cells_birthed: int,
        total_mutations_failed: int,
        final_best_fitness: float,
        termination_reason: str,
        generations_without_improvement: int,
        total_llm_cost: float = 0.0,
        total_tokens_used: int = 0,
    ) -> None:
        """
        Complete an evolution run with final statistics.

        Args:
            run_id: ID of the evolution run
            best_cell_id: ID of the best cell found
            total_cells_birthed: Total number of successful cells
            total_mutations_failed: Total number of failed mutations
            final_best_fitness: Best fitness achieved
            termination_reason: Why the run ended
            generations_without_improvement: Stagnation count
            total_llm_cost: Total LLM costs (for trading-learn mode)
            total_tokens_used: Total tokens used
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get start time to calculate duration
            cursor.execute("SELECT started_at FROM evolution_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                started_at = datetime.fromisoformat(row['started_at'])
                duration_seconds = (datetime.now() - started_at).total_seconds()
            else:
                duration_seconds = None

            cursor.execute("""
                UPDATE evolution_runs
                SET completed_at = CURRENT_TIMESTAMP,
                    duration_seconds = ?,
                    best_cell_id = ?,
                    total_cells_birthed = ?,
                    total_mutations_failed = ?,
                    final_best_fitness = ?,
                    termination_reason = ?,
                    generations_without_improvement = ?,
                    total_llm_cost = ?,
                    total_tokens_used = ?
                WHERE run_id = ?
            """, (
                duration_seconds, best_cell_id, total_cells_birthed, total_mutations_failed,
                final_best_fitness, termination_reason, generations_without_improvement,
                total_llm_cost, total_tokens_used, run_id
            ))

    def get_evolution_run(self, run_id: int) -> Optional[EvolutionRun]:
        """
        Get evolution run by ID.

        Args:
            run_id: ID of the evolution run

        Returns:
            EvolutionRun object or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evolution_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            return self._row_to_evolution_run(row) if row else None

    # CRUD Methods for Patterns
    def create_pattern(
        self,
        pattern_name: str,
        category: str,
        description: str,
        discovered_by_cell_id: int,
        typical_dsl_structure: Optional[str] = None,
        required_indicators: Optional[str] = None,
        typical_parameters: Optional[str] = None,
    ) -> int:
        """
        Create a new discovered pattern.

        Args:
            pattern_name: Human-readable pattern name
            category: Pattern category
            description: LLM's explanation of the pattern
            discovered_by_cell_id: ID of the cell that first exhibited this pattern
            typical_dsl_structure: Typical DSL structure (optional)
            required_indicators: JSON array of required indicators (optional)
            typical_parameters: JSON object of typical parameters (optional)

        Returns:
            pattern_id: ID of the created pattern
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO discovered_patterns (
                    pattern_name, category, description, discovered_by_cell_id,
                    typical_dsl_structure, required_indicators, typical_parameters
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_name, category, description, discovered_by_cell_id,
                typical_dsl_structure, required_indicators, typical_parameters
            ))
            return cursor.lastrowid

    def link_cell_to_pattern(
        self,
        cell_id: int,
        pattern_id: int,
        confidence: float = 1.0,
    ) -> None:
        """
        Link a cell to a pattern.

        Args:
            cell_id: ID of the cell
            pattern_id: ID of the pattern
            confidence: Confidence of the pattern match (0.0 to 1.0)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO cell_patterns (cell_id, pattern_id, confidence)
                VALUES (?, ?, ?)
            """, (cell_id, pattern_id, confidence))

            # Update pattern statistics
            cursor.execute("""
                UPDATE discovered_patterns
                SET cells_using_pattern = (
                    SELECT COUNT(*) FROM cell_patterns WHERE pattern_id = ?
                ),
                avg_fitness = (
                    SELECT AVG(c.fitness)
                    FROM cells c
                    INNER JOIN cell_patterns cp ON c.cell_id = cp.cell_id
                    WHERE cp.pattern_id = ?
                ),
                best_fitness = (
                    SELECT MAX(c.fitness)
                    FROM cells c
                    INNER JOIN cell_patterns cp ON c.cell_id = cp.cell_id
                    WHERE cp.pattern_id = ?
                ),
                worst_fitness = (
                    SELECT MIN(c.fitness)
                    FROM cells c
                    INNER JOIN cell_patterns cp ON c.cell_id = cp.cell_id
                    WHERE cp.pattern_id = ?
                ),
                best_cell_id = (
                    SELECT c.cell_id
                    FROM cells c
                    INNER JOIN cell_patterns cp ON c.cell_id = cp.cell_id
                    WHERE cp.pattern_id = ?
                    ORDER BY c.fitness DESC
                    LIMIT 1
                )
                WHERE pattern_id = ?
            """, (pattern_id, pattern_id, pattern_id, pattern_id, pattern_id, pattern_id))

    def get_patterns_by_category(self, category: str) -> List[DiscoveredPattern]:
        """
        Get all patterns in a category.

        Args:
            category: Pattern category

        Returns:
            List of DiscoveredPattern objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM discovered_patterns
                WHERE category = ?
                ORDER BY avg_fitness DESC
            """, (category,))
            return [self._row_to_pattern(row) for row in cursor.fetchall()]

    def get_cells_by_pattern(self, pattern_id: int) -> List[Cell]:
        """
        Get all cells that use a specific pattern.

        Args:
            pattern_id: ID of the pattern

        Returns:
            List of Cell objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.* FROM cells c
                INNER JOIN cell_patterns cp ON c.cell_id = cp.cell_id
                WHERE cp.pattern_id = ?
                ORDER BY c.fitness DESC
            """, (pattern_id,))
            return [self._row_to_cell(row) for row in cursor.fetchall()]

    def get_cell_count(self) -> int:
        """Get total number of cells in database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM cells")
            return cursor.fetchone()['count']

    def store_mutation_proposal(
        self,
        cell_id: int,
        proposed_genome: str,
        rationale: str,
        confidence: str,
        expected_improvement: str,
        actual_fitness_change: float,
    ) -> int:
        """Store LLM mutation proposal."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cell_mutation_proposals (
                    cell_id, proposed_genome, mutation_rationale, status, tested_at, result_cell_id
                ) VALUES (?, ?, ?, 'tested', CURRENT_TIMESTAMP, ?)
            """, (cell_id, proposed_genome, f"{rationale} | confidence:{confidence} | expected:{expected_improvement} | actual_change:${actual_fitness_change:.2f}", cell_id))
            return cursor.lastrowid

    def get_mutation_proposals_for_cell(self, cell_id: int, status: Optional[str] = None) -> List[dict]:
        """
        Get mutation proposals for a cell.

        Args:
            cell_id: ID of the cell
            status: Optional status filter ('pending', 'tested', 'abandoned')

        Returns:
            List of mutation proposal dicts with keys:
            - proposal_id, cell_id, proposed_genome, mutation_rationale,
              priority, status, proposed_at, tested_at, result_cell_id
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute("""
                    SELECT * FROM cell_mutation_proposals
                    WHERE cell_id = ? AND status = ?
                    ORDER BY proposed_at DESC
                """, (cell_id, status))
            else:
                cursor.execute("""
                    SELECT * FROM cell_mutation_proposals
                    WHERE cell_id = ?
                    ORDER BY proposed_at DESC
                """, (cell_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_phenotypes_for_cell(self, cell_id: int) -> List[CellPhenotype]:
        """Alias for get_phenotypes (for consistency with new code)."""
        return self.get_phenotypes(cell_id)
