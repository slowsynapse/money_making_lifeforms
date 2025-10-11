# Cell Storage API: Python Interface

## Overview

The `CellRepository` class provides a clean Python interface to the cell database. It handles:
- Cell lifecycle (birth, deprecation, status changes)
- Lineage queries
- Pattern management
- Phenotype recording
- Transaction safety

**Location**: `base_agent/src/storage/cell_repository.py` (to be implemented)

## Basic Usage

```python
from pathlib import Path
from base_agent.src.storage.cell_repository import CellRepository

# Initialize repository
db_path = Path("/home/agent/workdir/evolution/cells.db")
repo = CellRepository(db_path)

# Birth a new cell
cell_id = repo.birth_cell(
    genome="IF EPSILON(0)/EPSILON(20) > 1.5 THEN BUY ELSE HOLD",
    fitness=23.31,
    generation=47,
    parent_id=15
)

# Get cell details
cell = repo.get_cell(cell_id)
print(f"Cell #{cell.id}: {cell.genome}")
print(f"Fitness: ${cell.fitness:.2f}")

# Query top performers
top_cells = repo.get_top_cells(limit=10)
for cell in top_cells:
    print(f"Cell #{cell.id}: ${cell.fitness:.2f}")
```

## Core Classes

### CellRepository

Main interface to the database.

```python
class CellRepository:
    def __init__(self, db_path: Path):
        """Initialize connection to cell database."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Dict-like access
        self._ensure_schema()

    def _ensure_schema(self):
        """Create tables if they don't exist."""
        # Execute CREATE TABLE statements from DATABASE_SCHEMA.md
```

### Cell (Data Class)

Represents a single cell.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Cell:
    cell_id: int
    generation: int
    parent_cell_id: Optional[int]
    dsl_genome: str
    fitness: float
    status: str  # 'online', 'deprecated', 'archived', 'extinct'
    created_at: datetime

    # LLM semantics
    llm_name: Optional[str] = None
    llm_category: Optional[str] = None
    llm_hypothesis: Optional[str] = None
    llm_confidence: Optional[float] = None
    llm_analyzed_at: Optional[datetime] = None

    # Deprecation info
    deprecated_reason: Optional[str] = None
    superseded_by_cell_id: Optional[int] = None
```

### CellPhenotype (Data Class)

Represents market behavior of a cell.

```python
@dataclass
class CellPhenotype:
    phenotype_id: int
    cell_id: int
    symbol: str
    timeframe: str

    # Trading metrics
    total_trades: int
    profitable_trades: int
    losing_trades: int

    # Financial
    total_profit: float
    total_fees: float
    max_drawdown: Optional[float] = None
    max_runup: Optional[float] = None

    # Risk
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None

    # Trade characteristics
    avg_trade_duration_hours: Optional[float] = None
    avg_profit_per_trade: Optional[float] = None
    avg_loss_per_trade: Optional[float] = None

    # Trigger data
    trigger_conditions: Optional[str] = None  # JSON string
    tested_at: datetime = datetime.now()
```

### DiscoveredPattern (Data Class)

Represents an LLM-named pattern.

```python
@dataclass
class DiscoveredPattern:
    pattern_id: int
    pattern_name: str
    category: str
    description: str

    # Characteristics
    typical_dsl_structure: Optional[str] = None
    required_indicators: Optional[str] = None  # JSON array

    # Performance
    cells_using_pattern: int = 0
    avg_fitness: Optional[float] = None
    best_fitness: Optional[float] = None
    best_cell_id: Optional[int] = None

    discovered_at: datetime = datetime.now()
    discovered_by_cell_id: Optional[int] = None
```

## Cell Lifecycle Methods

### birth_cell()

Create a new cell from a successful mutation.

```python
def birth_cell(
    self,
    genome: str,
    fitness: float,
    generation: int,
    parent_id: Optional[int] = None,
    status: str = 'online'
) -> int:
    """
    Birth a new cell (successful mutation).

    Args:
        genome: DSL strategy string
        fitness: Economic performance score
        generation: Which generation this cell was born
        parent_id: Parent cell ID (None for Gen 0)
        status: Initial status (default: 'online')

    Returns:
        cell_id: Unique identifier for the new cell
    """
    cursor = self.conn.cursor()
    cursor.execute("""
        INSERT INTO cells (
            dsl_genome, fitness, generation, parent_cell_id, status
        ) VALUES (?, ?, ?, ?, ?)
    """, (genome, fitness, generation, parent_id, status))

    self.conn.commit()
    cell_id = cursor.lastrowid

    return cell_id
```

**Usage:**
```python
# After a mutation improves fitness
child_id = repo.birth_cell(
    genome="IF EPSILON(0)/AVG(EPSILON, 0, 20) > 2.0 THEN BUY ELSE HOLD",
    fitness=25.67,
    generation=parent.generation + 1,
    parent_id=parent.cell_id
)
```

### record_failure()

Log a failed mutation (statistics only, not a cell).

```python
def record_failure(
    self,
    parent_id: int,
    attempted_dsl: str,
    fitness: float,
    failure_reason: str,
    mutation_type: Optional[str] = None,
    error_message: Optional[str] = None
) -> None:
    """
    Record a failed mutation attempt.

    Args:
        parent_id: Which cell was mutated
        attempted_dsl: The DSL that failed
        fitness: Score achieved (always <= 0 or <= parent)
        failure_reason: Why it failed
        mutation_type: 'modify_rule', 'add_rule', 'remove_rule'
        error_message: Stack trace if crashed
    """
    parent = self.get_cell(parent_id)

    cursor = self.conn.cursor()
    cursor.execute("""
        INSERT INTO failed_mutations (
            generation, parent_cell_id, attempted_dsl, fitness,
            failure_reason, mutation_type, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        parent.generation + 1, parent_id, attempted_dsl, fitness,
        failure_reason, mutation_type, error_message
    ))

    self.conn.commit()
```

**Usage:**
```python
# After a mutation fails
repo.record_failure(
    parent_id=47,
    attempted_dsl="IF EPSILON(0)/EPSILON(0) > 1.5 THEN BUY ELSE HOLD",  # Division by zero
    fitness=-1000.0,
    failure_reason="Parse error: division by zero",
    mutation_type="modify_rule",
    error_message="ZeroDivisionError: division by zero at line 42"
)
```

### update_cell_status()

Change cell lifecycle status.

```python
def update_cell_status(
    self,
    cell_id: int,
    new_status: str,
    reason: Optional[str] = None,
    superseded_by: Optional[int] = None
) -> None:
    """
    Update cell status (online → deprecated/archived/extinct).

    Args:
        cell_id: Cell to update
        new_status: 'online', 'deprecated', 'archived', 'extinct'
        reason: Why the status changed
        superseded_by: If deprecated, which cell replaced it
    """
    cursor = self.conn.cursor()
    cursor.execute("""
        UPDATE cells
        SET status = ?,
            deprecated_reason = ?,
            superseded_by_cell_id = ?
        WHERE cell_id = ?
    """, (new_status, reason, superseded_by, cell_id))

    self.conn.commit()
```

**Usage:**
```python
# When a better version of a pattern is found
repo.update_cell_status(
    cell_id=15,
    new_status='deprecated',
    reason='Improved by Cell #47 with additional DELTA condition',
    superseded_by=47
)
```

## Query Methods

### get_cell()

Retrieve a single cell by ID.

```python
def get_cell(self, cell_id: int) -> Cell:
    """Get cell by ID."""
    cursor = self.conn.cursor()
    cursor.execute("SELECT * FROM cells WHERE cell_id = ?", (cell_id,))
    row = cursor.fetchone()

    if not row:
        raise ValueError(f"Cell #{cell_id} not found")

    return Cell(
        cell_id=row['cell_id'],
        generation=row['generation'],
        parent_cell_id=row['parent_cell_id'],
        dsl_genome=row['dsl_genome'],
        fitness=row['fitness'],
        status=row['status'],
        created_at=datetime.fromisoformat(row['created_at']),
        llm_name=row['llm_name'],
        llm_category=row['llm_category'],
        llm_hypothesis=row['llm_hypothesis'],
        llm_confidence=row['llm_confidence'],
        llm_analyzed_at=datetime.fromisoformat(row['llm_analyzed_at']) if row['llm_analyzed_at'] else None,
        deprecated_reason=row['deprecated_reason'],
        superseded_by_cell_id=row['superseded_by_cell_id']
    )
```

### get_top_cells()

Get highest-fitness cells.

```python
def get_top_cells(
    self,
    limit: int = 10,
    status: str = 'online',
    min_fitness: Optional[float] = None
) -> list[Cell]:
    """
    Get top performing cells.

    Args:
        limit: How many to return
        status: Filter by status (default: 'online')
        min_fitness: Only return cells with fitness >= this

    Returns:
        List of Cell objects, sorted by fitness descending
    """
    cursor = self.conn.cursor()

    query = "SELECT * FROM cells WHERE status = ?"
    params = [status]

    if min_fitness is not None:
        query += " AND fitness >= ?"
        params.append(min_fitness)

    query += " ORDER BY fitness DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    return [self._row_to_cell(row) for row in rows]
```

**Usage:**
```python
# Get top 10 online cells
top = repo.get_top_cells(limit=10, status='online')

for cell in top:
    print(f"#{cell.cell_id}: {cell.llm_name or 'Unnamed'} - ${cell.fitness:.2f}")
```

### get_lineage()

Get ancestry chain of a cell.

```python
def get_lineage(self, cell_id: int) -> list[Cell]:
    """
    Get full ancestry of a cell (recursive parent lookup).

    Returns:
        List of Cell objects from oldest ancestor to target cell
    """
    cursor = self.conn.cursor()
    cursor.execute("""
        WITH RECURSIVE lineage AS (
            SELECT * FROM cells WHERE cell_id = ?
            UNION ALL
            SELECT c.* FROM cells c
            JOIN lineage l ON c.cell_id = l.parent_cell_id
        )
        SELECT * FROM lineage ORDER BY generation ASC
    """, (cell_id,))

    rows = cursor.fetchall()
    return [self._row_to_cell(row) for row in rows]
```

**Usage:**
```python
# Trace ancestry
lineage = repo.get_lineage(cell_id=47)

print("Ancestry:")
for i, ancestor in enumerate(lineage):
    indent = "  " * i
    print(f"{indent}└─ Cell #{ancestor.cell_id} (Gen {ancestor.generation}): ${ancestor.fitness:.2f}")
```

Output:
```
Ancestry:
└─ Cell #1 (Gen 0): $6.17
  └─ Cell #5 (Gen 12): $15.32
    └─ Cell #15 (Gen 34): $18.45
      └─ Cell #47 (Gen 89): $23.31
```

### find_unanalyzed_cells()

Get cells that need LLM analysis.

```python
def find_unanalyzed_cells(
    self,
    limit: int = 50,
    min_fitness: float = 0.0
) -> list[Cell]:
    """
    Find cells that survived but haven't been analyzed by LLM yet.

    Args:
        limit: Max number to return
        min_fitness: Only return cells with fitness >= this

    Returns:
        List of Cell objects, sorted by fitness descending
    """
    cursor = self.conn.cursor()
    cursor.execute("""
        SELECT * FROM cells
        WHERE status = 'online'
          AND llm_analyzed_at IS NULL
          AND fitness >= ?
        ORDER BY fitness DESC
        LIMIT ?
    """, (min_fitness, limit))

    rows = cursor.fetchall()
    return [self._row_to_cell(row) for row in rows]
```

**Usage:**
```python
# Get work queue for trading-learn mode
unanalyzed = repo.find_unanalyzed_cells(limit=10, min_fitness=5.0)

print(f"Found {len(unanalyzed)} cells needing analysis:")
for cell in unanalyzed:
    print(f"  Cell #{cell.cell_id}: {cell.genome} (${cell.fitness:.2f})")
```

## Phenotype Methods

### record_phenotype()

Store market behavior data for a cell.

```python
def record_phenotype(
    self,
    cell_id: int,
    symbol: str,
    timeframe: str,
    total_trades: int,
    profitable_trades: int,
    losing_trades: int,
    total_profit: float,
    total_fees: float,
    **kwargs  # Additional optional metrics
) -> int:
    """
    Record phenotype (market behavior) for a cell.

    Args:
        cell_id: Which cell
        symbol: Trading pair (e.g., 'PURR')
        timeframe: Candle interval (e.g., '1h')
        total_trades: Number of trades executed
        profitable_trades: Winning trades
        losing_trades: Losing trades
        total_profit: Total P&L before fees
        total_fees: Transaction costs
        **kwargs: max_drawdown, sharpe_ratio, win_rate, etc.

    Returns:
        phenotype_id: Unique identifier for this phenotype record
    """
    cursor = self.conn.cursor()

    # Calculate derived metrics
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0.0

    cursor.execute("""
        INSERT INTO cell_phenotypes (
            cell_id, symbol, timeframe,
            total_trades, profitable_trades, losing_trades,
            total_profit, total_fees, win_rate,
            max_drawdown, sharpe_ratio, trigger_conditions
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        cell_id, symbol, timeframe,
        total_trades, profitable_trades, losing_trades,
        total_profit, total_fees, win_rate,
        kwargs.get('max_drawdown'),
        kwargs.get('sharpe_ratio'),
        kwargs.get('trigger_conditions')  # JSON string
    ))

    self.conn.commit()
    return cursor.lastrowid
```

**Usage:**
```python
# After backtesting a cell
phenotype_id = repo.record_phenotype(
    cell_id=47,
    symbol='PURR',
    timeframe='1h',
    total_trades=45,
    profitable_trades=28,
    losing_trades=17,
    total_profit=31.38,
    total_fees=8.05,
    max_drawdown=-5.23,
    sharpe_ratio=1.47,
    trigger_conditions=json.dumps({
        "buy_triggers": [...],
        "sell_triggers": [...]
    })
)
```

### get_phenotypes()

Get all phenotypes for a cell.

```python
def get_phenotypes(self, cell_id: int) -> list[CellPhenotype]:
    """Get all phenotype records for a cell."""
    cursor = self.conn.cursor()
    cursor.execute("""
        SELECT * FROM cell_phenotypes
        WHERE cell_id = ?
        ORDER BY tested_at DESC
    """, (cell_id,))

    rows = cursor.fetchall()
    return [self._row_to_phenotype(row) for row in rows]
```

## LLM Analysis Methods

### update_llm_analysis()

Store LLM's interpretation of a cell.

```python
def update_llm_analysis(
    self,
    cell_id: int,
    name: str,
    category: str,
    hypothesis: str,
    confidence: float = 1.0
) -> None:
    """
    Store LLM's analysis of a cell.

    Args:
        cell_id: Which cell was analyzed
        name: Human-readable name (e.g., "Volume Spike Reversal")
        category: Pattern category (e.g., "Volume Analysis")
        hypothesis: Why the LLM thinks it works
        confidence: LLM's confidence in analysis (0.0 - 1.0)
    """
    cursor = self.conn.cursor()
    cursor.execute("""
        UPDATE cells
        SET llm_name = ?,
            llm_category = ?,
            llm_hypothesis = ?,
            llm_confidence = ?,
            llm_analyzed_at = CURRENT_TIMESTAMP
        WHERE cell_id = ?
    """, (name, category, hypothesis, confidence, cell_id))

    self.conn.commit()
```

**Usage:**
```python
# After LLM analyzes a cell
repo.update_llm_analysis(
    cell_id=47,
    name="Volume Spike Reversal Detector",
    category="Volume Analysis",
    hypothesis="Detects institutional accumulation during local price dips. The 2x volume spike above 20-period average combined with price below 10-period value suggests large buyer entering.",
    confidence=0.85
)
```

## Pattern Methods

### create_pattern()

Register a new discovered pattern.

```python
def create_pattern(
    self,
    name: str,
    category: str,
    description: str,
    discovered_by_cell_id: Optional[int] = None,
    **kwargs
) -> int:
    """
    Create a new pattern in the taxonomy.

    Args:
        name: Pattern name (must be unique)
        category: Pattern category
        description: LLM's explanation
        discovered_by_cell_id: First cell exhibiting this pattern
        **kwargs: typical_dsl_structure, required_indicators, etc.

    Returns:
        pattern_id: Unique identifier
    """
    cursor = self.conn.cursor()
    cursor.execute("""
        INSERT INTO discovered_patterns (
            pattern_name, category, description,
            discovered_by_cell_id, typical_dsl_structure,
            required_indicators
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        name, category, description,
        discovered_by_cell_id,
        kwargs.get('typical_dsl_structure'),
        kwargs.get('required_indicators')  # JSON array
    ))

    self.conn.commit()
    return cursor.lastrowid
```

### link_cell_to_pattern()

Associate a cell with a pattern.

```python
def link_cell_to_pattern(
    self,
    cell_id: int,
    pattern_id: int,
    confidence: float = 1.0
) -> None:
    """
    Link a cell to a pattern (many-to-many).

    Args:
        cell_id: The cell
        pattern_id: The pattern
        confidence: How strongly does cell exhibit pattern (0.0-1.0)
    """
    cursor = self.conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO cell_patterns (
            cell_id, pattern_id, confidence, assigned_by
        ) VALUES (?, ?, ?, 'llm')
    """, (cell_id, pattern_id, confidence))

    self.conn.commit()
```

**Usage:**
```python
# After LLM identifies a pattern
pattern_id = repo.create_pattern(
    name="Volume Spike Reversal",
    category="Volume Analysis",
    description="Buys when volume spikes significantly above average during price dips",
    discovered_by_cell_id=47,
    required_indicators=json.dumps(["EPSILON", "DELTA"])
)

# Link cell to pattern
repo.link_cell_to_pattern(cell_id=47, pattern_id=pattern_id, confidence=0.95)
```

### find_cells_by_pattern()

Get all cells exhibiting a pattern.

```python
def find_cells_by_pattern(
    self,
    pattern_name: str,
    status: str = 'online'
) -> list[Cell]:
    """Find all cells tagged with a specific pattern."""
    cursor = self.conn.cursor()
    cursor.execute("""
        SELECT c.* FROM cells c
        JOIN cell_patterns cp ON c.cell_id = cp.cell_id
        JOIN discovered_patterns dp ON cp.pattern_id = dp.pattern_id
        WHERE dp.pattern_name = ? AND c.status = ?
        ORDER BY c.fitness DESC
    """, (pattern_name, status))

    rows = cursor.fetchall()
    return [self._row_to_cell(row) for row in rows]
```

## Transaction Safety

### Context Manager Support

```python
class CellRepository:
    def begin_transaction(self):
        """Start a transaction."""
        self.conn.execute("BEGIN")

    def commit(self):
        """Commit current transaction."""
        self.conn.commit()

    def rollback(self):
        """Rollback current transaction."""
        self.conn.rollback()

    def __enter__(self):
        """Context manager entry."""
        self.begin_transaction()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic rollback on error."""
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
```

**Usage:**
```python
# Atomic operation: birth cell + record phenotype
with repo:
    cell_id = repo.birth_cell(genome="...", fitness=23.31, generation=47, parent_id=15)
    repo.record_phenotype(cell_id=cell_id, symbol='PURR', ...)
    # If any error occurs, both operations are rolled back
```

## Complete Example: Evolution Loop Integration

```python
from base_agent.src.storage.cell_repository import CellRepository
from base_agent.src.dsl.mutator import DslMutator
from base_agent.src.dsl.interpreter import DslInterpreter

# Initialize
repo = CellRepository(Path("/home/agent/workdir/evolution/cells.db"))
mutator = DslMutator()
interpreter = DslInterpreter()

# Get current best cell
current_best = repo.get_top_cells(limit=1)[0]

# Mutate
mutated_program = mutator.mutate(interpreter.parse(current_best.genome))
mutated_genome = mutator.to_string(mutated_program)

# Backtest (pseudo-code)
fitness = run_backtest(mutated_genome)

# Decision: birth or record failure?
if fitness > current_best.fitness:
    # Success! Birth a new cell
    with repo:
        child_id = repo.birth_cell(
            genome=mutated_genome,
            fitness=fitness,
            generation=current_best.generation + 1,
            parent_id=current_best.cell_id
        )

        repo.record_phenotype(
            cell_id=child_id,
            symbol='PURR',
            timeframe='1h',
            # ... phenotype data
        )

    print(f"✓ Cell #{child_id} born! Fitness: ${fitness:.2f}")
else:
    # Failure - log it
    repo.record_failure(
        parent_id=current_best.cell_id,
        attempted_dsl=mutated_genome,
        fitness=fitness,
        failure_reason=f"Fitness {fitness:.2f} <= parent {current_best.fitness:.2f}",
        mutation_type="modify_rule"
    )

    print(f"✗ Mutation failed: ${fitness:.2f} <= ${current_best.fitness:.2f}")
```

## Summary

The `CellRepository` provides:
- **Simple API** for cell lifecycle management
- **Type-safe** data classes for cells, phenotypes, patterns
- **Transaction safety** via context managers
- **Efficient queries** for common operations
- **Extensible** via metadata and pattern linking

Next: See `LLM_INTEGRATION.md` for how the LLM uses this API to analyze cells and discover patterns.
