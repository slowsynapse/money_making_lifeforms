# Database Schema: Cell Storage

## Overview

The system uses **SQLite** for local, zero-configuration cell storage. The database contains:
- All birthed cells (successful strategies)
- Failed mutation statistics
- Phenotype data (market behavior)
- LLM-discovered patterns
- Evolution run metadata

**Location**: `/home/agent/workdir/evolution/cells.db` (inside Docker) or `results/interactive_output/evolution/cells.db` (on host)

## Why SQLite?

- **Single file**: Easy backup, version control, sharing
- **Zero config**: No server to run
- **ACID transactions**: Safe for concurrent reads
- **Fast**: Sufficient for 10,000+ cells
- **Portable**: Works identically in Docker and on host
- **Embedded**: No network overhead

## Complete Schema

### Table: `cells`

**Primary entity** - Each successful strategy

```sql
CREATE TABLE cells (
    -- Identity
    cell_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation INTEGER NOT NULL,
    parent_cell_id INTEGER,  -- NULL for Gen 0
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Genome (immutable)
    dsl_genome TEXT NOT NULL,

    -- Fitness (context-dependent)
    fitness REAL NOT NULL,

    -- Lifecycle
    status TEXT DEFAULT 'online',  -- 'online', 'deprecated', 'archived', 'extinct'
    deprecated_reason TEXT,
    superseded_by_cell_id INTEGER,  -- If deprecated, which cell replaced it?

    -- LLM Semantics (populated by trading-learn mode)
    llm_name TEXT,           -- e.g., "Volume Spike Reversal Detector"
    llm_category TEXT,       -- e.g., "Volume Analysis"
    llm_hypothesis TEXT,     -- Why the LLM thinks it works
    llm_confidence REAL,     -- 0.0 - 1.0 confidence in analysis
    llm_analyzed_at TIMESTAMP,

    -- Foreign keys
    FOREIGN KEY (parent_cell_id) REFERENCES cells(cell_id),
    FOREIGN KEY (superseded_by_cell_id) REFERENCES cells(cell_id),

    -- Constraints
    CHECK (status IN ('online', 'deprecated', 'archived', 'extinct')),
    CHECK (llm_confidence IS NULL OR (llm_confidence >= 0.0 AND llm_confidence <= 1.0))
);
```

**Indexes**:
```sql
CREATE INDEX idx_cells_fitness ON cells(fitness DESC);
CREATE INDEX idx_cells_generation ON cells(generation);
CREATE INDEX idx_cells_status ON cells(status);
CREATE INDEX idx_cells_parent ON cells(parent_cell_id);
CREATE INDEX idx_cells_llm_analyzed ON cells(llm_analyzed_at);
CREATE INDEX idx_cells_category ON cells(llm_category);
```

### Table: `cell_phenotypes`

**Market behavior** - How a genome performs

```sql
CREATE TABLE cell_phenotypes (
    phenotype_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_id INTEGER NOT NULL,

    -- Test context
    symbol TEXT NOT NULL,           -- e.g., "PURR"
    timeframe TEXT NOT NULL,        -- e.g., "1h"
    data_start_date TEXT,           -- ISO8601: "2025-08-10"
    data_end_date TEXT,

    -- Trading performance
    total_trades INTEGER NOT NULL,
    profitable_trades INTEGER NOT NULL,
    losing_trades INTEGER NOT NULL,

    -- Financial metrics
    total_profit REAL NOT NULL,
    total_fees REAL NOT NULL,
    max_drawdown REAL,
    max_runup REAL,

    -- Risk metrics
    sharpe_ratio REAL,
    sortino_ratio REAL,
    win_rate REAL,                  -- profitable_trades / total_trades
    profit_factor REAL,             -- total_profit / total_loss

    -- Trade characteristics
    avg_trade_duration_hours REAL,
    avg_profit_per_trade REAL,
    avg_loss_per_trade REAL,
    longest_winning_streak INTEGER,
    longest_losing_streak INTEGER,

    -- Trigger analysis (for LLM)
    trigger_conditions TEXT,        -- JSON: market states when strategy activated

    -- Metadata
    tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (cell_id) REFERENCES cells(cell_id) ON DELETE CASCADE
);
```

**Indexes**:
```sql
CREATE INDEX idx_phenotypes_cell ON cell_phenotypes(cell_id);
CREATE INDEX idx_phenotypes_symbol ON cell_phenotypes(symbol);
CREATE INDEX idx_phenotypes_sharpe ON cell_phenotypes(sharpe_ratio DESC);
```

**trigger_conditions JSON format**:
```json
{
  "buy_triggers": [
    {
      "timestamp": "2025-09-15T14:00:00Z",
      "price": 0.18234,
      "volume": 125000,
      "indicator_values": {
        "EPSILON(0)": 125000,
        "EPSILON(20)": 62000,
        "DELTA(0)": 0.18234,
        "DELTA(10)": 0.19100
      }
    }
  ],
  "sell_triggers": [ /* ... */ ]
}
```

### Table: `failed_mutations`

**Evolutionary dead-ends** - Statistics only, not full cells

```sql
CREATE TABLE failed_mutations (
    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation INTEGER NOT NULL,
    parent_cell_id INTEGER NOT NULL,

    -- What was attempted
    attempted_dsl TEXT NOT NULL,
    mutation_type TEXT,  -- 'modify_rule', 'add_rule', 'remove_rule'

    -- Why it failed
    fitness REAL NOT NULL,           -- Always <= 0 or <= parent
    failure_reason TEXT,             -- "Fitness worse than parent", "Parse error", etc.
    error_message TEXT,              -- Stack trace if crashed

    -- Metadata
    failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (parent_cell_id) REFERENCES cells(cell_id)
);
```

**Indexes**:
```sql
CREATE INDEX idx_failures_generation ON failed_mutations(generation);
CREATE INDEX idx_failures_parent ON failed_mutations(parent_cell_id);
```

**Use case**: Analyze common failure patterns, mutation types that rarely succeed.

### Table: `evolution_runs`

**Run metadata** - Track complete evolution sessions

```sql
CREATE TABLE evolution_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Run type
    run_type TEXT NOT NULL,  -- 'evolution' (offline) or 'trading-learn' (LLM-powered)

    -- Timing
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL,

    -- Configuration
    max_generations INTEGER,
    fitness_goal REAL,
    symbol TEXT,
    timeframe TEXT,
    initial_capital REAL,
    transaction_fee_rate REAL,

    -- Results
    best_cell_id INTEGER,
    total_cells_birthed INTEGER DEFAULT 0,
    total_mutations_failed INTEGER DEFAULT 0,
    final_best_fitness REAL,

    -- Termination
    termination_reason TEXT,  -- 'goal_reached', 'max_generations', 'stagnation', 'user_stopped'
    generations_without_improvement INTEGER,

    -- Costs (for trading-learn mode)
    total_llm_cost REAL DEFAULT 0.0,
    total_tokens_used INTEGER DEFAULT 0,

    FOREIGN KEY (best_cell_id) REFERENCES cells(cell_id),

    CHECK (run_type IN ('evolution', 'trading-learn'))
);
```

**Indexes**:
```sql
CREATE INDEX idx_runs_type ON evolution_runs(run_type);
CREATE INDEX idx_runs_started ON evolution_runs(started_at DESC);
```

### Table: `discovered_patterns`

**Pattern taxonomy** - LLM-named trading patterns

```sql
CREATE TABLE discovered_patterns (
    pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Pattern identity
    pattern_name TEXT UNIQUE NOT NULL,  -- e.g., "Volume Spike Reversal"
    category TEXT,                      -- e.g., "Volume Analysis", "Mean Reversion"
    description TEXT,                   -- LLM's explanation

    -- Pattern characteristics
    typical_dsl_structure TEXT,         -- Common DSL pattern
    required_indicators TEXT,           -- JSON: ["EPSILON", "DELTA"]
    typical_parameters TEXT,            -- JSON: {"lookback_range": [10, 30]}

    -- Pattern discovery
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    discovered_by_cell_id INTEGER,      -- First cell exhibiting this pattern

    -- Performance tracking
    cells_using_pattern INTEGER DEFAULT 0,
    avg_fitness REAL,
    best_fitness REAL,
    worst_fitness REAL,
    best_cell_id INTEGER,               -- Best performer of this pattern

    -- Market applicability
    works_on_symbols TEXT,              -- JSON: ["PURR", "HFUN", "BTC"]
    fails_on_symbols TEXT,              -- JSON: ["ETH"] - regime-specific

    FOREIGN KEY (discovered_by_cell_id) REFERENCES cells(cell_id),
    FOREIGN KEY (best_cell_id) REFERENCES cells(cell_id)
);
```

**Indexes**:
```sql
CREATE INDEX idx_patterns_name ON discovered_patterns(pattern_name);
CREATE INDEX idx_patterns_category ON discovered_patterns(category);
CREATE INDEX idx_patterns_fitness ON discovered_patterns(best_fitness DESC);
```

### Table: `cell_patterns`

**Many-to-many** - Links cells to patterns

```sql
CREATE TABLE cell_patterns (
    cell_id INTEGER NOT NULL,
    pattern_id INTEGER NOT NULL,

    -- How strongly does this cell exhibit this pattern?
    confidence REAL DEFAULT 1.0,  -- 0.0 - 1.0

    -- When was this association made?
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assigned_by TEXT,  -- 'llm' or 'manual'

    PRIMARY KEY (cell_id, pattern_id),
    FOREIGN KEY (cell_id) REFERENCES cells(cell_id) ON DELETE CASCADE,
    FOREIGN KEY (pattern_id) REFERENCES discovered_patterns(pattern_id) ON DELETE CASCADE,

    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CHECK (assigned_by IN ('llm', 'manual'))
);
```

**Use case**: A cell can exhibit multiple patterns (e.g., both "Volume Analysis" and "Mean Reversion").

### Table: `cell_metadata`

**Extensible key-value** - Arbitrary cell properties

```sql
CREATE TABLE cell_metadata (
    metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_id INTEGER NOT NULL,

    key TEXT NOT NULL,
    value TEXT,  -- JSON-encoded for complex types
    value_type TEXT DEFAULT 'string',  -- 'string', 'number', 'boolean', 'json'

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (cell_id) REFERENCES cells(cell_id) ON DELETE CASCADE,

    UNIQUE (cell_id, key)  -- One value per key per cell
);
```

**Indexes**:
```sql
CREATE INDEX idx_metadata_cell ON cell_metadata(cell_id);
CREATE INDEX idx_metadata_key ON cell_metadata(key);
```

**Example usage**:
```python
# Store arbitrary properties without schema changes
repository.set_metadata(cell_id=47, key='tested_on_btc', value='true', value_type='boolean')
repository.set_metadata(cell_id=47, key='robustness_score', value='0.87', value_type='number')
repository.set_metadata(cell_id=47, key='discovered_anomaly', value='{"date": "2025-09-15", "type": "flash_crash"}', value_type='json')
```

## Entity-Relationship Diagram

```
┌──────────────┐
│evolution_runs│
└──────┬───────┘
       │
       │ best_cell_id
       ▼
┌──────────────┐       ┌──────────────────┐
│    cells     │◄──────┤cell_phenotypes   │ (many phenotypes per cell)
└──────┬───────┘       └──────────────────┘
       │
       ├──► parent_cell_id (self-reference for lineage)
       │
       ├──► superseded_by_cell_id (for deprecation)
       │
       ├──────────────────┐
       │                  │
       ▼                  ▼
┌──────────────┐   ┌──────────────┐
│cell_patterns │   │cell_metadata │
└──────┬───────┘   └──────────────┘
       │
       │ pattern_id
       ▼
┌─────────────────────┐
│discovered_patterns  │
└─────────────────────┘
       ▲
       │ discovered_by_cell_id, best_cell_id
       │
       └──────────────┘

┌──────────────────┐
│failed_mutations  │ (references cells via parent_cell_id)
└──────────────────┘
```

## Common Queries

### 1. Get Top 10 Online Cells

```sql
SELECT cell_id, dsl_genome, fitness, llm_name, generation
FROM cells
WHERE status = 'online'
ORDER BY fitness DESC
LIMIT 10;
```

### 2. Get Full Lineage of a Cell

```sql
WITH RECURSIVE lineage AS (
    -- Start with the target cell
    SELECT cell_id, parent_cell_id, dsl_genome, fitness, generation, 0 AS depth
    FROM cells WHERE cell_id = ?

    UNION ALL

    -- Recursively get parents
    SELECT c.cell_id, c.parent_cell_id, c.dsl_genome, c.fitness, c.generation, l.depth + 1
    FROM cells c
    JOIN lineage l ON c.cell_id = l.parent_cell_id
)
SELECT * FROM lineage ORDER BY depth DESC;
```

### 3. Find Unanalyzed Cells (LLM Work Queue)

```sql
SELECT cell_id, dsl_genome, fitness, generation
FROM cells
WHERE status = 'online'
  AND llm_analyzed_at IS NULL
  AND fitness > 0
ORDER BY fitness DESC
LIMIT 50;
```

### 4. Get Evolution Run Summary

```sql
SELECT
    er.*,
    c.dsl_genome AS best_strategy,
    c.fitness AS best_fitness,
    c.llm_name AS best_strategy_name
FROM evolution_runs er
LEFT JOIN cells c ON er.best_cell_id = c.cell_id
WHERE run_id = ?;
```

### 5. Find Cells by Pattern

```sql
SELECT c.cell_id, c.dsl_genome, c.fitness, cp.confidence
FROM cells c
JOIN cell_patterns cp ON c.cell_id = cp.cell_id
JOIN discovered_patterns dp ON cp.pattern_id = dp.pattern_id
WHERE dp.pattern_name = 'Volume Spike Reversal'
  AND c.status = 'online'
ORDER BY c.fitness DESC;
```

### 6. Get Pattern Performance Stats

```sql
SELECT
    dp.pattern_name,
    dp.category,
    COUNT(cp.cell_id) AS total_cells,
    AVG(c.fitness) AS avg_fitness,
    MAX(c.fitness) AS best_fitness,
    MIN(c.fitness) AS worst_fitness
FROM discovered_patterns dp
LEFT JOIN cell_patterns cp ON dp.pattern_id = cp.pattern_id
LEFT JOIN cells c ON cp.cell_id = c.cell_id
WHERE c.status = 'online'
GROUP BY dp.pattern_id
ORDER BY avg_fitness DESC;
```

### 7. Find Similar Cells by Shared Patterns

```sql
SELECT c1.cell_id, c1.dsl_genome, c1.fitness,
       COUNT(*) AS shared_patterns
FROM cells c1
JOIN cell_patterns cp1 ON c1.cell_id = cp1.cell_id
JOIN cell_patterns cp2 ON cp1.pattern_id = cp2.pattern_id
WHERE cp2.cell_id = ?  -- Reference cell
  AND c1.cell_id != ?
  AND c1.status = 'online'
GROUP BY c1.cell_id
HAVING shared_patterns >= 2  -- At least 2 shared patterns
ORDER BY shared_patterns DESC, c1.fitness DESC
LIMIT 10;
```

### 8. Evolution Progress Over Time

```sql
SELECT
    generation,
    COUNT(*) AS cells_birthed,
    AVG(fitness) AS avg_fitness,
    MAX(fitness) AS best_fitness
FROM cells
GROUP BY generation
ORDER BY generation;
```

### 9. Mutation Failure Analysis

```sql
SELECT
    mutation_type,
    COUNT(*) AS total_attempts,
    AVG(fitness) AS avg_fitness,
    COUNT(CASE WHEN failure_reason LIKE '%Parse error%' THEN 1 END) AS parse_errors,
    COUNT(CASE WHEN failure_reason LIKE '%worse than parent%' THEN 1 END) AS performance_failures
FROM failed_mutations
GROUP BY mutation_type;
```

### 10. Cell Performance Across Symbols

```sql
SELECT
    cp.symbol,
    c.cell_id,
    c.dsl_genome,
    cp.total_trades,
    cp.win_rate,
    cp.sharpe_ratio
FROM cells c
JOIN cell_phenotypes cp ON c.cell_id = cp.cell_id
WHERE c.cell_id = ?
ORDER BY cp.symbol;
```

## Database Initialization

```python
import sqlite3
from pathlib import Path

def init_database(db_path: Path):
    """Initialize the cell database with schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute all CREATE TABLE statements
    # (schema from above)

    # Create indexes
    # (indexes from above)

    conn.commit()
    conn.close()

    print(f"✓ Database initialized: {db_path}")
```

## Backup and Migration

### Backup

```bash
# Simple file copy (database is single file)
cp results/interactive_output/evolution/cells.db results/backups/cells_2025-10-09.db

# Or use SQLite backup API
sqlite3 results/interactive_output/evolution/cells.db ".backup results/backups/cells_2025-10-09.db"
```

### Export to JSON

```bash
# Export all cells
sqlite3 -json results/interactive_output/evolution/cells.db "SELECT * FROM cells" > cells_export.json

# Export specific run
sqlite3 -json results/interactive_output/evolution/cells.db "SELECT * FROM cells WHERE generation <= 100" > run1_cells.json
```

### Migration Between Versions

```python
def migrate_v1_to_v2(db_path: Path):
    """Add new columns without breaking existing data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Example: Add llm_confidence column
    cursor.execute("ALTER TABLE cells ADD COLUMN llm_confidence REAL")

    conn.commit()
    conn.close()
```

## Performance Considerations

- **Indexes**: All foreign keys and commonly queried fields are indexed
- **Cascade deletes**: Phenotypes and patterns are auto-deleted when cells are removed
- **VACUUM**: Run periodically to reclaim space: `sqlite3 cells.db "VACUUM;"`
- **Concurrent access**: SQLite supports multiple readers, one writer
- **Size estimate**: ~10KB per cell with phenotype → 10,000 cells ≈ 100MB database

## Summary

The database schema provides:
- **Complete cell lifecycle** tracking
- **Rich phenotype** data for analysis
- **Pattern taxonomy** built by LLM
- **Efficient queries** for common operations
- **Extensibility** via metadata table
- **Portability** via single-file SQLite

Next: See `CELL_STORAGE_API.md` for Python interface to this schema.
