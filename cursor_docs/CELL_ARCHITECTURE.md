# Cell Architecture: Living Strategies

## Overview

The system treats successful trading strategies as **living cells** with identity, lineage, and semantic meaning. This biological metaphor enables:
- **Persistent identity**: Each successful strategy is a unique, trackable entity
- **Lineage tracking**: Full ancestry from parent to child mutations
- **Semantic layer**: LLM-assigned names, categories, and hypotheses
- **Evolution history**: Complete record of what worked and why

## Cell vs Failed Mutation

### What Gets Saved as a Cell?

**Birthed Cells** (persistent entities):
```
Conditions for birth:
1. Fitness > 0 (strategy survives)
2. OR: Fitness > parent fitness (improvement even if negative)
3. OR: First strategy of a generation (Gen 0)
```

These become "online cells" - living members of the population with:
- Unique `cell_id`
- Full genome (DSL string)
- Performance data (phenotype)
- Lineage information
- Potential LLM analysis

**Failed Mutations** (statistics only):
```
Recorded but not birthed:
- Fitness <= 0 AND fitness <= parent
- Parse errors
- Backtest crashes
```

These are logged for statistics (failure rate, common error patterns) but don't become cells. They're evolutionary dead-ends.

## Cell Identity: The Five Components

### 1. Genome (DNA)

The DSL strategy string itself:

```python
genome = "IF EPSILON(0) / EPSILON(20) > 1.5 AND DELTA(0) < DELTA(10) THEN BUY ELSE HOLD"
```

This is the "genetic code" - the actual executable trading logic.

**Immutable**: Once birthed, a cell's genome never changes. Mutations create new children.

### 2. Phenotype (Behavior)

How the genome expresses itself in the market:

```python
phenotype = {
    'symbol': 'PURR',
    'timeframe': '1h',
    'total_trades': 45,
    'profitable_trades': 28,
    'win_rate': 0.622,
    'total_profit': 31.38,
    'total_fees': 8.05,
    'max_drawdown': -5.23,
    'sharpe_ratio': 1.47,
    'trigger_conditions': {...}  # Market states when strategy activated
}
```

**Mutable**: Re-testing the same genome on new data creates a new phenotype record.

### 3. Fitness (Survival Value)

Economic performance metric:

```python
fitness = trading_profit - transaction_fees - llm_costs
fitness = 31.38 - 8.05 - 0.0165 = 23.31
```

**Primary selection pressure**: Only cells with `fitness > 0` survive long-term.

### 4. Lineage (Ancestry)

Parent-child relationships:

```python
Cell #47:
  parent_cell_id: 15
  generation: 89

  Ancestry chain:
    Cell #1 (Gen 0) → Cell #5 (Gen 12) → Cell #15 (Gen 34) → Cell #47 (Gen 89)
```

Enables:
- Tracking mutation paths
- Identifying successful lineages
- Understanding which mutations were productive

### 5. Semantics (LLM Interpretation)

Human-readable meaning assigned by LLM:

```python
semantics = {
    'llm_name': 'Volume Spike Reversal Detector',
    'llm_category': 'Volume Analysis',
    'llm_hypothesis': 'Detects institutional accumulation during local price dips. The 2x volume spike above 20-period average combined with price below 10-period value suggests large buyer entering. Similar to traditional "volume climax" but mathematically precise.',
    'llm_analyzed_at': '2025-10-09 12:34:56'
}
```

**Key insight**: The genome stays abstract (`EPSILON`, `DELTA`), but the LLM names what it does.

## Cell Lifecycle

```
┌──────────────┐
│   MUTATION   │ (DSL mutator creates variant)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   BACKTEST   │ (Test on market data)
└──────┬───────┘
       │
       ├──> Fitness <= 0 or <= parent? ──> FAILED (logged, not birthed)
       │
       └──> Fitness > 0 or > parent? ──> BIRTHED
                                          └──────┬───────┐
                                                 │       │
                                                 ▼       ▼
                                           ┌─────────┐ ┌────────────┐
                                           │ ONLINE  │ │ DEPRECATED │
                                           └────┬────┘ └────────────┘
                                                │
                                                ├──> Better version found? ──> DEPRECATED
                                                ├──> Interesting but not top? ──> ARCHIVED
                                                └──> Pattern no longer works? ──> EXTINCT
```

### Status Definitions

**ONLINE** (Active population):
- Current best performers
- Competing for selection
- Candidates for LLM analysis
- Eligible for mutation

**DEPRECATED**:
- Replaced by a better version of the same pattern
- Example: Cell #12 was `IF EPSILON(20) > 1.5 THEN BUY`, Cell #47 improved it to `IF EPSILON(0)/EPSILON(20) > 1.5 AND DELTA(0) < DELTA(10) THEN BUY`
- Kept for lineage tracking but not actively evolved

**ARCHIVED**:
- Interesting strategy but outperformed
- Might be useful for different market regime
- Kept for analysis and potential resurrection

**EXTINCT**:
- Pattern that stopped working
- Detected via walk-forward validation failure
- Kept for historical analysis ("what stopped working and when?")

## Cell Queries

### Get Top 10 Living Cells

```sql
SELECT cell_id, dsl_genome, fitness, llm_name, generation
FROM cells
WHERE status = 'online'
ORDER BY fitness DESC
LIMIT 10;
```

### Get Cell with Full Context

```python
cell = repository.get_cell(cell_id=47)

print(f"Cell #{cell.id}: {cell.llm_name or 'Unnamed'}")
print(f"Genome: {cell.genome}")
print(f"Fitness: ${cell.fitness:.2f}")
print(f"Generation: {cell.generation}")
print(f"Parent: Cell #{cell.parent_id}")

# Get phenotype
phenotype = repository.get_phenotype(cell_id=47)
print(f"Win Rate: {phenotype.win_rate:.1%}")
print(f"Trades: {phenotype.total_trades}")
```

### Trace Lineage (Ancestry)

```python
# Get all ancestors
lineage = repository.get_lineage(cell_id=47)

print("Ancestry Chain:")
for i, ancestor in enumerate(lineage):
    indent = "  " * i
    print(f"{indent}└─ Cell #{ancestor.id} (Gen {ancestor.generation}): ${ancestor.fitness:.2f}")
```

Output:
```
Ancestry Chain:
└─ Cell #1 (Gen 0): $6.17
  └─ Cell #5 (Gen 12): $15.32
    └─ Cell #15 (Gen 34): $18.45
      └─ Cell #47 (Gen 89): $23.31
```

### Find Unanalyzed Cells (LLM Work Queue)

```python
# Get cells that survived but LLM hasn't named yet
unanalyzed = repository.find_unanalyzed_cells(limit=50)

for cell in unanalyzed:
    print(f"Cell #{cell.id}: {cell.genome} (fitness: ${cell.fitness:.2f})")
```

This is the queue for `trading-learn` mode to process.

## Cell Comparison

### By Genome Similarity

```python
# Find cells with similar DSL structure
similar = repository.find_similar_genomes(
    reference_cell_id=47,
    max_edit_distance=10  # Levenshtein distance
)
```

Use case: Identify variations of the same strategy pattern.

### By Phenotype Clustering

```python
# Find cells with similar market behavior
similar = repository.find_similar_phenotypes(
    reference_cell_id=47,
    metrics=['win_rate', 'sharpe_ratio', 'avg_trade_duration']
)
```

Use case: Discover convergent evolution (different genomes, similar behavior).

### By Pattern Tags

```python
# Find all cells tagged with "Volume Analysis"
cells = repository.find_by_pattern(pattern_name="Volume Analysis")
```

Use case: LLM can analyze all volume-based strategies together.

## Cell Metadata

Each cell can store arbitrary key-value metadata:

```python
repository.add_cell_metadata(
    cell_id=47,
    key='discovered_on_symbol',
    value='PURR'
)

repository.add_cell_metadata(
    cell_id=47,
    key='robustness_score',
    value=0.87  # Tested on multiple symbols
)
```

This allows extending cell attributes without schema changes.

## Cell Birth Example

```python
# Evolution finds a mutation that improves fitness
parent_cell = repository.get_cell(15)  # fitness: $18.45
mutated_genome = "IF EPSILON(0)/EPSILON(20) > 1.5 AND DELTA(0) < DELTA(10) THEN BUY ELSE HOLD"

# Backtest the mutation
fitness = backtest(mutated_genome)  # Returns: $23.31

# Fitness improved! Birth a new cell.
if fitness > parent_cell.fitness:
    child_id = repository.birth_cell(
        genome=mutated_genome,
        fitness=fitness,
        generation=parent_cell.generation + 1,
        parent_id=parent_cell.id
    )

    # Record phenotype
    repository.record_phenotype(
        cell_id=child_id,
        symbol='PURR',
        timeframe='1h',
        total_trades=45,
        profitable_trades=28,
        total_profit=31.38,
        # ... etc
    )

    print(f"✓ Cell #{child_id} born! Fitness: ${fitness:.2f}")
```

## Cell Deletion/Deprecation

Cells are never truly deleted (for lineage integrity), but their status changes:

```python
# Mark a cell as deprecated when a better version exists
repository.deprecate_cell(
    cell_id=15,
    reason="Improved by Cell #47",
    superseded_by_cell_id=47
)

# Cell #15 now has status='deprecated' but remains in database
```

## Summary

| Component | Description | Mutable? |
|-----------|-------------|----------|
| **Cell ID** | Unique identifier | No (auto-increment) |
| **Genome** | DSL strategy string | No (immutable) |
| **Phenotype** | Market behavior | Yes (retested on new data) |
| **Fitness** | Economic performance | Yes (context-dependent) |
| **Lineage** | Parent/generation | No (historical fact) |
| **Semantics** | LLM interpretation | Yes (analysis improves) |
| **Status** | online/deprecated/archived/extinct | Yes (lifecycle) |

**Key Design Principle**: Cells are persistent, identifiable entities that accumulate knowledge over time. The LLM doesn't analyze "a population" - it analyzes specific cells by ID, building a taxonomy of discovered patterns that future analysis can reference.
