# Sprint 2 Started: Evolution Integration

**Date**: 2025-10-09
**Status**: Sprint 2.1 in progress

## Sprint 1 Recap ✅

Sprint 1 is **complete** with all infrastructure in place:
- ✅ Storage module with 8 database tables
- ✅ CellRepository with 19 CRUD methods
- ✅ Multi-timeframe data fetching (1H/4H/1D)
- ✅ Multi-timeframe backtesting with phenotype metrics
- ✅ 26/26 unit tests passing

## Sprint 2 Tasks

### 2.1 🔄 Integrate Cell Birth into Evolution Loop (IN PROGRESS)

**File**: `base_agent/agent.py` - `run_trading_evolve()` function (lines 811-1042)

**Current behavior**:
- Uses in-memory `population_history` list
- Tracks best strategy in local variables
- No persistent storage

**Changes needed**:
1. Initialize `CellRepository` at start of function
2. Start evolution run: `repo.start_evolution_run()`
3. Birth Gen 0 cell after initial test
4. For each generation mutation:
   - If fitness > parent: birth new cell
   - If fitness <= parent: record failed mutation
5. Complete evolution run: `repo.complete_evolution_run()`

**Code changes**:
```python
# At start of run_trading_evolve():
from .src.storage.cell_repository import CellRepository
from .src.data.hyperliquid_fetcher import HyperliquidDataFetcher

# Initialize database
db_path = results_dir / "cells.db"
repo = CellRepository(db_path)

# Start evolution run
run_id = repo.start_evolution_run(
    run_type='evolution',
    max_generations=generations,
    fitness_goal=fitness_goal,
    symbol='PURR',
    timeframe='1h',
    initial_capital=100.0,
    transaction_fee_rate=0.00045,
)

# Fetch multi-timeframe data
fetcher = HyperliquidDataFetcher()
multi_tf_data = fetcher.fetch_multi_timeframe('PURR', lookback_days=30)

# After Gen 0 test:
if current_fitness > 0:
    cell_id = repo.birth_cell(
        generation=0,
        parent_cell_id=None,
        dsl_genome=current_strategy,
        fitness=current_fitness,
        status='online',
    )
    # Store phenotypes for each timeframe
    for tf, df in multi_tf_data.items():
        # Run backtest and create phenotype
        # Store with repo.store_phenotype()

# In evolution loop:
if mutated_fitness > current_fitness:
    # Birth child cell
    child_id = repo.birth_cell(
        generation=gen,
        parent_cell_id=current_cell_id,
        dsl_genome=mutated_strategy,
        fitness=mutated_fitness,
    )
else:
    # Record failed mutation
    repo.record_failed_mutation(
        parent_cell_id=current_cell_id,
        attempted_genome=mutated_strategy,
        generation=gen,
        failure_reason='lower_than_parent',
        fitness_achieved=mutated_fitness,
    )

# At end:
repo.complete_evolution_run(
    run_id=run_id,
    best_cell_id=best_cell_id,
    total_cells_birthed=cells_birthed_count,
    total_mutations_failed=mutations_failed_count,
    final_best_fitness=best_fitness,
    termination_reason=termination_reason,
    generations_without_improvement=generations_without_improvement,
)
```

### 2.2 ⏳ Update Mutation Selection (PENDING)

**Current**: Parent is tracked in-memory (`current_strategy` variable)

**Change**: Query database for best parent
```python
# Instead of tracking current_strategy in memory:
best_cells = repo.get_top_cells(limit=1, status='online')
if best_cells:
    parent_cell = best_cells[0]
    program = interpreter.parse(parent_cell.dsl_genome)
```

### 2.3 ⏳ Implement DSL V2 Phase 1 - Arithmetic (PENDING)

**Files to modify**:
- `base_agent/src/dsl/grammar.py` - Add `BinaryOp` class
- `base_agent/src/dsl/interpreter.py` - Add `evaluate_expression()`
- `base_agent/src/dsl/mutator.py` - Add arithmetic mutations

**New syntax support**:
```
IF (EPSILON(0) / EPSILON(20)) > 1.5 THEN BUY ELSE SELL
IF (DELTA(0) - DELTA(10)) > 0 THEN BUY ELSE HOLD
```

### 2.4 ⏳ Test Evolution with Cell Storage (PENDING)

Run evolution and verify:
- Cells stored correctly
- Lineage tracked
- Failed mutations recorded
- Phenotypes stored per timeframe
- Database queries work

**Test command**:
```bash
docker run --rm \
  -v $(pwd)/base_agent:/home/agent/agent_code \
  -v $(pwd)/benchmark_data:/home/agent/benchmark_data \
  -v $(pwd)/results/interactive_output:/home/agent/workdir \
  sica_sandbox \
  python -m agent_code.agent trading-evolve -g 20 -f 100.0
```

## Key Design Decisions

1. **Cell birth criteria**: `fitness > 0 OR fitness > parent_fitness`
2. **Failed mutations**: Record statistics only, don't create cells
3. **Multi-timeframe**: Test on 1H/4H/1D, birth cell with best timeframe fitness
4. **Database location**: `workdir/evolution/cells.db`
5. **Phenotypes**: Store detailed metrics for each timeframe tested

## Expected Outcome

After Sprint 2.1-2.2:
- Evolution loop stores cells in database ✓
- Can query lineage and history ✓
- Multi-timeframe testing works ✓
- Ready for DSL V2 arithmetic (2.3)

After Sprint 2.3:
- Arithmetic DSL breaks $6.17 stagnation barrier ✓
- More expressive strategies possible ✓

## Next Session Tasks

1. Complete 2.1: Update `run_trading_evolve()` function
2. Test changes in Docker
3. Move to 2.2: Update mutation selection
4. Begin 2.3: DSL V2 arithmetic

---

**Status**: Ready to implement Sprint 2.1 integration
