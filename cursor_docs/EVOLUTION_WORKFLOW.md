# Evolution Workflow: Complete Cycle

## Overview

The system uses a **hybrid evolutionary approach** that combines:
- **Blind mutation** (cheap, massive parallelism)
- **LLM analysis** (expensive, selective insight)
- **Cell-based storage** (persistent knowledge accumulation)

This document describes the complete workflow from initial mutation through LLM analysis and back to guided evolution.

## Two Modes

### Evolution Mode (Offline)

**Purpose**: Pure natural selection without LLM costs

**Process**:
```
Gen 0: Random strategy → Backtest → Birth Cell #1
Gen 1-N: Mutate best → Backtest → Birth if better, else record failure
```

**Cost**: $0 after Gen 0 (no LLM calls)

**Termination**:
- Fitness goal reached
- Max generations completed
- Stagnation (20 generations without improvement)

**Output**: Database of cells with lineage

### Trading-Learn Mode (LLM-Powered)

**Purpose**: Combine evolution with LLM pattern discovery

**Process**:
```
Iteration 1:
  1. LLM generates strategy (or mutates previous best)
  2. Backtest
  3. Birth cell if fitness > 0
  4. LLM immediately analyzes successful cell
  5. Updates pattern taxonomy

Iteration 2-N:
  1. LLM sees history of previous attempts
  2. Uses pattern knowledge to propose better strategy
  3. Repeat
```

**Cost**: ~$0.02 per iteration (LLM generation + analysis)

**Output**: Database of cells + semantic analysis + pattern taxonomy

## Complete Workflow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                     GENERATION N                                │
└────────────────────────────────────────────────────────────────┘

1. SELECT PARENT
   ├─ Query: Get best online cell
   └─ repo.get_top_cells(limit=1, status='online')

2. GENERATE MUTATION
   ├─ Mode A: Random (80% probability)
   │   ├─ Parse parent genome
   │   ├─ mutator.mutate(program)
   │   └─ Generate random variation
   │
   └─ Mode B: LLM-Guided (20% probability, if trading-learn)
       ├─ Query: Get LLM's suggested mutations for this cell
       ├─ repo.get_llm_proposed_mutations(parent_id)
       └─ Select one suggestion

3. BACKTEST
   ├─ Load market data (1H, 4H, 1D × 30 days)
   ├─ Execute strategy on all timeframes
   ├─ Calculate fitness for each
   └─ Get best fitness across timeframes

4. DECISION POINT
   │
   ├─ Fitness > 0 OR Fitness > parent? ──┐
   │                                      │
   │                              ┌───────▼────────┐
   │                              │  BIRTH CELL    │
   │                              │                │
   │                              │ 1. repo.birth_cell()
   │                              │ 2. repo.record_phenotype() (per timeframe)
   │                              │ 3. Update evolution_runs stats
   │                              └────────┬───────┘
   │                                       │
   │                              ┌────────▼────────────┐
   │                              │ LLM ANALYSIS?       │
   │                              │ (if trading-learn)  │
   │                              └────────┬────────────┘
   │                                       │
   │                              ┌────────▼───────────────────┐
   │                              │ 1. Prepare context         │
   │                              │ 2. Call LLM                │
   │                              │ 3. Parse analysis JSON     │
   │                              │ 4. repo.update_llm_analysis()│
   │                              │ 5. Create/link pattern     │
   │                              │ 6. Store mutation proposals│
   │                              └────────────────────────────┘
   │
   └─ Fitness <= 0 AND <= parent? ──┐
                                     │
                            ┌────────▼──────────┐
                            │  RECORD FAILURE   │
                            │                   │
                            │ repo.record_failure()│
                            │ (statistics only) │
                            └───────────────────┘

5. CHECK TERMINATION
   ├─ Fitness >= goal? → STOP (success)
   ├─ Generation >= max? → STOP (completed)
   ├─ No improvement for 20 gen? → STOP (stagnation)
   └─ Otherwise → Continue to Gen N+1
```

## Detailed Phase Breakdown

### Phase 1: Parent Selection

```python
def select_parent(repo: CellRepository, mode: str = 'best') -> Cell:
    """Select parent cell for next mutation."""

    if mode == 'best':
        # Always use current best
        parents = repo.get_top_cells(limit=1, status='online')
        return parents[0]

    elif mode == 'fitness_proportional':
        # Probabilistic selection based on fitness
        top_cells = repo.get_top_cells(limit=10, status='online')
        fitnesses = [c.fitness for c in top_cells]
        total = sum(fitnesses)
        probabilities = [f / total for f in fitnesses]
        return random.choices(top_cells, weights=probabilities)[0]

    elif mode == 'tournament':
        # Select 5 random, pick best
        candidates = repo.get_random_cells(limit=5, status='online')
        return max(candidates, key=lambda c: c.fitness)
```

**Default**: Use 'best' mode for deterministic evolution

### Phase 2: Mutation Generation

#### Random Mutation (Evolution Mode)

```python
def generate_random_mutation(parent: Cell) -> str:
    """Generate random mutation of parent genome."""

    program = interpreter.parse(parent.genome)
    mutated_program = mutator.mutate(program)
    mutated_genome = mutator.to_string(mutated_program)

    return mutated_genome
```

#### LLM-Guided Mutation (Trading-Learn Mode)

```python
def generate_guided_mutation(parent: Cell, repo: CellRepository) -> tuple[str, int] | None:
    """Get LLM-suggested mutation if available."""

    # Check if this cell has LLM-proposed mutations (as dicts with id and genome)
    suggestions = repo.get_llm_proposed_mutations(parent.cell_id, status='pending')

    if not suggestions:
        return None

    # Pick random suggestion
    suggestion = random.choice(suggestions)
    return suggestion['proposed_genome'], suggestion['proposal_id']
```

#### Hybrid Strategy

```python
def generate_mutation(parent: Cell, repo: CellRepository, mode: str) -> tuple[str, int | None]:
    """Generate mutation using hybrid strategy."""

    if mode == 'trading-learn' and random.random() < 0.2:
        # 20% chance: Try LLM-guided mutation
        guided_result = generate_guided_mutation(parent, repo)
        if guided_result:
            genome, proposal_id = guided_result
            print(f"Using LLM-guided mutation (Proposal #{proposal_id})")
            return genome, proposal_id

    # Fall back to random mutation
    print(f"Using random mutation")
    return generate_random_mutation(parent), None
```

### Phase 3: Multi-Timeframe Backtesting

```python
def backtest_all_timeframes(
    genome: str,
    symbol: str = 'PURR'
) -> dict[str, BacktestResult]:
    """Test strategy on all timeframes."""

    results = {}

    for timeframe in ['1H', '4H', '1D']:
        # Load market data for this timeframe
        market_data = load_market_data(symbol, timeframe, days=30)

        # Run backtest
        result = run_backtest(genome, market_data, timeframe)

        results[timeframe] = result

    return results
```

**Result aggregation**:
```python
# Option A: Best timeframe
best_tf = max(results.items(), key=lambda x: x[1].fitness)
fitness = best_tf[1].fitness
primary_timeframe = best_tf[0]

# Option B: Average across timeframes
avg_fitness = sum(r.fitness for r in results.values()) / len(results)
```

**Recommendation**: Use Option A (best timeframe) for cell birth decision, but record all phenotypes.

### Phase 4: Cell Birth

```python
def birth_new_cell(
    genome: str,
    parent: Cell,
    backtest_results: dict[str, BacktestResult],
    repo: CellRepository
) -> int | None:
    """Birth a new cell if it qualifies."""

    # Get best fitness across timeframes
    best_result = max(backtest_results.values(), key=lambda r: r.fitness)

    # Check if cell qualifies for birth
    if best_result.fitness <= 0 and best_result.fitness <= parent.fitness:
        # Record failure
        repo.record_failure(
            parent_id=parent.cell_id,
            attempted_dsl=genome,
            fitness=best_result.fitness,
            failure_reason=f"Fitness {best_result.fitness:.2f} <= parent {parent.fitness:.2f}",
            mutation_type='random'
        )
        return None

    # Birth the cell
    with repo:
        cell_id = repo.birth_cell(
            genome=genome,
            fitness=best_result.fitness,
            generation=parent.generation + 1,
            parent_id=parent.cell_id
        )

        # Record phenotype for each timeframe
        for timeframe, result in backtest_results.items():
            repo.record_phenotype(
                cell_id=cell_id,
                symbol='PURR',
                timeframe=timeframe,
                total_trades=result.total_trades,
                profitable_trades=result.profitable_trades,
                losing_trades=result.losing_trades,
                total_profit=result.total_profit,
                total_fees=result.total_fees,
                sharpe_ratio=result.sharpe_ratio,
                win_rate=result.win_rate,
                trigger_conditions=json.dumps(result.triggers)
            )

    print(f"✓ Cell #{cell_id} born! Fitness: ${best_result.fitness:.2f} ({timeframe})")
    return cell_id
```

### Phase 5: LLM Analysis (Trading-Learn Only)

```python
async def analyze_cell_if_needed(
    cell_id: int,
    repo: CellRepository,
    mode: str,
    budget_remaining: float
) -> dict | None:
    """Analyze cell with LLM if in trading-learn mode."""

    if mode != 'trading-learn':
        return None

    # Check budget
    if budget_remaining < 0.01:  # ~$0.01 per analysis
        print(f"⚠ Skipping analysis: budget low")
        return None

    # Prepare context
    cell = repo.get_cell(cell_id)
    phenotypes = repo.get_phenotypes(cell_id)
    lineage = repo.get_lineage(cell_id)

    context = prepare_cell_context(cell, phenotypes, lineage)

    # Call LLM
    analysis = await analyze_cell_with_llm(context)

    # Store analysis
    repo.update_llm_analysis(
        cell_id=cell_id,
        name=analysis['name'],
        category=analysis['category'],
        hypothesis=analysis['hypothesis'],
        confidence=analysis['confidence']
    )

    # Create or link pattern
    pattern = repo.find_pattern_by_name(analysis['name'])
    if not pattern:
        pattern_id = repo.create_pattern(
            name=analysis['name'],
            category=analysis['category'],
            description=analysis['explanation'],
            discovered_by_cell_id=cell_id
        )
    else:
        pattern_id = pattern.pattern_id

    repo.link_cell_to_pattern(cell_id, pattern_id, analysis['confidence'])

    # Store mutation proposals for future use
    repo.store_mutation_proposals(cell_id, analysis['mutations'])

    print(f"✓ Cell #{cell_id} analyzed: {analysis['name']}")

    return analysis
```

### Phase 5.5: Closing the Loop on LLM Guidance (Meta-Learning)

A critical, but subtle, part of the workflow is tracking the performance of the LLM's own suggestions. This creates a meta-learning loop where the system can learn which kinds of guidance are actually effective.

This is achieved by linking the newly birthed cell back to the specific mutation proposal that generated it. The `cell_mutation_proposals` table has a `result_cell_id` column for this purpose. When a cell is born from a guided mutation, we update this record.

```python
def link_proposal_to_result(
    repo: CellRepository,
    proposal_id: int,
    child_cell_id: int
):
    """Update a mutation proposal with the ID of the cell it created."""
    repo.update_mutation_proposal_result(proposal_id, child_cell_id)
    print(f"✓ Linked LLM proposal #{proposal_id} to successful child Cell #{child_cell_id}")
```

This allows for powerful future analysis:
- "Which types of LLM suggestions lead to the biggest fitness gains?"
- "Is the LLM getting better at proposing mutations over time?"
- "Can we fine-tune the analysis prompt based on which rationales lead to successful mutations?"

This step transforms the system from one that simply *uses* AI to one that actively *evaluates and improves* its AI guidance.

### Phase 6: Termination Checks

```python
def check_termination(
    current_fitness: float,
    best_fitness: float,
    generation: int,
    max_generations: int,
    fitness_goal: float,
    generations_without_improvement: int
) -> tuple[bool, str]:
    """Check if evolution should terminate."""

    # Goal reached
    if current_fitness >= fitness_goal:
        return True, f"goal_reached (${current_fitness:.2f} >= ${fitness_goal:.2f})"

    # Max generations
    if generation >= max_generations:
        return True, f"max_generations ({generation})"

    # Stagnation
    if generations_without_improvement >= 20:
        return True, f"stagnation (no improvement for 20 generations)"

    return False, ""
```

## Complete Evolution Loop (Pseudocode)

```python
async def run_evolution(
    mode: str = 'evolution',  # 'evolution' or 'trading-learn'
    max_generations: int = 50,
    fitness_goal: float = 100.0,
    symbol: str = 'PURR'
):
    """Run complete evolution cycle."""

    # Initialize
    repo = CellRepository(db_path)
    run_id = repo.create_evolution_run(
        run_type=mode,
        max_generations=max_generations,
        fitness_goal=fitness_goal,
        symbol=symbol
    )

    # Generation 0: Create initial strategy
    if mode == 'evolution':
        initial_genome = generate_random_strategy()
    else:  # trading-learn
        initial_genome = await llm_generate_initial_strategy()

    # Test and birth Gen 0
    backtest_results = backtest_all_timeframes(initial_genome, symbol)
    best_result = max(backtest_results.values(), key=lambda r: r.fitness)

    with repo:
        cell_id = repo.birth_cell(
            genome=initial_genome,
            fitness=best_result.fitness,
            generation=0,
            parent_id=None
        )
        # Record phenotypes...

    best_fitness = best_result.fitness
    best_cell_id = cell_id
    generations_without_improvement = 0

    print(f"Gen 0: Cell #{cell_id} - ${best_result.fitness:.2f}")

    # Evolution loop
    for generation in range(1, max_generations + 1):
        print(f"\n{'='*70}")
        print(f"GENERATION {generation}")
        print(f"{'='*70}")

        # 1. Select parent
        parent = repo.get_cell(best_cell_id)

        # 2. Generate mutation
        mutated_genome, proposal_id = generate_mutation(parent, repo, mode)

        # 3. Backtest
        backtest_results = backtest_all_timeframes(mutated_genome, symbol)
        best_result = max(backtest_results.values(), key=lambda r: r.fitness)

        # 4. Birth or fail
        new_cell_id = birth_new_cell(mutated_genome, parent, backtest_results, repo)

        if new_cell_id:
            # 5. LLM analysis (if trading-learn)
            if mode == 'trading-learn':
                await analyze_cell_if_needed(new_cell_id, repo, mode, budget_remaining=10.0)

            # 5.5 Close the loop if this was a guided mutation
            if proposal_id:
                link_proposal_to_result(repo, proposal_id, new_cell_id)

            # 6. Update best
            if best_result.fitness > best_fitness:
                best_fitness = best_result.fitness
                best_cell_id = new_cell_id
                generations_without_improvement = 0
                print(f"🏆 NEW BEST: ${best_fitness:.2f}")
            else:
                generations_without_improvement += 1
        else:
            generations_without_improvement += 1
            print(f"✗ Mutation failed")

        # 7. Check termination
        should_stop, reason = check_termination(
            current_fitness=best_result.fitness,
            best_fitness=best_fitness,
            generation=generation,
            max_generations=max_generations,
            fitness_goal=fitness_goal,
            generations_without_improvement=generations_without_improvement
        )

        if should_stop:
            print(f"\n⚠ Terminating: {reason}")
            break

    # Finalize run
    repo.complete_evolution_run(
        run_id=run_id,
        best_cell_id=best_cell_id,
        termination_reason=reason
    )

    print(f"\n{'='*70}")
    print(f"EVOLUTION COMPLETE")
    print(f"{'='*70}")
    print(f"Best Cell: #{best_cell_id}")
    print(f"Best Fitness: ${best_fitness:.2f}")
    print(f"Total Generations: {generation}")
```

## Multi-Run Strategies

### Walk-Forward Validation

Test evolved strategies on out-of-sample data:

```python
# Train on first 20 days
train_data = load_market_data('PURR', '1H', days=20, offset=0)
best_cell = run_evolution(train_data, max_generations=50)

# Test on next 10 days
test_data = load_market_data('PURR', '1H', days=10, offset=20)
test_fitness = backtest(best_cell.genome, test_data)

print(f"Train fitness: ${best_cell.fitness:.2f}")
print(f"Test fitness: ${test_fitness:.2f}")
print(f"Generalization: {test_fitness / best_cell.fitness:.1%}")
```

### Multi-Symbol Robustness

Test same strategy on multiple symbols:

```python
symbols = ['PURR', 'HFUN', 'BTC', 'ETH', 'SOL']

for symbol in symbols:
    fitness = backtest(best_cell.genome, load_market_data(symbol))
    repo.record_phenotype(best_cell.cell_id, symbol=symbol, ...)
    print(f"{symbol}: ${fitness:.2f}")
```

LLM can then identify: "This pattern works on PURR and HFUN (mid-caps) but fails on BTC (large-cap)"

## Summary

The evolution workflow combines:
- **Cheap exploration** (random mutations)
- **Expensive interpretation** (LLM analysis)
- **Persistent knowledge** (cell database)
- **Multi-timeframe testing** (1H, 4H, 1D)
- **Pattern taxonomy building** (LLM-discovered categories)

**Key insight**: Evolution finds profitable structures blindly. LLM interprets what was found and names it. This creates a growing library of known patterns that future analysis can reference.

Next: See updated existing docs for how this workflow integrates with the current system.
