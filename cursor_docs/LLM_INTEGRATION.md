# LLM Integration: Pattern Discovery and Semantic Analysis

## Overview

The LLM's role is **not** to design strategies, but to:
1. **Analyze survivors** - Interpret what successful cells are doing
2. **Name patterns** - Create human-readable labels (like humans created "MACD", "RSI")
3. **Build taxonomy** - Organize discovered patterns into categories
4. **Propose mutations** - Suggest intelligent variations based on understanding
5. **Generate hypotheses** - Explain why patterns might work

**Key Principle**: Evolution discovers structure through blind mutation. LLM interprets meaning after the fact.

## The Two-System Collaboration

```
┌────────────────────────────────────┐
│      EVOLUTION (Offline, Free)     │
│  - Generates 1000s of mutations    │
│  - Selects survivors (fitness > 0) │
│  - Fast, parallel, no LLM cost     │
└───────────┬────────────────────────┘
            │
            │ Top 10 survivors
            ▼
┌────────────────────────────────────┐
│    LLM (Selective, Expensive)      │
│  - Analyzes specific cells by ID   │
│  - Names patterns                  │
│  - Proposes intelligent mutations  │
└───────────┬────────────────────────┘
            │
            │ Named patterns + mutation suggestions
            ▼
┌────────────────────────────────────┐
│   EVOLUTION (Guided + Random)      │
│  - 80% random mutations            │
│  - 20% LLM-suggested mutations     │
└────────────────────────────────────┘
```

## When LLM Intervenes

### Trigger Points

1. **Every N generations** (e.g., every 10 generations)
   - Analyze top 5 cells
   - Look for emerging patterns

2. **New best fitness achieved**
   - Immediate analysis of breakthrough cell
   - Understand what changed

3. **User request** (trading-learn mode)
   - Manual analysis of specific cell
   - Deep dive into specific pattern

4. **Pattern clustering detected**
   - When 3+ similar cells survive
   - Identify common pattern

### Selective Analysis

LLM **does not** analyze every mutation. It only looks at:
- Cells with `fitness > 0` (survivors)
- Top performers (by fitness ranking)
- Unanalyzed cells (where `llm_analyzed_at IS NULL`)

This keeps costs low while maximizing insight.

## Analysis Workflow

### Step 1: Query Unanalyzed Cells

```python
from base_agent.src.storage.cell_repository import CellRepository

repo = CellRepository(db_path)

# Get top 10 unanalyzed survivors
cells_to_analyze = repo.find_unanalyzed_cells(limit=10, min_fitness=5.0)

print(f"Found {len(cells_to_analyze)} cells needing analysis")
```

### Step 2: Prepare Context for LLM

For each cell, gather:

```python
def prepare_cell_context(cell_id: int) -> dict:
    """Prepare rich context for LLM analysis."""

    cell = repo.get_cell(cell_id)
    phenotypes = repo.get_phenotypes(cell_id)
    lineage = repo.get_lineage(cell_id)

    # Load market data where this cell was tested
    market_data = load_market_data(symbol='PURR', timeframe='1h')

    # Get trigger points (when strategy activated)
    triggers = extract_trigger_points(cell.genome, market_data, phenotypes[0])

    return {
        'cell_id': cell.cell_id,
        'genome': cell.genome,
        'fitness': cell.fitness,
        'generation': cell.generation,

        # Performance data
        'phenotype': {
            'win_rate': phenotypes[0].win_rate,
            'total_trades': phenotypes[0].total_trades,
            'sharpe_ratio': phenotypes[0].sharpe_ratio,
            'max_drawdown': phenotypes[0].max_drawdown
        },

        # Lineage context
        'parent': lineage[-2].genome if len(lineage) > 1 else None,
        'ancestry_fitness': [a.fitness for a in lineage],

        # Market context
        'trigger_examples': triggers[:5],  # First 5 trigger points
        'symbol': 'PURR',
        'timeframe': '1h'
    }
```

### Step 3: LLM Prompt Template

```python
ANALYSIS_PROMPT = """
You are analyzing a successful trading strategy that evolved through natural selection.

## Cell Information
- **Cell ID**: {cell_id}
- **Generation**: {generation}
- **Fitness**: ${fitness:.2f}
- **Genome**: {genome}

## Performance Metrics
- Win Rate: {win_rate:.1%}
- Total Trades: {total_trades}
- Sharpe Ratio: {sharpe_ratio:.2f}
- Max Drawdown: {max_drawdown:.2f}%

## Symbol Mappings (Current)
The DSL uses abstract symbols. Currently mapped to raw OHLCV:
- ALPHA = open price
- BETA = high price
- GAMMA = low price
- DELTA = close price
- EPSILON = volume
- ZETA = number of trades
- OMEGA = funding rate
- PSI = open interest

Parameter N means "N candles ago" (lookback).

## Market Context
Symbol: {symbol}, Timeframe: {timeframe}

## Trigger Examples
Here are 5 examples of when this strategy triggered BUY:

{trigger_examples}

## Your Task

1. **Name this pattern**: Give it a short, descriptive name (like humans created "MACD", "RSI", "Head and Shoulders")

2. **Categorize it**: What type of pattern is this?
   - Volume Analysis
   - Mean Reversion
   - Trend Following
   - Momentum
   - Volatility
   - Whale Detection
   - Other: [specify]

3. **Explain what it detects**: In plain English, what market condition does this strategy identify? Why might it be profitable?

4. **Hypothesis**: Why do you think this pattern works? What market inefficiency or behavior does it exploit?

5. **Confidence**: How confident are you in this analysis? (0.0 - 1.0)

6. **Suggest mutations**: Propose 3 intelligent variations of this strategy that might improve it further.

Respond in JSON format:
{{
  "name": "Volume Spike Reversal",
  "category": "Volume Analysis",
  "explanation": "...",
  "hypothesis": "...",
  "confidence": 0.85,
  "mutations": [
    "IF EPSILON(0)/AVG(EPSILON, 0, 20) > 2.0 AND DELTA(0) < DELTA(10) THEN BUY ELSE HOLD",
    "...",
    "..."
  ]
}}
"""
```

### Step 4: Call LLM

```python
async def analyze_cell_with_llm(cell_id: int) -> dict:
    """Analyze a cell using LLM."""

    context = prepare_cell_context(cell_id)

    # Format prompt
    prompt = ANALYSIS_PROMPT.format(**context)

    # Call LLM (using existing agent infrastructure)
    from base_agent.src.llm.client import get_llm_client

    client = get_llm_client()
    response = await client.generate(
        prompt=prompt,
        max_tokens=1000,
        temperature=0.3  # Lower temperature for analysis
    )

    # Parse JSON response
    import json
    analysis = json.loads(response.content)

    return analysis
```

### Step 5: Store Analysis

```python
async def analyze_and_store(cell_id: int):
    """Analyze cell and store results."""

    analysis = await analyze_cell_with_llm(cell_id)

    # Store semantic analysis
    repo.update_llm_analysis(
        cell_id=cell_id,
        name=analysis['name'],
        category=analysis['category'],
        hypothesis=analysis['hypothesis'],
        confidence=analysis['confidence']
    )

    # Check if pattern already exists
    pattern = repo.find_pattern_by_name(analysis['name'])

    if not pattern:
        # Create new pattern
        pattern_id = repo.create_pattern(
            name=analysis['name'],
            category=analysis['category'],
            description=analysis['explanation'],
            discovered_by_cell_id=cell_id
        )
    else:
        # Pattern already discovered
        pattern_id = pattern.pattern_id

    # Link cell to pattern
    repo.link_cell_to_pattern(
        cell_id=cell_id,
        pattern_id=pattern_id,
        confidence=analysis['confidence']
    )

    print(f"✓ Cell #{cell_id} analyzed: {analysis['name']}")

    return analysis
```

## Pattern Naming Examples

### Example 1: Volume Spike Reversal

**Genome**:
```
IF EPSILON(0)/EPSILON(20) > 1.5 AND DELTA(0) < DELTA(10) THEN BUY ELSE HOLD
```

**LLM Analysis**:
```json
{
  "name": "Volume Spike Reversal",
  "category": "Volume Analysis",
  "explanation": "Buys when current volume is 50%+ above 20-period average AND price is below 10-period value. Detects institutional accumulation during local dips.",
  "hypothesis": "Large buyers enter during price weakness to minimize market impact. The volume spike indicates serious accumulation, while the price dip provides favorable entry. Similar to 'volume climax' in classical TA but mathematically precise.",
  "confidence": 0.85
}
```

### Example 2: Momentum Breakout (if DSL V2 existed)

**Genome** (hypothetical):
```
IF (DELTA(0) - DELTA(10)) / DELTA(10) > 0.05 THEN BUY ELSE HOLD
```

**LLM Analysis**:
```json
{
  "name": "Momentum Breakout",
  "category": "Trend Following",
  "explanation": "Buys when 10-period price change exceeds 5%. Rides short-term momentum.",
  "hypothesis": "Crypto markets exhibit momentum persistence in the 1-10 hour timeframe. Once price moves 5%+, it tends to continue due to FOMO and liquidation cascades.",
  "confidence": 0.72
}
```

### Example 3: Contrarian Volume Fade

**Genome**:
```
IF EPSILON(0) < EPSILON(20) AND DELTA(0) > DELTA(5) THEN SELL ELSE HOLD
```

**LLM Analysis**:
```json
{
  "name": "Contrarian Volume Fade",
  "category": "Mean Reversion",
  "explanation": "Sells when volume is below 20-period average but price is above 5-period value. Fades price moves that lack volume confirmation.",
  "hypothesis": "Price increases without volume support are unsustainable. Likely driven by thin orderbooks or stop-hunt rather than genuine demand. Mean reversion is imminent.",
  "confidence": 0.68
}
```

## Building Pattern Taxonomy

### Automatic Categorization

As the LLM analyzes cells, it builds a taxonomy:

```python
def get_pattern_taxonomy() -> dict:
    """Get all discovered patterns grouped by category."""

    patterns = repo.get_all_patterns()

    taxonomy = {}
    for pattern in patterns:
        if pattern.category not in taxonomy:
            taxonomy[pattern.category] = []

        taxonomy[pattern.category].append({
            'name': pattern.pattern_name,
            'cells': pattern.cells_using_pattern,
            'avg_fitness': pattern.avg_fitness,
            'best_fitness': pattern.best_fitness
        })

    return taxonomy
```

**Example Output**:
```python
{
  "Volume Analysis": [
    {"name": "Volume Spike Reversal", "cells": 12, "avg_fitness": 18.45, "best_fitness": 25.67},
    {"name": "Contrarian Volume Fade", "cells": 5, "avg_fitness": 8.23, "best_fitness": 12.10}
  ],
  "Mean Reversion": [
    {"name": "Overbought Correction", "cells": 8, "avg_fitness": 15.32, "best_fitness": 19.88}
  ],
  "Trend Following": [
    {"name": "Momentum Breakout", "cells": 15, "avg_fitness": 22.11, "best_fitness": 31.45}
  ]
}
```

## Intelligent Mutation Proposals

### LLM Suggests Variations

After analyzing a cell, LLM proposes variations:

```python
def propose_mutations(cell_id: int) -> list[str]:
    """Get LLM-suggested mutations for a cell."""

    cell = repo.get_cell(cell_id)

    # This was stored during analysis
    mutations = get_llm_proposed_mutations(cell_id)

    return mutations
```

**Example**:
```python
Cell #47: "IF EPSILON(0)/EPSILON(20) > 1.5 THEN BUY ELSE HOLD"

LLM proposes:
1. "IF EPSILON(0)/EPSILON(20) > 2.0 AND DELTA(0) < DELTA(10) THEN BUY ELSE HOLD"
   (Add price dip confirmation)

2. "IF EPSILON(0)/EPSILON(30) > 1.5 THEN BUY ELSE HOLD"
   (Test different lookback window)

3. "IF EPSILON(0)/EPSILON(20) > 1.5 AND ZETA(0) > ZETA(20) THEN BUY ELSE HOLD"
   (Add trade count confirmation)
```

### Hybrid Mutation Strategy

Evolution uses both random and guided mutations:

```python
def get_next_mutation(parent_cell_id: int, mode: str = 'hybrid') -> str:
    """Get next mutation strategy."""

    if mode == 'hybrid':
        # 80% random, 20% LLM-guided
        if random.random() < 0.2:
            # LLM-guided mutation
            suggestions = propose_mutations(parent_cell_id)
            if suggestions:
                return random.choice(suggestions)

    # Fall back to random mutation
    parent = repo.get_cell(parent_cell_id)
    program = interpreter.parse(parent.genome)
    mutated = mutator.mutate(program)
    return mutator.to_string(mutated)
```

## Pattern Evolution Tracking

### Watching Patterns Improve

```python
def track_pattern_evolution(pattern_name: str):
    """Track how a pattern's best fitness evolves over time."""

    cells = repo.find_cells_by_pattern(pattern_name)

    # Group by generation
    by_generation = {}
    for cell in cells:
        if cell.generation not in by_generation:
            by_generation[cell.generation] = []
        by_generation[cell.generation].append(cell.fitness)

    # Plot evolution
    for gen in sorted(by_generation.keys()):
        best = max(by_generation[gen])
        avg = sum(by_generation[gen]) / len(by_generation[gen])
        print(f"Gen {gen:3d}: Best=${best:8.2f}, Avg=${avg:8.2f}, Count={len(by_generation[gen])}")
```

**Output**:
```
Gen   5: Best=$   8.23, Avg=$   8.23, Count=1
Gen  12: Best=$  12.45, Avg=$  10.34, Count=2
Gen  34: Best=$  18.67, Avg=$  14.22, Count=5
Gen  89: Best=$  25.67, Avg=$  19.88, Count=12
```

## Real-Time Analysis in trading-learn Mode

### Interactive LLM Analysis

```bash
# Run trading-learn with LLM analysis
docker run --rm -p 8080:8080 \
  -v ${PWD}/base_agent:/home/agent/agent_code:ro \
  -v ${PWD}/benchmark_data:/home/agent/benchmark_data:ro \
  -v ${PWD}/results/interactive_output:/home/agent/workdir:rw \
  --env-file .env \
  sica_sandbox \
  python -m agent_code.agent trading-learn --iterations 5 --server
```

**What happens**:
1. Agent generates strategy
2. Backtest determines fitness
3. If fitness > 0, **LLM immediately analyzes**
4. Cell is birthed with semantic metadata
5. Pattern taxonomy is updated
6. Next iteration uses both random + guided mutations

## Costs and Budgeting

### LLM Cost per Analysis

**Estimated tokens**:
- Input: ~1500 tokens (cell context + trigger examples)
- Output: ~300 tokens (analysis JSON)

**Cost** (Claude 3.5 Sonnet):
- Input: 1500 * $3/1M = $0.0045
- Output: 300 * $15/1M = $0.0045
- **Total: ~$0.009 per analysis**

### Budget Management

```python
def should_analyze_cell(cell_id: int, budget_remaining: float) -> bool:
    """Decide if we should analyze this cell given budget."""

    COST_PER_ANALYSIS = 0.01  # Conservative estimate

    if budget_remaining < COST_PER_ANALYSIS:
        return False

    cell = repo.get_cell(cell_id)

    # Always analyze if fitness is exceptional
    if cell.fitness > 50.0:
        return True

    # Skip if fitness is marginal
    if cell.fitness < 5.0:
        return False

    # Analyze top performers
    top_cells = repo.get_top_cells(limit=10)
    if cell_id in [c.cell_id for c in top_cells]:
        return True

    return False
```

## Summary

The LLM's role:
- **Interprets** successful strategies (doesn't design them)
- **Names** patterns (like humans named "MACD")
- **Builds** taxonomy of discovered patterns
- **Proposes** intelligent mutations
- **Explains** why patterns might work

Key principles:
- **Selective** analysis (only survivors)
- **After-the-fact** interpretation (evolution finds, LLM names)
- **Cost-aware** (budget for analysis)
- **Incremental** knowledge building (taxonomy grows over time)

**The LLM doesn't tell evolution what to do. It interprets what evolution discovered and gives it human-readable names.**

Next: See `DSL_V2_SPEC.md` for how extending the DSL with arithmetic and aggregations will enable richer patterns for the LLM to discover and name.
