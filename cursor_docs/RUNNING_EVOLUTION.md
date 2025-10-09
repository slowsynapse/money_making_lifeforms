# Running the Trading Evolution System

## Quick Start

The system has three modes:

1. **Evolution Mode** (Offline, Free) - Pure mutation-based evolution, no LLM costs
2. **Trading-Learn Mode** (LLM-Powered) - Combines evolution with pattern analysis
3. **Meta-Agent Mode** (Traditional) - Agent modifies its own code (deprecated for trading)

## Running Trading Evolution

### Option 1: Evolution Mode (Offline, Free)

Pure natural selection with no LLM costs after Gen 0:

```bash
# Inside Docker container
docker run --rm \
  -v ${PWD}/base_agent:/home/agent/agent_code:ro \
  -v ${PWD}/benchmark_data:/home/agent/benchmark_data:ro \
  -v ${PWD}/results/interactive_output:/home/agent/workdir:rw \
  sica_sandbox \
  python -m agent_code.agent trading-evolve -g 50 -f 100.0
```

**What happens:**
- Generation 0: Random initial strategy → Backtest → Birth Cell #1
- Generations 1-50:
  1. Mutate best cell
  2. Backtest on 1H, 4H, 1D data
  3. If better → Birth new cell
  4. If worse → Record failure (statistics only)
  5. Repeat

**Output:**
- Cell database: `results/interactive_output/evolution/cells.db`
- Best strategy: `results/interactive_output/evolution/best_strategy.txt`
- Summary: `results/interactive_output/evolution/evolution_summary.txt`

**Cost**: $0 after Gen 0 (no LLM calls)

### Option 2: Trading-Learn Mode (LLM-Powered)

Combines evolution with pattern discovery:

```bash
docker run --rm -p 8080:8080 \
  -v ${PWD}/base_agent:/home/agent/agent_code:ro \
  -v ${PWD}/benchmark_data:/home/agent/benchmark_data:ro \
  -v ${PWD}/results/interactive_output:/home/agent/workdir:rw \
  --env-file .env \
  sica_sandbox \
  python -m agent_code.agent trading-learn --iterations 10 --server
```

**What happens:**
- Each iteration:
  1. LLM generates/mutates strategy
  2. Backtest on real data
  3. If fitness > 0 → Birth cell + LLM analyzes pattern
  4. Store semantic analysis (name, category, hypothesis)
  5. LLM proposes intelligent mutations for next round

**Output:**
- Cell database with LLM analysis
- Pattern taxonomy
- Mutation proposals

**Cost**: ~$0.02 per iteration (LLM generation + analysis)
**Advantage**: Faster convergence, interpretable patterns

### Option 2: Interactive Single Generation

To run one generation at a time and inspect results:

```bash
# Initialize experiment (run 1 iteration)
python runner.py --evolution-mode --iterations 1

# Check the results
cat results/run_1/agent_0/benchmarks/trading/results.jsonl

# Run next generation
python runner.py --evolution-mode --experiment-id 1 --iterations 2

# And so on...
```

### Option 3: Test Mode (No Evolution)

Just test the trading benchmark on the latest agent:

```bash
# First, make sure you have at least one agent iteration
python runner.py --evolution-mode --iterations 1

# Then test the trading benchmark
python runner.py test --name trading
```

## Querying the Cell Database

All successful strategies are stored in SQLite. Here's how to explore them:

### View Top Cells

```bash
# Top 10 by fitness
sqlite3 results/interactive_output/evolution/cells.db \
  "SELECT cell_id, fitness, llm_name, generation FROM cells WHERE status='online' ORDER BY fitness DESC LIMIT 10"
```

### Get Cell Details

```bash
# Full cell information
sqlite3 results/interactive_output/evolution/cells.db \
  "SELECT * FROM cells WHERE cell_id=47"
```

### Trace Cell Lineage

```python
from base_agent.src.storage.cell_repository import CellRepository
from pathlib import Path

repo = CellRepository(Path("results/interactive_output/evolution/cells.db"))

# Get ancestry
lineage = repo.get_lineage(cell_id=47)

print("Lineage:")
for i, cell in enumerate(lineage):
    indent = "  " * i
    print(f"{indent}└─ Cell #{cell.cell_id} (Gen {cell.generation}): ${cell.fitness:.2f}")
```

Output:
```
Lineage:
└─ Cell #1 (Gen 0): $6.17
  └─ Cell #5 (Gen 12): $15.32
    └─ Cell #15 (Gen 34): $18.45
      └─ Cell #47 (Gen 89): $23.31
```

### Find Unanalyzed Cells

```python
# Get cells needing LLM analysis
unanalyzed = repo.find_unanalyzed_cells(limit=10, min_fitness=5.0)

for cell in unanalyzed:
    print(f"Cell #{cell.cell_id}: {cell.genome} (${cell.fitness:.2f})")
```

### View Pattern Taxonomy

```bash
# Get all discovered patterns
sqlite3 results/interactive_output/evolution/cells.db \
  "SELECT pattern_name, category, cells_using_pattern, avg_fitness FROM discovered_patterns ORDER BY avg_fitness DESC"
```

### Get Cells by Pattern

```bash
# Find all "Volume Analysis" cells
sqlite3 results/interactive_output/evolution/cells.db \
  "SELECT c.cell_id, c.fitness, c.llm_name
   FROM cells c
   JOIN cell_patterns cp ON c.cell_id = cp.cell_id
   JOIN discovered_patterns dp ON cp.pattern_id = dp.pattern_id
   WHERE dp.category='Volume Analysis'
   ORDER BY c.fitness DESC"
```

### Watch Evolution Progress

```bash
# In one terminal, run the evolution
python runner.py --evolution-mode --iterations 10

# In another terminal, tail the logs
tail -f results/run_1/agent_*/benchmarks/trading/results.jsonl
```

### Visualize the Best Strategies

```bash
# Extract all strategies and their fitness scores
for i in {0..9}; do
  echo "=== Generation $i ==="
  cat results/run_1/agent_$i/benchmarks/trading/trend_following_1/answer/answer.txt 2>/dev/null || echo "No strategy"
  cat results/run_1/agent_$i/benchmarks/trading/results.jsonl 2>/dev/null | jq -r '.score' || echo "No score"
  echo ""
done
```

## Understanding the Output

### Fitness Report Format

```json
{
  "problem_id": "trend_following_1",
  "score": 199.88,
  "discussion": "[SURVIVED] Trading Profit: $200.00, LLM Cost: $0.0165, Fitness: $199.98, Trades: 1. Backtest complete..."
}
```

- **score**: Final fitness (profit - LLM cost)
- **discussion**: Detailed breakdown of the backtest

### Strategy Evolution Examples

```
Generation 0 (Random):
IF GAMMA(14) < OMEGA() THEN BUY ELSE SELL
Fitness: -$9,850.00 (DIED - balance went to zero)

Generation 1 (Mutation):
IF GAMMA(14) >= OMEGA() THEN BUY ELSE SELL
Fitness: $5.23 (SURVIVED)

Generation 2 (Mutation):
IF GAMMA(14) >= OMEGA() THEN HOLD ELSE SELL
Fitness: $12.45 (SURVIVED)

Generation 3 (Mutation):
IF ALPHA(10) >= OMEGA() THEN HOLD ELSE SELL
Fitness: $187.34 (SURVIVED - best so far!)
```

## Advanced Options

### Custom Parameters

```bash
# Run with more workers (parallel problem solving)
python runner.py --evolution-mode --iterations 20 --workers 8

# Resume from a specific iteration
python runner.py --evolution-mode --experiment-id 1 --iterations 30
```

### Modify Initial Conditions

Edit `base_agent/src/benchmarks/trading_benchmarks/trading_benchmark.py`:

```python
# Change initial capital
initial_capital = 10000.0  # Start with $10k

# Change transaction costs
transaction_cost = 0.10  # $0.10 per trade

# Change LLM cost estimation
estimated_input_tokens = 5000
estimated_output_tokens = 100
```

### Expand Market Data

Add more rows to `benchmark_data/trading/ohlcv.csv` to test strategies on longer timeframes.

## Monitoring with Web Interface

The agent has a built-in web server for monitoring (port 8080), but for the evolution runner, you'll primarily use logs and file inspection.

## Troubleshooting

### No strategies being generated

Check if generation 0 created a seed strategy:
```bash
cat results/run_1/agent_0/agent_code/seed_strategy.txt
```

### All strategies dying

The market data might be too harsh, or transaction costs too high. Try:
- Using a trending market (edit `ohlcv.csv`)
- Reducing transaction costs
- Increasing initial capital

### Evolution not improving

This is normal early on! Evolution requires:
- Many generations (10-50+)
- Sufficient variation through mutation
- Appropriate selection pressure

Be patient—profitable patterns emerge slowly through random exploration.

## Next Steps

Once you have a few generations running:

1. **Analyze Survivors**: Look at which symbol combinations appear in surviving strategies
2. **Track Lineage**: See how mutations propagate through generations
3. **Identify Patterns**: Do certain operators (>, >=, ==) dominate?
4. **Expand the DSL**: Add more symbols or mutation types
5. **Real Data**: Replace mock data with real market data from Hyperliquid API

## Key Insight

You're not optimizing hyperparameters—you're **discovering which symbolic patterns have economic value** through pure market selection. The DSL's abstract nature means you're not constrained by human assumptions about "what should work."

**The market is the fitness function. Profitability is the selection pressure. Time is the teacher.**
