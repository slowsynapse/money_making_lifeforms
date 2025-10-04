# Running the Trading Evolution System

## Quick Start

The system has two modes:

1. **Evolution Mode** (Trading DSL) - Strategies evolve through mutation
2. **Meta-Agent Mode** (Traditional) - Agent modifies its own code

## Running Trading Evolution

### Option 1: Full Evolutionary Loop

This runs multiple generations automatically:

```bash
# Run 10 generations of evolution
python runner.py --evolution-mode --iterations 10 --workers 4
```

**What happens:**
- Generation 0: Creates a random initial DSL strategy
- Generations 1-9: Each iteration:
  1. Runs the current strategy on the trading benchmark
  2. Calculates fitness (profit - costs)
  3. If fitness > 0, strategy survives
  4. Mutates the best strategy to create the next generation
  5. Repeats

**Monitoring:**
- Watch the logs for fitness scores and mutations
- Results saved to `results/run_X/agent_Y/benchmarks/trading/`

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

## Viewing Results

### Check Fitness Scores

```bash
# View results for a specific generation
cat results/run_1/agent_0/benchmarks/trading/results.jsonl | jq .

# See the strategy that was tested
cat results/run_1/agent_0/benchmarks/trading/trend_following_1/answer/answer.txt

# See the full discussion (fitness breakdown)
cat results/run_1/agent_0/benchmarks/trading/results.jsonl | jq -r '.discussion'
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
