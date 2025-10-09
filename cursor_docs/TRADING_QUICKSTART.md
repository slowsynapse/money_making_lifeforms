# Trading Evolution System - Quick Start

## 🚀 You're Ready!

The trading evolution system is **fully integrated** into the agent. Here's how to use it:

## Interactive Docker Usage

### 1. Start the Container
```bash
make int
```

### 2. Inside the Container - Run These Commands:

#### Demo the System
```bash
python -m agent_code.agent trading-demo
```
Shows DSL parsing, mutation, and explains the evolutionary concept.

#### 🧠 **Agent Learns to Trade** (The Real Thing!)
```bash
python -m agent_code.agent trading-learn -n 5 -s
```
**The agent generates 5 strategies, sees its fitness after each, and learns to improve!**
- Agent uses LLM reasoning to create strategies
- Sees performance history and learns from mistakes
- Each iteration gets smarter based on feedback
- Open http://localhost:8080 to watch it think!

#### Test a Specific Strategy
```bash
python -m agent_code.agent trading-test --strategy "IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL"
```
Runs a backtest and shows if the strategy survives (fitness > 0).

#### Try Different Strategies
```bash
# Moving average crossover (abstract)
python -m agent_code.agent trading-test --strategy "IF ALPHA(10) > BETA(50) THEN BUY ELSE HOLD"

# Threshold-based
python -m agent_code.agent trading-test --strategy "IF GAMMA(14) < 30 THEN BUY ELSE SELL"

# Parameterless symbols
python -m agent_code.agent trading-test --strategy "IF OMEGA() >= PSI() THEN HOLD ELSE SELL"

# Equality check
python -m agent_code.agent trading-test --strategy "IF DELTA(20) == EPSILON(100) THEN BUY ELSE HOLD"
```

### 3. Run Standard Agent (Original Functionality)
```bash
python -m agent_code.agent -s -p "Write a hello world script"
```
Open http://localhost:8080 to watch the agent work.

## Host Machine Usage (Evolution Loop)

### Prerequisites
On your host (outside Docker):
```bash
pip install -r base_agent/requirements.txt
```

### Run the Full Evolution Loop
```bash
# Run 10 generations of evolution
python3 runner.py --evolution-mode --iterations 10 --workers 1

# Resume from a previous run
python3 runner.py --evolution-mode --experiment-id 1 --iterations 20

# Run with more parallelism
python3 runner.py --evolution-mode --iterations 50 --workers 4
```

## What Each Mode Does

### `trading-demo`
- ✓ Loads the trading benchmark
- ✓ Parses DSL strategies
- ✓ Shows mutations
- ✓ Explains the evolutionary concept
- **No LLM calls**, **No API costs**

### `trading-test`
- ✓ Tests ONE strategy through backtest
- ✓ Shows fitness score
- ✓ Indicates SURVIVED or DIED
- ✓ Displays profit breakdown
- **No LLM calls**, **No API costs**

### `--evolution-mode` (runner.py)
- ✓ Runs MULTIPLE generations
- ✓ Finds best strategy from previous iterations
- ✓ Mutates the best performer
- ✓ Creates evolutionary lineage
- **No LLM calls for strategy generation** (pure DSL mutation)
- Only LLM costs are estimated for fitness calculation

## Understanding the Output

### Successful Strategy
```
Fitness Score: $1881.05
[SURVIVED] Trading Profit: $1881.07, LLM Cost: $0.0165, Fitness: $1881.05, Trades: 1
✓ Strategy SURVIVED (fitness > 0)
```
- Strategy made $1,881 profit
- After subtracting costs: $1,881.05 fitness
- **This strategy will propagate to next generation**

### Failed Strategy
```
Fitness Score: -$9,850.00
[DIED] Agent died on day 45. Portfolio value: $0.00. Balance went to zero.
✗ Strategy DIED (fitness ≤ 0)
```
- Strategy lost all capital mid-backtest
- Immediate termination (oxygen check)
- **This strategy is discarded**

## The DSL Language

### Format
```
IF SYMBOL(N) OPERATOR SYMBOL(N) THEN ACTION ELSE ACTION
```

### Symbols (8 available)
- **Parameterized**: `ALPHA(N)`, `BETA(N)`, `GAMMA(N)`, `DELTA(N)`, `EPSILON(N)`, `ZETA(N)`
- **Parameterless**: `OMEGA()`, `PSI()`
- N = any integer (typically 5-200)

### Operators (5 available)
- `>` greater than
- `<` less than
- `>=` greater or equal
- `<=` less or equal
- `==` equal

### Actions (3 available)
- `BUY` - Enter long position with all capital
- `SELL` - Exit position completely
- `HOLD` - Do nothing

## Key Concepts

### Fitness Function
```
Fitness = Trading Profit - Transaction Costs - LLM API Costs
```

### Financial State Tracking

The agent's financial state is tracked **outside the LLM** as real state:

```python
class TradingBenchmark:
    INITIAL_CAPITAL = 100.0  # Starting with $1,000 (configurable)
    TRANSACTION_COST = 0.10   # $0.10 per trade (configurable)
    
    # Tracked state (not in prompts):
    agent_balance         # Current balance
    total_llm_costs      # Cumulative API costs
    total_trading_profit # Cumulative trading profit
    attempts             # Number of strategies tried
```

**The agent balance is REAL**:
- Starts with $1,000
- Each strategy attempt costs ~$0.02 (LLM API)
- Each successful strategy adds profit to balance
- If balance hits $0, the agent is bankrupt

### To Change Starting Capital

Edit `base_agent/src/benchmarks/trading_benchmarks/trading_benchmark.py`:

```python
class TradingBenchmark(BaseBenchmark):
    INITIAL_CAPITAL = 5000.0  # Change to $5,000
    TRANSACTION_COST = 0.50   # Change to $0.50 per trade
```

These are stored as **class attributes**, not in prompts!

### Survival Rules
1. **Real-time check**: If balance hits $0 → DIE immediately
2. **Post-backtest**: If fitness ≤ 0 → DIE
3. **Only survivors propagate** to next generation
4. **Agent balance tracks across attempts** - it's cumulative state!

### Mutation
Currently mutates **operators only**:
- `IF ALPHA(10) > BETA(50) ...` → `IF ALPHA(10) >= BETA(50) ...`

Future: Can expand to mutate symbols, parameters, actions, add branches, etc.

### Abstract Symbols
- **No predefined meaning**: ALPHA doesn't mean "SMA" or "RSI"
- **Market discovers meaning**: Profitable combinations survive
- **Zero human bias**: No trading folklore injected

## Example Session

```bash
# Start Docker
make int

# Inside container:
bash-5.2$ python -m agent_code.agent trading-demo
# ... shows system overview ...

bash-5.2$ python -m agent_code.agent trading-test --strategy "IF GAMMA(20) > DELTA(50) THEN BUY ELSE SELL"
Fitness Score: $-150.23
✗ Strategy DIED (fitness ≤ 0)

bash-5.2$ python -m agent_code.agent trading-test --strategy "IF ALPHA(10) > BETA(50) THEN BUY ELSE HOLD"
Fitness Score: $1881.05
✓ Strategy SURVIVED (fitness > 0)

bash-5.2$ exit

# On host, run evolution:
python3 runner.py --evolution-mode --iterations 5
# Watches 5 generations evolve through mutation and selection
```

## Files to Explore

- `base_agent/src/dsl/grammar.py` - DSL definition
- `base_agent/src/dsl/interpreter.py` - DSL parser & executor
- `base_agent/src/dsl/mutator.py` - Mutation engine
- `base_agent/src/benchmarks/trading_benchmarks/trading_benchmark.py` - Fitness calculation
- `benchmark_data/trading/ohlcv.csv` - Market data (100 days)
- `runner.py` - Evolution loop orchestrator
- `cursor_docs/DSL_DESIGN.md` - Philosophy & rationale
- `cursor_docs/EVOLUTIONARY_LOOP.md` - System architecture

## Next Steps

1. **Try different strategies** - Use `trading-test` to explore the fitness landscape
2. **Run evolution** - Let natural selection find profitable patterns
3. **Expand mutations** - Add symbol/parameter mutations to `mutator.py`
4. **Add more data** - Expand `ohlcv.csv` with more market regimes
5. **Real-time data** - Connect to Hyperliquid API for live data

## The Vision

Over 100+ generations, this system will discover which abstract symbol combinations (`ALPHA`, `BETA`, `GAMMA`...) consistently generate positive fitness.

**These patterns emerge from market selection, not human assumptions.**

That's the power of evolutionary computation with zero priors! 🧬📈
