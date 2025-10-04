# Evolutionary Loop: Trading Survival with "Oxygen Check"

## Overview

This self-improving agent system uses a **natural selection** model based on trading profitability. The agent only survives if it can generate more profit than it costs to run. This creates evolutionary pressure that drives the agent towards increasingly profitable strategies.

The agent generates strategies in an **abstract symbolic DSL** with no predefined technical analysis concepts, allowing pure evolutionary discovery.

## Core Concept: Fitness = Profit - Costs

The agent's fitness is calculated as:

```
Fitness = Trading Profit - Transaction Costs - LLM API Costs
```

**If Fitness ≤ 0, the agent dies.**

This is not an arbitrary benchmark score—it's a real survival constraint. The agent must pay for its own existence through trading profits.

## The Domain-Specific Language (DSL)

The agent generates strategies using an abstract symbolic language:

```
IF SYMBOL(N) OPERATOR SYMBOL(N) THEN ACTION ELSE ACTION
```

**Available Symbols:**
- `ALPHA(N)`, `BETA(N)`, `GAMMA(N)`, `DELTA(N)`, `EPSILON(N)`, `ZETA(N)` - Parameterized symbols
- `OMEGA()`, `PSI()` - Parameterless symbols

**Operators:**
- `>`, `<`, `>=`, `<=`, `==`

**Actions:**
- `BUY` - Enter position
- `SELL` - Exit position
- `HOLD` - Do nothing

**Example Strategies:**
```
IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL
IF GAMMA(14) < 30 THEN BUY ELSE HOLD
IF OMEGA() >= PSI() THEN HOLD ELSE SELL
```

**Key Design Principle:** The DSL intentionally uses abstract symbols (Greek letters) instead of technical indicator names. This prevents human bias from being injected into the evolutionary process. The agent must discover what these symbols should represent through natural selection, not through pre-programmed "wisdom" about RSI, MACD, or moving averages.

## The "Oxygen Check"

### Two Levels of Survival Checks

#### 1. Real-Time Trading Balance Check (Immediate Death)

During backtesting, the agent's portfolio value is monitored **at every tick**:

```python
portfolio_value = cash + (position * current_price)

if portfolio_value <= 0:
    # AGENT DIES IMMEDIATELY
    return -initial_capital, num_trades, False, "Agent died on day X"
```

If the agent's balance hits zero at any point during the simulation, the backtest terminates immediately with a harsh penalty. This simulates reality: if you lose all your capital, you can't trade anymore.

#### 2. Post-Backtest Fitness Check (Evolutionary Selection)

After the backtest completes (if the agent survived), fitness is calculated:

```python
trading_profit = final_capital - initial_capital
fitness = trading_profit - llm_cost

if fitness <= 0:
    # Agent failed to sustain itself
    survival_status = "DIED"
else:
    survival_status = "SURVIVED"
```

Even if the agent completes the backtest with money remaining, it still fails if the profit doesn't cover the LLM API costs.

## LLM Cost Estimation

We approximate Anthropic's Claude 3.5 Sonnet pricing:
- **Input tokens**: $3 per million tokens
- **Output tokens**: $15 per million tokens

For a typical DSL generation task:
- Estimated input: ~5,000 tokens
- Estimated output: ~100 tokens
- **Total cost**: ~$0.0165 per generation

This is currently a rough estimate. In production, we would use actual token counts from the agent's metering system.

## The Evolutionary Loop

```
┌──────────────────────────────────────────────────────────┐
│                  EVOLUTIONARY CYCLE                      │
└──────────────────────────────────────────────────────────┘

1. Agent generates a DSL trading strategy (abstract symbols)
   ↓
2. Strategy is backtested on historical OHLCV data
   ↓
3. Real-time survival check:
   - Balance hits $0? → DIES (score = -10000)
   - Balance remains positive? → Continue
   ↓
4. Calculate fitness:
   Fitness = Trading Profit - LLM Cost
   ↓
5. Evolutionary selection:
   - Fitness > 0? → Strategy SURVIVES and propagates
   - Fitness ≤ 0? → Strategy DIES and is discarded
   ↓
6. Mutate the surviving strategy (change operators, symbols, actions)
   ↓
7. Repeat from step 1
```

## No Gradients, No Labels, No Supervised Loss, No Human Bias

This is **pure evolutionary computation**:
- There are no labeled examples of "good" strategies.
- There is no gradient descent.
- There is no supervised loss function.
- **There are no predefined technical indicators or trading concepts.**

**The market is the teacher.**

The DSL uses abstract symbols that have no inherent meaning. Over generations, the evolutionary pressure will cause certain symbol combinations to emerge as more fit than others—not because a human told the system "RSI < 30 means oversold," but because those particular symbol combinations happened to generate profit in the market environment.

Strategies evolve from randomness. If a mutation increases fitness, it propagates. If a mutation decreases fitness, it gets replaced. Over time, this mimics natural selection.

## Example Scenarios

### Scenario 1: Agent Survives and Thrives
```
Initial Capital: $10,000
Trading Profit: $200.00
LLM Cost: $0.0165
Fitness: $199.98

Result: [SURVIVED] ✓
```

### Scenario 2: Agent Makes Money but Doesn't Cover Costs
```
Initial Capital: $10,000
Trading Profit: $0.01
LLM Cost: $0.0165
Fitness: -$0.0065

Result: [DIED] ✗
```

### Scenario 3: Agent Loses All Capital Mid-Backtest
```
Initial Capital: $10,000
Day 1: $8,000
Day 2: $3,000
Day 3: $0

Result: [DIED] Backtest terminated early. Score: -$10,000 ✗
```

## Backtesting Details

The backtest runs on OHLCV (Open, High, Low, Close, Volume) data with additional Hyperliquid API fields:
- `timestamp`: Unix timestamp
- `trades`: Number of trades in the period
- `funding_rate`: Perpetual futures funding rate
- `open_interest`: Total open contracts

### Trading Rules
1. **Start with $10,000 initial capital**
2. Execute DSL strategy on each tick (Close price)
3. **BUY**: Buy with all available cash (minus transaction cost)
4. **SELL**: Sell entire position (minus transaction cost)
5. **HOLD**: Do nothing
6. **Transaction cost**: $0.10 per trade
7. **Liquidate** any remaining position at the end

### Portfolio Tracking
```python
cash = initial_capital
position = 0.0  # shares/contracts

# On BUY:
position = (cash - transaction_cost) / current_price
cash = 0

# On SELL:
cash = position * current_price - transaction_cost
position = 0

# Check survival:
portfolio_value = cash + (position * current_price)
if portfolio_value <= 0:
    AGENT DIES
```

## Running the System

### Run the full test suite:
```bash
make test
```

### Run only trading benchmark tests:
```bash
docker run --rm --env-file .env \
  -v "$(pwd)/base_agent":/home/agent/agent_code:rw \
  sica_sandbox \
  pytest agent_code/tests/benchmarks/test_trading_benchmark.py -v
```

### Run the agent interactively:
```bash
make int
```

Then in the container:
```bash
python -m agent_code.agent --server -p "Generate a profitable DSL trading strategy"
```

## Testing the Evolutionary Pressure

We have comprehensive tests in `base_agent/tests/benchmarks/test_trading_benchmark.py`:

1. **test_score_problem_successful_backtest**: Agent survives with positive fitness
2. **test_score_problem_agent_dies_zero_balance**: Agent dies when balance hits zero
3. **test_score_problem_invalid_dsl**: Agent dies when DSL is unparseable
4. **test_score_problem_no_data_file**: Agent dies when market data is missing

## Future Enhancements

1. **Real Token Tracking**: Replace estimated LLM costs with actual token counts from the agent's metering system
2. **Dynamic Initial Capital**: Adjust starting capital based on agent performance history
3. **Multi-Asset Backtesting**: Test strategies across multiple trading pairs
4. **Real-Time Data Integration**: Connect to Hyperliquid API for live data
5. **Advanced Mutations**: Implement more sophisticated DSL mutation strategies (add/remove branches, change indicators, etc.)
6. **Population-Based Evolution**: Run multiple agent instances simultaneously and breed the best performers

## Key Files

- `base_agent/src/benchmarks/trading_benchmarks/trading_benchmark.py`: Core fitness evaluation
- `base_agent/src/dsl/grammar.py`: DSL language definition
- `base_agent/src/dsl/interpreter.py`: DSL parsing and execution
- `base_agent/src/dsl/mutator.py`: DSL mutation logic
- `benchmark_data/trading/ohlcv.csv`: Historical market data
- `base_agent/tests/benchmarks/test_trading_benchmark.py`: Comprehensive tests
