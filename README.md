<p align="center">
  <h1 align="center">💰 Money Making Lifeforms 🧬</h1>
  <p align="center">AI agents that must earn their own existence through trading profits.</p>
  <p align="center">
    <img src="figures/agent_loop.png" alt="Agent Loop" width="80%"/>
  </p>
</p>

⚠️ **This is a fork** of [self_improving_coding_agent](https://github.com/MaximeRobeyns/self_improving_coding_agent).

## The Concept

**What if an AI agent had to pay for its own compute through trading profits?**

This system implements **evolutionary natural selection** where agents generate trading strategies in an abstract symbolic language with **no predefined technical indicators**. Fitness is purely economic:

```
Fitness = Trading Profit - Transaction Costs - LLM API Costs
```

**If Fitness ≤ 0, the agent dies.**

### The "Oxygen Check"

Like biological organisms need oxygen to survive, these digital organisms need positive cash flow. The agent:
- Starts with initial capital ($100-$10,000 configurable)
- Pays ~$0.02 per strategy generation (LLM API costs)
- Earns money through successful trading strategies
- **Dies immediately** if balance hits $0 during backtesting
- **Dies in selection** if profit doesn't cover costs

This creates genuine evolutionary pressure. No arbitrary benchmarks—only real economic constraints.

### Abstract Symbolic Language (Zero Human Bias)

Instead of using human-designed indicators like RSI or MACD, strategies use meaningless symbols:

```
IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL
IF GAMMA(14) < 30 THEN BUY ELSE HOLD
IF OMEGA() >= PSI() THEN HOLD ELSE SELL
```

**The symbols have no predefined meaning.** The agent doesn't "know" that moving averages exist or that "oversold" is a concept. It must discover profitable patterns through pure trial and error—natural selection with zero priors.

Over hundreds of generations, certain symbol combinations emerge as more fit than others. **These patterns are discovered, not designed.**

See [`cursor_docs/DSL_DESIGN.md`](cursor_docs/DSL_DESIGN.md) for the philosophical rationale.

## How It Works

```
┌──────────────────────────────────────────────────────────┐
│                  EVOLUTIONARY CYCLE                      │
└──────────────────────────────────────────────────────────┘

1. Agent generates DSL trading strategy (abstract symbols)
   ↓
2. Strategy backtested on historical OHLCV data
   ↓
3. Real-time survival check:
   - Balance hits $0? → DIES (score = -10000)
   - Balance positive? → Continue
   ↓
4. Calculate fitness: Profit - LLM Cost
   ↓
5. Evolutionary selection:
   - Fitness > 0? → Strategy SURVIVES and propagates
   - Fitness ≤ 0? → Strategy DIES
   ↓
6. Mutate surviving strategy (operators/symbols/actions)
   ↓
7. Repeat from step 1
```

**No gradients. No supervised learning. No human trading folklore.**

Just mutation, selection, and survival of the economically fit.

See [`cursor_docs/EVOLUTIONARY_LOOP.md`](cursor_docs/EVOLUTIONARY_LOOP.md) for implementation details.

## Quick Start 🚀

> **IMPORTANT**: Always run in Docker for isolation. The agent executes shell commands.

### 1. Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/money_making_lifeforms

# Export at least one LLM API key
export ANTHROPIC_API_KEY=your_key_here  # Recommended: Claude 3.5 Sonnet
export DEEPSEEK_API_KEY=your_key_here   # Budget option
# ... or other providers (OpenAI, Gemini, Fireworks, Vertex)

# Build Docker image
make image  # or make image-mac for Apple Silicon

# Install local dependencies (for evolution runner)
pip install -r base_agent/requirements.txt
```

### 2. Try It Out

**Demo Mode** (No API costs, just explains the system):
```bash
make int  # Enter Docker container
python -m agent_code.agent trading-demo
```

**Test a Single Strategy** (No API costs):
```bash
python -m agent_code.agent trading-test --strategy "IF ALPHA(10) > BETA(50) THEN BUY ELSE HOLD"
```

**Agent Learning Mode** (Uses LLM, costs ~$0.02/iteration):
```bash
python -m agent_code.agent trading-learn -n 5 -s
# Then open http://localhost:8080 to watch it think!
```

**Full Evolution** (Runs generations outside Docker):
```bash
exit  # Leave Docker
python runner.py --evolution-mode --iterations 10 --workers 4
```

### 3. Understand the Results

```bash
# View fitness progression
cat results/run_1/agent_*/benchmarks/trading/results.jsonl | jq -r '.score'

# See evolved strategies
cat results/run_1/agent_*/benchmarks/trading/trend_following_1/answer/answer.txt
```

**Survival indicators:**
- ✅ `Fitness: $199.88` → Strategy SURVIVED (profit > costs)
- ❌ `Fitness: -$9,850.00` → Strategy DIED (balance hit zero)
- ❌ `Fitness: -$0.0065` → Strategy DIED (profit < costs)

See [`TRADING_QUICKSTART.md`](TRADING_QUICKSTART.md) for detailed usage examples.

## Evolution Modes

### Trading Evolution (DSL Mutation)

This runs pure evolutionary computation with the abstract symbolic DSL:

```bash
# Run multiple generations of mutation and selection
python runner.py --evolution-mode --iterations 10 --workers 4

# Resume from a previous experiment
python runner.py --evolution-mode --experiment-id 1 --iterations 20
```

**How it works:**
1. Generation 0 creates a random DSL strategy
2. Each generation:
   - Backtests the current strategy
   - Calculates fitness (profit - costs)
   - If fitness > 0, strategy survives
   - Mutates the strategy (change operators/symbols/actions)
3. Repeat until patterns emerge

Results saved to `results/run_<id>/agent_<n>/benchmarks/trading/`

### Meta-Agent Mode (Traditional Self-Improvement)

The original mode where the agent modifies its own codebase:

```bash
# First, configure benchmarks in base_agent/src/benchmarks/__init__.py
python runner.py --id 1 --workers 6
```

This runs the classic self-improvement loop on coding benchmarks.

## Research Directions

Potential extensions for the Money Making Lifeforms system:

### DSL & Evolution
- [ ] **Expand mutation types**: Mutate symbols, parameters, add nested conditions (AND/OR logic)
- [ ] **Crossover breeding**: Combine successful strategies from multiple lineages
- [ ] **Adaptive mutation rates**: Increase mutation when stuck, decrease when improving
- [ ] **Multi-objective optimization**: Balance profit, risk, and drawdown

### Market Integration
- [ ] **Real-time data**: Connect to Hyperliquid API for live market data
- [ ] **Multi-asset evolution**: Test strategies across BTC, ETH, SOL simultaneously
- [ ] **Paper trading**: Deploy surviving strategies to paper trading accounts
- [ ] **Live deployment**: Run the best evolved strategies with real capital (at your own risk!)

### Economic Modeling
- [ ] **Dynamic capital allocation**: Adjust starting capital based on agent performance
- [ ] **Transaction cost modeling**: Include slippage, market impact, and funding rates
- [ ] **Risk-adjusted fitness**: Sharpe ratio or Sortino ratio instead of raw profit
- [ ] **Multi-agent competition**: Run populations competing for limited capital

### Meta-Learning
- [ ] **Agent learns to mutate**: LLM proposes mutations instead of random changes
- [ ] **Strategy explanation**: Agent analyzes why certain symbols work
- [ ] **Transfer learning**: Apply patterns discovered in one market to another
- [ ] **Self-modification**: Agent improves its own DSL grammar and interpreter

## System Architecture

### Trading Evolution Components

The Money Making Lifeforms system consists of:

**Core DSL Engine:**
- `base_agent/src/dsl/grammar.py` - Abstract symbolic language definition
- `base_agent/src/dsl/interpreter.py` - Strategy parser and backtesting executor
- `base_agent/src/dsl/mutator.py` - Evolutionary mutation logic

**Fitness Evaluation:**
- `base_agent/src/benchmarks/trading_benchmarks/trading_benchmark.py` - Economic survival evaluation
- `benchmark_data/trading/ohlcv.csv` - Historical market data (100 days)

**Evolution Runner:**
- `runner.py` - Multi-generation orchestrator with mutation and selection

**Agent Modes:**
- `trading-demo` - Explains the system (no API costs)
- `trading-test` - Tests a single strategy (no API costs)
- `trading-learn` - Agent uses LLM to generate and improve strategies
- `--evolution-mode` - Pure DSL mutation across generations

### Base Agent Framework

The agent inherits from the original self-improving coding agent framework. It still supports traditional meta-improvement on coding benchmarks, but the primary focus is now trading evolution.

See `base_agent/README.md` for the original agent framework documentation.

```
├── base_agent/
│   ├── src/
│   │   ├── benchmarks/
│   │   │   └── trading_benchmarks/
│   │   │       ├── trading_benchmark.py  # Fitness evaluation
│   │   │       └── problems.py           # Trading problem definitions
│   │   ├── dsl/
│   │   │   ├── grammar.py      # Abstract symbolic language
│   │   │   ├── interpreter.py  # Strategy parser & backtester
│   │   │   └── mutator.py      # Evolutionary mutations
│   │   ├── agents/             # Agent implementations
│   │   ├── llm/                # LLM providers
│   │   ├── tools/              # Agent tools
│   │   ├── web_server/         # Real-time visualization (port 8080)
│   │   └── ...
│   └── tests/
│       ├── benchmarks/
│       │   └── test_trading_benchmark.py  # Comprehensive tests
│       ├── dsl/
│       │   ├── test_interpreter.py
│       │   └── test_mutator.py
│       └── ...
├── benchmark_data/
│   └── trading/
│       └── ohlcv.csv           # Historical market data
├── cursor_docs/
│   ├── DSL_DESIGN.md           # Philosophy & rationale
│   ├── EVOLUTIONARY_LOOP.md    # System architecture
│   ├── RUNNING_EVOLUTION.md    # Usage guide
│   ├── TROUBLESHOOTING.md      # Common issues
│   └── WEB_INTERFACE.md        # Visualization guide
├── results/
│   ├── run_<id>/               # Evolution results
│   └── interactive_output/     # Interactive mode outputs
├── runner.py                   # Evolution orchestrator
├── TRADING_QUICKSTART.md       # Quick reference
└── sandbox/                    # Docker environment
```

### Results Organization

```
results/run_{id}/
├── metadata.json          # Experiment metadata
└── agent_{i}/             # Generation i results
    ├── agent_code/        # Agent/DSL code for this generation
    ├── benchmarks/
    │   └── trading/
    │       ├── results.jsonl         # Fitness scores
    │       │   # {"problem_id": "trend_following_1", "score": 199.88, ...}
    │       ├── trend_following_1/
    │       │   └── answer/
    │       │       └── answer.txt    # Generated DSL strategy
    │       └── traces/               # Detailed execution traces
    └── meta_improvement/  # (If using meta-agent mode)
```

**Key files:**
- `answer.txt` - The DSL strategy that was tested
- `results.jsonl` - Fitness score and survival status
- Higher generation numbers (`agent_5/`, `agent_10/`) contain evolved strategies

## Documentation

Comprehensive guides in [`cursor_docs/`](cursor_docs/):

- **[DSL_DESIGN.md](cursor_docs/DSL_DESIGN.md)** - Why abstract symbols? Why no technical indicators?
- **[EVOLUTIONARY_LOOP.md](cursor_docs/EVOLUTIONARY_LOOP.md)** - How fitness evaluation and natural selection work
- **[RUNNING_EVOLUTION.md](cursor_docs/RUNNING_EVOLUTION.md)** - Detailed usage instructions and monitoring
- **[TROUBLESHOOTING.md](cursor_docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[WEB_INTERFACE.md](cursor_docs/WEB_INTERFACE.md)** - Real-time visualization guide
- **[TRADING_QUICKSTART.md](TRADING_QUICKSTART.md)** - Quick reference for all trading modes

## Warning ⚠️

**This is experimental software that executes trading strategies. It is for research and educational purposes only.**

- The system uses historical backtesting, not real trading
- No guarantees of profitability
- Evolved strategies may overfit to training data
- Use at your own risk if deploying with real capital
- Past performance does not indicate future results

**The "survival" mechanism is a research metaphor for economic constraints, not financial advice.**

## Citation & Attribution

### This Fork

This "Money Making Lifeforms" fork explores evolutionary trading with abstract symbolic languages:

```
@misc{money_making_lifeforms_2025,
    title={Money Making Lifeforms: Evolutionary Trading with Economic Survival Constraints},
    author={[Your Name]},
    year={2025},
    note={Fork of SICA with trading evolution and abstract symbolic DSL},
    url={https://github.com/YOUR_USERNAME/money_making_lifeforms}
}
```

### Original Work

Based on the SICA (Self-Improving Coding Agent) framework:

```
@inproceedings{
    robeyns2025sica,
    title={{SICA} A Self-Improving Coding Agent},
    author={Maxime Robeyns, Martin Szummer, and Laurence Aitchison},
    booktitle={ICLR 2025 Workshop on Scaling Self-Improving Foundation Models},
    year={2025},
    url={https://openreview.net/forum?id=rShJCyLsOr}
}
```

Original repository: [github.com/MaximeRobeyns/self_improving_coding_agent](https://github.com/MaximeRobeyns/self_improving_coding_agent)

---

**"Nature doesn't know about RSI or MACD. Let's give the agents the same freedom."** 🧬
