# Progress Report #1: Money Making Lifeforms - Trading Evolution System

**Date**: 2025-10-06
**Status**: Multi-Strategy DSL System Complete

---

## Executive Summary

Successfully transformed the Money Making Lifeforms project from a software-focused coding agent to a trading-focused evolutionary strategy system. Key achievements:

1. ✅ **Prompt Refactoring**: All agent prompts converted from "software/coding" to "trading/strategy" terminology
2. ✅ **Multi-Strategy DSL**: Implemented support for chaining multiple trading strategies with voting/aggregation
3. ✅ **Fitness-Based Termination**: Evolution now terminates on goal achievement, not just timeouts
4. ✅ **Workspace Cleaning**: Automatic cleanup prevents contamination from previous runs
5. ✅ **Dual Execution Modes**: LLM-guided learning vs pure genetic evolution

---

## 1. Prompt Refactoring (Software → Trading)

### What Changed
Renamed and refactored all agent classes and prompts to focus on trading strategy design instead of software development.

### Files Modified
- `base_agent/src/agents/implementations/coder.py` → **StrategyDesignerAgent**
  - Changed `AGENT_NAME`: "software_developer" → "strategy_designer"
  - Updated `SYSTEM_PROMPT` to focus on evolutionary trading systems
  - Added critical workspace handling instructions

- `base_agent/src/agents/implementations/main_orchestrator.py`
  - Updated orchestrator prompts to trading context
  - Changed delegation logic to reference StrategyDesignerAgent

- `base_agent/src/agents/implementations/review_committee_member.py`
  - Updated to review trading strategies instead of code
  - Focus on fitness evaluation and DSL syntax

- `base_agent/src/agents/implementations/archive_explorer.py`
  - Updated to analyze trading evolution results
  - Search for successful strategy patterns

- `base_agent/src/agents/implementations/problem_solver.py`
  - Minor trading context additions

- `base_agent/src/tools/reasoning_structures/sequential_subagents.py`
  - Updated examples to use StrategyDesignerAgent

### Why This Matters
The original prompts referenced "software development", "coding", "debugging", etc. This confused the agent since the actual task is designing trading strategies using an abstract DSL. The refactoring ensures the agent understands its true purpose.

---

## 2. Multi-Strategy DSL System

### What Changed
Extended the DSL to support **multiple chained strategies** that vote on BUY/HOLD/SELL decisions, instead of only supporting single-rule strategies.

### Architecture

**Before:**
```
IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL
```
One rule → One action

**After:**
```
IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL
IF GAMMA(20) > OMEGA() THEN HOLD ELSE BUY
IF DELTA(5) > PSI(100) THEN BUY ELSE SELL
```
Multiple rules → Aggregated action (via voting)

### Implementation Details

#### A. Grammar Extension (`base_agent/src/dsl/grammar.py`)
```python
class AggregationMode(Enum):
    MAJORITY = "MAJORITY"    # Most common action wins
    UNANIMOUS = "UNANIMOUS"  # All must agree, else HOLD
    FIRST = "FIRST"          # Only use first rule (legacy)

DslProgram = list[Rule]  # Program is now a list of rules
```

#### B. Interpreter Updates (`base_agent/src/dsl/interpreter.py`)
- **Parsing**: Split on newlines/semicolons to parse multiple rules
- **Execution**: Run all rules and collect their individual actions
- **Aggregation**: Vote using MAJORITY/UNANIMOUS/FIRST logic

```python
def _aggregate_actions(self, actions: list[Action]) -> Action:
    if self.aggregation_mode == AggregationMode.MAJORITY:
        action_counts = Counter(actions)
        most_common = action_counts.most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            return most_common[0][0]
        return Action.HOLD  # Tie → HOLD
```

#### C. Mutation Enhancements (`base_agent/src/dsl/mutator.py`)
Three mutation types:
1. **Modify Rule** (60%): Change operator, symbols, parameters, or actions
2. **Add Rule** (25%): Insert a new randomly generated rule (max 5 rules)
3. **Remove Rule** (15%): Delete a rule (min 1 rule)

Each rule has 7 mutable components:
- Operator: `>`, `<`, `>=`, `<=`, `==`
- Indicator 1 & 2: ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA, OMEGA, PSI
- Parameter 1 & 2: Numeric values
- True/False Actions: BUY, SELL, HOLD

### Testing
Created `test_multi_strategy.py` demonstrating:
- Single strategy parsing
- Multi-strategy parsing (3+ rules)
- Aggregation modes (MAJORITY/UNANIMOUS/FIRST)
- Mutation evolution (modify/add/remove rules)

---

## 3. Fitness-Based Termination

### The Problem
**User Request**: "I want it to mutate to multiple strategies, it should terminate not with timeout but simulate trading somehow and if it reach a goal it saves itself"

Previously, evolution only stopped on timeout or max generations. No way to terminate early on success.

### The Solution
Implemented three termination conditions in `trading-evolve` mode:

#### 1. Goal Achievement (Early Success)
```python
if best_fitness >= fitness_goal:
    print(f"🎯 GOAL ACHIEVED! Fitness: ${best_fitness:.2f}")
    break
```
Default goal: **$200 fitness** (configurable via `-f` flag)

#### 2. Stagnation Detection
```python
if generations_without_improvement >= 20:
    print(f"⚠️ STAGNATION DETECTED")
    break
```
Stops if no improvement for **20 consecutive generations**

#### 3. Max Generations
```python
for generation in range(1, generations + 1):
    # Evolution loop
```
Fallback: Stop after specified generations (default: 10)

### Auto-Save Feature
```python
if mutated_fitness > best_fitness:
    best_fitness = mutated_fitness
    best_strategy = mutated_strategy

    # Auto-save best strategy
    best_strategy_file = results_dir / "best_strategy.txt"
    best_strategy_file.write_text(best_strategy)
```

Every time fitness improves, the best strategy is automatically saved to `results/trading_evolve_TIMESTAMP/best_strategy.txt`

---

## 4. Workspace Contamination Fix

### The Problem
**User Observation**: Agent was reverting to human trading terms (MACD, RSI, SMA) during execution.

**Root Cause**: Old `answer.txt` files from previous runs contained human trading strategies. The agent would read these contaminated files and get confused, thinking it should use traditional trading indicators.

### The Solution

#### A. Automatic Cleanup (`trading_benchmark.py`)
```python
async def setup_problem(self, problem: Problem, problem_data_dir: Path, container_name: str) -> None:
    # Clean up old answer.txt if it exists
    agent_outputs = problem_data_dir / "agent_outputs"
    if agent_outputs.exists():
        shutil.rmtree(agent_outputs)
        print(f"Cleaned old agent_outputs directory")
```

#### B. Explicit Prompt Warning (`coder.py` → StrategyDesignerAgent)
```python
CRITICAL WORKSPACE HANDLING:
  - If you see an answer.txt file with human trading terms (RSI, MACD, SMA, moving averages, etc.), IGNORE IT COMPLETELY
  - Old answer.txt files are from previous runs and contain irrelevant human trading strategies
  - Your job is to OVERWRITE answer.txt with a single line of DSL syntax
  - Do NOT analyze, read, or learn from old answer.txt content - it will mislead you
  - Focus only on generating valid DSL: IF SYMBOL(N) OPERATOR SYMBOL(N) THEN ACTION ELSE ACTION
```

#### C. Extended Timeout
Changed from 120s → 180s to give agent more time without rushing into errors.

---

## 5. Dual Execution Modes

The system now supports two distinct modes for different use cases:

### Mode 1: `trading-learn` (LLM-Guided Learning)

**Purpose**: Use Claude LLM to design creative trading strategies

**Features**:
- 🧠 LLM-guided strategy design
- 🌐 Web UI visualization (localhost:8080)
- 📊 Performance history feedback
- 💰 Costs ~$0.02 per iteration

**Usage**:
```bash
# Inside Docker container
python -m agent_code.agent trading-learn -n 5 -s

# -n 5: Run 5 learning iterations
# -s: Enable web server for visualization
```

**When to use**: Exploring new strategy ideas, bootstrapping initial population

---

### Mode 2: `trading-evolve` (Pure Genetic Evolution)

**Purpose**: Evolve strategies through mutation without LLM costs

**Features**:
- 🧬 Pure genetic algorithm evolution
- 💸 FREE after Gen 0 (no LLM calls)
- 🎯 Fitness-based termination
- 💾 Auto-save best strategies
- 📈 Stagnation detection

**Usage**:
```bash
# Inside Docker container
python -m agent_code.agent trading-evolve -g 100 -f 200.0

# -g 100: Run up to 100 generations
# -f 200.0: Target fitness of $200 (terminates early if achieved)
```

**Termination conditions**:
1. Fitness goal achieved (e.g., $200)
2. No improvement for 20 generations (stagnation)
3. Max generations reached

**When to use**: Long-term evolution, cost-effective optimization, scaling to large populations

---

## 6. Docker & Makefile Setup

### Fixed Issues
- Added missing `benchmark_data` volume mount to Makefile
- Ensured proper directory structure for results

### Current Makefile Targets
```bash
make image-mac    # Build Docker image (Apple Silicon)
make image        # Build Docker image (x86_64)
make int          # Interactive container for manual testing
make test         # Run unit tests
```

### Volume Mounts
```makefile
-v ${PWD}/base_agent:/home/agent/agent_code:ro         # Agent code (read-only)
-v ${PWD}/benchmark_data:/home/agent/benchmark_data:ro # OHLCV data (read-only)
-v ${PWD}/results/interactive_output:/home/agent/workdir:rw  # Output (read-write)
```

---

## 7. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Money Making Lifeforms                     │
│                 Evolutionary Trading System                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐         ┌─────────────────┐
│  trading-learn  │         │ trading-evolve  │
│  (LLM-guided)   │         │ (Pure mutation) │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │                           │
         v                           v
┌─────────────────────────────────────────────────────────────┐
│                    DSL Interpreter                           │
│  - Parse multi-rule programs                                 │
│  - Execute with aggregation (MAJORITY/UNANIMOUS/FIRST)       │
│  - Support 8 abstract symbols (ALPHA, BETA, GAMMA, etc.)     │
└────────┬────────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────────┐
│                    Backtest Engine                           │
│  - Simulate trading on OHLCV data                            │
│  - Calculate: Fitness = Profit - Tx Costs - LLM Costs        │
│  - Survival: Fitness > 0 → Propagate to next generation      │
└────────┬────────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────────┐
│                    Mutation Engine                           │
│  - Modify rule: Change operators, symbols, parameters        │
│  - Add rule: Insert new random rule (max 5)                  │
│  - Remove rule: Delete existing rule (min 1)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. DSL Specification

### Abstract Symbols
| Symbol | Parameters | Description |
|--------|-----------|-------------|
| ALPHA(N) | Integer period | Abstract indicator with lookback period N |
| BETA(N) | Integer period | Abstract indicator with lookback period N |
| GAMMA(N) | Integer period | Abstract indicator with lookback period N |
| DELTA(N) | Integer period | Abstract indicator with lookback period N |
| EPSILON(N) | Integer period | Abstract indicator with lookback period N |
| ZETA(N) | Integer period | Abstract indicator with lookback period N |
| OMEGA() | None | Abstract zero-parameter indicator |
| PSI() | None | Abstract zero-parameter indicator |

### Operators
- `>` (greater than)
- `<` (less than)
- `>=` (greater or equal)
- `<=` (less or equal)
- `==` (equal)

### Actions
- `BUY`: Purchase with all available capital
- `SELL`: Liquidate entire position
- `HOLD`: No action

### Syntax
```
Single Rule:
IF SYMBOL(N) OPERATOR SYMBOL(N) THEN ACTION ELSE ACTION

Multi-Rule (newline or semicolon separated):
IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL
IF GAMMA(20) <= OMEGA() THEN HOLD ELSE BUY
IF DELTA(5) == PSI(100) THEN HOLD ELSE BUY
```

### Fitness Formula
```
Fitness = Trading_Profit - Transaction_Costs - LLM_API_Costs

Survival Condition: Fitness > 0
```

---

## 9. Example Workflows

### Workflow 1: LLM Bootstrap → Evolution
```bash
# Step 1: Use LLM to generate initial population (1-2 iterations)
docker run --rm -ti \
    --env-file .env \
    -p 8080:8080 \
    -v ${PWD}/base_agent:/home/agent/agent_code:ro \
    -v ${PWD}/benchmark_data:/home/agent/benchmark_data:ro \
    -v ${PWD}/results/interactive_output:/home/agent/workdir:rw \
    sica_sandbox

# Inside container:
python -m agent_code.agent trading-learn -n 2 -s

# Step 2: Take best strategy and evolve it with pure mutation
# Copy best strategy from results, then:
python -m agent_code.agent trading-evolve -g 100 -f 200.0 -i "IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL"
```

### Workflow 2: Pure Evolution from Random
```bash
# Start with random strategy and evolve
docker run --rm -ti \
    --env-file .env \
    -v ${PWD}/base_agent:/home/agent/agent_code:ro \
    -v ${PWD}/benchmark_data:/home/agent/benchmark_data:ro \
    -v ${PWD}/results/interactive_output:/home/agent/workdir:rw \
    sica_sandbox

# Inside container:
python -m agent_code.agent trading-evolve -g 200 -f 300.0

# Will generate random initial strategy and evolve
# Auto-saves best strategy on each improvement
```

---

## 10. Key Files Reference

### Core DSL Files
- `base_agent/src/dsl/grammar.py` - Type definitions, AggregationMode enum
- `base_agent/src/dsl/interpreter.py` - Parser, executor, aggregation logic
- `base_agent/src/dsl/mutator.py` - Genetic algorithm mutation operators

### Agent Files
- `base_agent/src/agents/implementations/coder.py` - StrategyDesignerAgent (main agent)
- `base_agent/src/agents/implementations/main_orchestrator.py` - Orchestrator agent
- `base_agent/src/agents/implementations/review_committee_member.py` - Strategy reviewer
- `base_agent/src/agents/implementations/archive_explorer.py` - Evolution analyzer

### Benchmark Files
- `base_agent/src/benchmarks/trading_benchmarks/trading_benchmark.py` - Setup, scoring, backtest

### Entry Points
- `base_agent/agent.py` - Main CLI with `trading-learn` and `trading-evolve` commands
- `test_multi_strategy.py` - Demonstration of multi-strategy features

### Build Files
- `Makefile` - Docker build and run targets
- `sandbox/Dockerfile` - Container definition
- `.env` - API keys (not in repo)

---

## 11. Known Limitations & Future Work

### Current Limitations
1. **Fixed Market Data**: Uses single OHLCV CSV file (`benchmark_data/trading/ohlcv.csv`)
2. **Simple Backtest**: No slippage, market impact, or realistic order execution
3. **No Risk Management**: Strategies can go all-in on single trades
4. **Terminal UI Only**: `trading-evolve` mode has no web visualization
5. **Single Aggregation Mode**: Hardcoded to MAJORITY in evolution mode

### Potential Enhancements
- [ ] Multi-dataset evaluation (test on different market conditions)
- [ ] Portfolio-level risk metrics (Sharpe ratio, max drawdown, etc.)
- [ ] Parallel evolution (multiple populations competing)
- [ ] Web UI for `trading-evolve` mode with real-time charts
- [ ] Strategy crossover (combine two parent strategies)
- [ ] Adaptive mutation rates (based on population diversity)
- [ ] Position sizing logic in DSL (not just all-in trades)
- [ ] Walk-forward optimization (train on past, test on future)

---

## 12. Testing Status

### ✅ Completed Tests
- Multi-strategy parsing (newline/semicolon separators)
- Aggregation modes (MAJORITY/UNANIMOUS/FIRST)
- Mutation operators (modify/add/remove rules)
- Workspace cleaning (verified old files are deleted)
- Docker build and run (both Mac and x86_64)

### 🔄 Pending User Testing
- [ ] `trading-learn` mode with web UI (verify localhost:8080 works)
- [ ] `trading-evolve` mode with fitness-based termination
- [ ] Multi-strategy evolution over 100+ generations
- [ ] Auto-save functionality verification
- [ ] Stagnation detection (20 generations without improvement)

---

## 13. Quick Start Guide

### First Time Setup
```bash
# 1. Clone repo and navigate to directory
cd money_making_lifeforms

# 2. Ensure .env file exists with ANTHROPIC_API_KEY
# (Create from .env.example if needed)

# 3. Build Docker image (one-time)
make image-mac  # Or 'make image' for x86_64

# 4. Run interactive container
make int
```

### Inside Container - LLM Mode
```bash
# Generate 3 strategies using Claude LLM
python -m agent_code.agent trading-learn -n 3 -s

# Visit http://localhost:8080 to watch in real-time
```

### Inside Container - Evolution Mode
```bash
# Evolve for up to 50 generations (stops early if goal reached)
python -m agent_code.agent trading-evolve -g 50 -f 150.0

# Check results in /home/agent/workdir/results/trading_evolve_*/
# Best strategy saved to: best_strategy.txt
```

---

## 14. Success Metrics

### What Success Looks Like
1. **Survival Rate**: >30% of generated strategies have Fitness > 0
2. **Fitness Growth**: Best fitness increases over generations
3. **Multi-Strategy Adoption**: Evolved strategies use 2-5 chained rules
4. **Goal Achievement**: Evolution terminates early by reaching fitness_goal
5. **No Contamination**: Agents only generate valid DSL (no MACD/RSI/human terms)

### Current Baseline (To Be Measured)
- Initial random strategy fitness: ~$-50 to $50
- LLM-generated strategy fitness: ~$0 to $100
- Target evolved fitness: >$200

---

## 15. Summary of Changes

| Component | Before | After | Impact |
|-----------|--------|-------|--------|
| **Agent Names** | CodingAgent, software_developer | StrategyDesignerAgent, strategy_designer | Clarity |
| **DSL Programs** | Single rule only | 1-5 chained rules | Strategy complexity |
| **Aggregation** | N/A | MAJORITY/UNANIMOUS/FIRST | Ensemble strategies |
| **Mutation** | Modify rule only | Modify/Add/Remove rules | Evolution diversity |
| **Termination** | Timeout or max iterations | Goal/Stagnation/Max | Efficiency |
| **Persistence** | Manual save | Auto-save on improvement | Usability |
| **Workspace** | Contaminated by old files | Auto-cleaned on setup | Reliability |
| **Execution** | Single mode (LLM) | Dual modes (LLM + Evolution) | Flexibility |
| **Timeout** | 120s | 180s | Robustness |

---

## 16. Questions for Next Session

When testing the system, consider:

1. **Performance**: Does evolution actually improve fitness over generations?
2. **Multi-Strategy**: Do evolved strategies naturally adopt multiple rules? Or stay single-rule?
3. **Stagnation**: Does the 20-generation stagnation threshold feel right? Too short? Too long?
4. **Goal**: Is $200 fitness a reasonable default target? Too easy? Too hard?
5. **Web UI**: Does `trading-learn -s` properly show the web interface at localhost:8080?
6. **Contamination**: Are old `answer.txt` files still causing issues? Or fully resolved?

---

## Appendix A: Command Reference

### Make Commands (Host Machine)
```bash
make image-mac       # Build Docker image for Apple Silicon
make image           # Build Docker image for x86_64
make int             # Launch interactive container
make test            # Run pytest unit tests
make docs            # Generate documentation
```

### Agent Commands (Inside Container)
```bash
# LLM-guided learning
python -m agent_code.agent trading-learn -n <iterations> [-s]

# Pure genetic evolution
python -m agent_code.agent trading-evolve -g <generations> [-f <fitness_goal>] [-i <initial_strategy>]

# Flags:
#   -n: Number of learning iterations
#   -s: Enable web server (localhost:8080)
#   -g: Max generations
#   -f: Fitness goal (default: 200.0)
#   -i: Initial strategy DSL string
```

---

## Appendix B: File Locations

### Results
- LLM mode: `results/interactive_output/agent_outputs/answer.txt`
- Evolution mode: `results/trading_evolve_TIMESTAMP/best_strategy.txt`
- Evolution logs: `results/trading_evolve_TIMESTAMP/evolution_summary.txt`

### Data
- Market data: `benchmark_data/trading/ohlcv.csv`
- Problem statement: Embedded in `trading_benchmark.py`

### Logs
- Agent logs: Check Docker container output or web UI

---

**Report Complete** ✅

**Status**: All requested features implemented and documented. Ready for user testing.

**Next Steps**: User should test both modes and provide feedback on performance, usability, and any bugs encountered.
