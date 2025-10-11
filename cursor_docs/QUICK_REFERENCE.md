# SICA Quick Reference

## What is SICA?

Self-Improving Coding Agent with a focus on **trading strategy evolution** using a custom DSL (Domain Specific Language).

## Quick Start

```bash
# Build Docker image
make image

# Run 100 generations of evolution (FREE)
docker run --rm -v $(pwd):/app sica_sandbox \
  ./trading_cli.py evolve --generations 100 --fitness-goal 50.0

# View results
docker run --rm -v $(pwd):/app sica_sandbox \
  ./trading_cli.py query top-cells --limit 10
```

## Trading Modes

| Mode | Purpose | Cost | Command |
|------|---------|------|---------|
| **demo** | Show how DSL and evolution works | Free | `./trading_cli.py demo` |
| **test** | Backtest a single strategy | Free | `./trading_cli.py test "STRATEGY_DSL"` |
| **evolve** | Pure genetic evolution (no LLM) | Free | `./trading_cli.py evolve -g 100` |
| **learn** | LLM-guided mutations | Paid/Free* | `./trading_cli.py learn -n 10` |

*Free with local Ollama (`--use-local-llm`)

## DSL Examples

```
Basic trend following:
IF DELTA(0) > DELTA(20) THEN BUY ELSE SELL

Multi-timeframe:
IF DELTA(0,1H) > DELTA(20,1H) AND VOLUME(0,1H) > VOLUME(10,1H) THEN BUY ELSE SELL

Mean reversion:
IF DELTA(0) < -2.0 THEN BUY ELSE IF DELTA(0) > 2.0 THEN SELL ELSE HOLD
```

## File Locations

| What | Where |
|------|-------|
| Trading logic | `base_agent/src/trading/` |
| DSL interpreter | `base_agent/src/dsl/` |
| Cell storage | `base_agent/src/storage/` |
| Results | `results/evolve_*/` |
| Cell database | `results/evolve_*/evolution/cells.db` |

## Common Tasks

**Check evolution progress:**
```bash
tail -f results/evolve_*/evolution/evolution_summary.txt
```

**Query database:**
```bash
sqlite3 results/evolve_*/evolution/cells.db "SELECT * FROM cells ORDER BY fitness DESC LIMIT 5;"
```

**Test after code changes:**
```bash
# Restart Docker to flush old code
docker run --rm -v $(pwd):/app sica_sandbox ./trading_cli.py demo
```

## Documentation

- **Implementation tasks**: IMPLEMENTATION_TODO.md
- **Docker usage**: DOCKER_COMMANDS.md
- **Testing**: TESTING.md
- **Vision**: VISION.md
- **Deep technical docs**: `reference/` subdirectory

## Workflow Recommendations

1. **Build cell library**: Run `evolve` with 100+ generations
2. **Analyze patterns**: Run `learn` with local LLM to discover winning strategies
3. **Extract insights**: Query database for top performers
4. **Iterate**: Use insights to guide next evolution run

## Key Concepts

- **Cell**: A single strategy + its performance metrics
- **Generation**: One iteration of the evolutionary loop
- **Fitness**: Profit metric (initial capital + profit)
- **DSL**: Abstract trading language (not executable Python)
- **Mutation**: Random modification of DSL strategy
