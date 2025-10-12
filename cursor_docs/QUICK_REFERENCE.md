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
| **evolve --dish** | **[NEW]** Named experiment (resumable) | Free | `./trading_cli.py evolve --dish "name" -g 100` |
| **learn** | LLM-guided mutations | Paid/Free* | `./trading_cli.py learn -n 10` |
| **list-dishes** | **[NEW]** Show all experiments | Free | `./trading_cli.py list-dishes` |
| **query** | Query cell database | Free | `./trading_cli.py query summary --dish "name"` |

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
| Dish manager | `base_agent/src/dish_manager.py` |
| **[NEW]** Experiments | `experiments/{dish_name}/` |
| **[NEW]** Dish database | `experiments/{dish_name}/evolution/cells.db` |
| **[NEW]** Dish config | `experiments/{dish_name}/dish_config.json` |
| Legacy results | `results/evolve_*/` (timestamp-based, no --dish flag) |

## Common Tasks

**Create a new experiment:**
```bash
./trading_cli.py evolve --dish "baseline_purr" --generations 100
```

**Resume an experiment:**
```bash
./trading_cli.py evolve --dish "baseline_purr" --resume --generations 100
```

**List all experiments:**
```bash
./trading_cli.py list-dishes
```

**Query experiment results:**
```bash
./trading_cli.py query summary --dish "baseline_purr"
./trading_cli.py query top-cells --dish "baseline_purr" --limit 10
```

**Check evolution progress:**
```bash
tail -f experiments/baseline_purr/evolution/evolution_summary.txt
```

**Query database directly:**
```bash
sqlite3 experiments/baseline_purr/evolution/cells.db "SELECT cell_name, generation, fitness FROM cells ORDER BY fitness DESC LIMIT 5;"
```

**Test after code changes:**
```bash
# Restart Docker to flush old code (if using Docker)
docker run --rm -v $(pwd):/app sica_sandbox ./trading_cli.py demo
```

## Documentation

- **Implementation tasks**: IMPLEMENTATION_TODO.md
- **Docker usage**: DOCKER_COMMANDS.md
- **Testing**: TESTING.md
- **Vision**: VISION.md
- **Deep technical docs**: `reference/` subdirectory

## Workflow Recommendations

### Petri Dish Workflow (Recommended)

1. **Create named experiments**: `./trading_cli.py evolve --dish "baseline" -g 100`
2. **Build cell library**: Continue experiments with `--resume`
3. **Compare approaches**: Create multiple dishes with different parameters
4. **Analyze patterns**: Run `learn` with local LLM on specific dishes
5. **Extract insights**: Query dish-specific results
6. **Iterate**: Resume best-performing dishes

### Legacy Workflow

1. **Build cell library**: Run `evolve` with 100+ generations (creates timestamp-based folder)
2. **Analyze patterns**: Run `learn` with local LLM to discover winning strategies
3. **Extract insights**: Query database for top performers
4. **Iterate**: Use insights to guide next evolution run

## Key Concepts

- **Cell**: A single strategy + its performance metrics
- **Cell Name**: **[NEW]** Unique identifier like `baseline_purr_g114_c001` (dish_generation_counter)
- **Dish**: **[NEW]** Named experiment with isolated database (e.g., "baseline_purr", "aggressive_mut")
- **Generation**: One iteration of the evolutionary loop
- **Fitness**: Profit metric (initial capital + profit)
- **DSL**: Abstract trading language (not executable Python)
- **Mutation**: Random modification of DSL strategy
- **Resume**: **[NEW]** Continue evolution from last generation in a dish
