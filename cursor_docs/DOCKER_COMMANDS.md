# Docker Commands for Trading Evolution System

This document provides the essential Docker commands for running the trading evolution system. All commands should be run from the root of the project directory.

**Primary Entry Point**: The system should be controlled via `./trading_cli.py`. This script is a wrapper that provides a clean interface to the underlying agent logic.

## Prerequisites

- Docker must be installed and running
- The `sica_sandbox` Docker image must be built (`make image`)
- For local LLM mode, Ollama must be running on the host: `http://localhost:11434`

## Trading Evolution Commands

### 1. Trading-Evolve (Pure Mutation, FREE)

Evolves trading strategies through random DSL mutations. No LLM costs.

**Basic command:**
```bash
docker run --rm \
  -v $(pwd):/app \
  sica_sandbox \
  ./trading_cli.py evolve --generations 100 --fitness-goal 50.0
```

**Parameters (See `./trading_cli.py evolve --help` for full list):**
- `-g, --generations`: Number of generations to evolve
- `-f, --fitness-goal`: Target fitness for early termination
- `--output-dir`: Directory to save results

**Example with custom parameters:**
```bash
docker run --rm \
  -v $(pwd):/app \
  sica_sandbox \
  ./trading_cli.py evolve --generations 200 --fitness-goal 100.0 --symbol BTCUSD
```

**Note on Web Visualization**: The web server is currently being refactored. Use the CLI for monitoring evolution.

**Background mode (long runs):**
```bash
docker run --rm \
  -v $(pwd):/app \
  sica_sandbox \
  ./trading_cli.py evolve --generations 1000 --fitness-goal 500.0 \
  > evolution_run_1.log 2>&1 &
```

### 2. Trading-Learn (LLM-Guided, Intelligent Mutations)

Uses an LLM to analyze cells and propose intelligent mutations. Requires a cell database from a prior `evolve` run.

**With Anthropic Claude (cloud LLM):**
```bash
docker run --rm \
  -v $(pwd):/app \
  -e ANTHROPIC_API_KEY="your-api-key-here" \
  sica_sandbox \
  ./trading_cli.py learn --iterations 10 --cost-limit 1.0
```

**With Local LLM (Ollama - FREE):**
```bash
docker run --rm --network host \
  -v $(pwd):/app \
  sica_sandbox \
  ./trading_cli.py learn --iterations 10 --cost-limit 1.0 --use-local-llm
```

**Parameters (See `./trading_cli.py learn --help` for full list):**
- `-n, --iterations`: Number of LLM-guided mutation iterations
- `-c, --cost-limit`: Maximum LLM cost in USD
- `--use-local-llm`: Flag to use a local Ollama instance

### 3. Trading-Test (Single Strategy Backtest)

Test a specific DSL strategy on historical data.

```bash
docker run --rm \
  -v $(pwd):/app \
  sica_sandbox \
  ./trading_cli.py test "IF DELTA(0) > DELTA(20) THEN BUY ELSE SELL"
```

### 4. Trading-Demo (System Overview)

Shows DSL parsing, mutation, and explains the evolutionary concept.

```bash
docker run --rm \
  -v $(pwd):/app \
  sica_sandbox \
  ./trading_cli.py demo
```

## Database Queries

Query the cell database from inside a container or directly on the host if `sqlite3` is installed. The most reliable method is to use the `query` command of the CLI.

**View top cells from the most recent evolution run:**
```bash
docker run --rm \
  -v $(pwd):/app \
  sica_sandbox \
  ./trading_cli.py query top-cells --limit 10
```

**View lineage of a specific cell:**
```bash
docker run --rm \
  -v $(pwd):/app \
  sica_sandbox \
  ./trading_cli.py query lineage --cell-id 42
```

## Recommended Workflow

### Phase 1: Build Cell Library (FREE)
```bash
# Run 100+ generations of pure mutation to build genetic diversity
docker run --rm \
  -v $(pwd):/app \
  sica_sandbox \
  ./trading_cli.py evolve --generations 100 --fitness-goal 50.0
```

### Phase 2: LLM Pattern Discovery (Local LLM - FREE)
```bash
# Analyze cells and propose intelligent mutations
docker run --rm --network host \
  -v $(pwd):/app \
  sica_sandbox \
  ./trading_cli.py learn --iterations 20 --cost-limit 1.0 --use-local-llm
```

### Phase 3: Analyze Results
```bash
# Query the database for insights
docker run --rm \
  -v $(pwd):/app \
  sica_sandbox \
  ./trading_cli.py query top-cells --limit 5
```

## Ollama Setup (for Local LLM)

Install and start Ollama on the host machine:
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Gemma 2 27B model (or your preferred model)
ollama pull gemma2:27b

# Verify Ollama is running
ollama list
```
Ollama automatically runs on `http://localhost:11434`. The `--network host` flag in the Docker command allows the container to access it.

## Troubleshooting

**Issue**: Cannot connect to Ollama from Docker
- **Solution**: Ensure you are using the `--network host` flag in your `docker run` command.

**Issue**: Permission denied on volumes
- **Solution**: Ensure your current user has permissions to write to the `results/` directory on the host. If running on Linux, you may need to manage Docker permissions or run as root.

**Issue**: `trading_cli.py` not found or not executable
- **Solution**: Ensure you are running the command from the root of the project directory. If needed, make the script executable: `chmod +x trading_cli.py`.

**Issue**: Cell database not found when running `learn` or `query`
- **Solution**: Run the `evolve` command first. It creates the `cells.db` file in its output directory (e.g., `results/evolve_.../`). The `query` command will automatically find the most recent database.

## Output Locations

All outputs are stored in the mounted volume, typically inside the `results/` directory on your host machine:
- **Evolution Runs**: `results/evolve_<timestamp>/`
- **Learn Runs**: `results/learn_<timestamp>/`
- **Cell Database**: `results/evolve_<timestamp>/evolution/cells.db`
- **Best Strategy**: `results/evolve_<timestamp>/evolution/best_strategy.txt`
- **Summary**: `results/evolve_<timestamp>/evolution/evolution_summary.txt`

## Performance Tips

1. **Use local LLM (Ollama)** for cost-free experimentation.
2. **Run `evolve` first** to build a substantial cell library (100+ generations recommended).
3. **Use background mode** for long `evolve` runs by appending `&` and redirecting output.
4. **Use the `query` command** to inspect results without needing to manually find the database file.
5. **Set a `fitness-goal`** to terminate evolution early if a highly profitable strategy is found.

---

**Note**: The old method of calling `python -m base_agent.agent ...` is deprecated. Please use `./trading_cli.py` for all interactions.
