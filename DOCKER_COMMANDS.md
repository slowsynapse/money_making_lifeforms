# Docker Commands for Trading Evolution System

This document provides the essential Docker commands for running the trading evolution system. All commands run inside Docker containers to avoid dependency issues with the host system.

## Prerequisites

- Docker must be installed and running
- The `sica_sandbox` Docker image must be built
- Ollama must be running on host (for local LLM mode): `http://localhost:11434`

## Trading Evolution Commands

### 1. Trading-Evolve (Pure Mutation, FREE)

Evolves trading strategies through random DSL mutations. No LLM costs after generation 0.

**Basic command:**
```bash
docker run --rm \
  -v $(pwd)/base_agent:/home/agent/agent_code \
  -v $(pwd)/benchmark_data:/home/agent/benchmark_data \
  -v $(pwd)/results/interactive_output:/home/agent/workdir \
  sica_sandbox \
  python -m agent_code.agent trading-evolve -g 100 -f 50.0
```

**Parameters:**
- `-g, --generations`: Number of generations to evolve (default: 10)
- `-f, --fitness-goal`: Target fitness for early termination (default: 200.0)
- `--initial-strategy`: Starting DSL strategy (optional, generates random if not provided)

**Example with custom parameters:**
```bash
docker run --rm \
  -v $(pwd)/base_agent:/home/agent/agent_code \
  -v $(pwd)/benchmark_data:/home/agent/benchmark_data \
  -v $(pwd)/results/interactive_output:/home/agent/workdir \
  sica_sandbox \
  python -m agent_code.agent trading-evolve \
    -g 200 \
    -f 100.0 \
    --initial-strategy "IF DELTA(0) > DELTA(20) THEN BUY ELSE SELL"
```

**Background mode (long runs):**
```bash
docker run --rm \
  -v $(pwd)/base_agent:/home/agent/agent_code \
  -v $(pwd)/benchmark_data:/home/agent/benchmark_data \
  -v $(pwd)/results/interactive_output:/home/agent/workdir \
  sica_sandbox \
  python -m agent_code.agent trading-evolve -g 100 -f 50.0 \
  > evolution_log.txt 2>&1 &
```

### 2. Trading-Learn (LLM-Guided, Intelligent Mutations)

Uses LLM to analyze cells and propose intelligent mutations. Requires a cell database from prior `trading-evolve` run.

**With Anthropic Claude (cloud LLM):**
```bash
docker run --rm \
  -v $(pwd)/base_agent:/home/agent/agent_code \
  -v $(pwd)/benchmark_data:/home/agent/benchmark_data \
  -v $(pwd)/results/interactive_output:/home/agent/workdir \
  -e ANTHROPIC_API_KEY="your-api-key-here" \
  sica_sandbox \
  python -m agent_code.agent trading-learn -n 10 -c 1.0
```

**With Local LLM (Ollama - FREE):**
```bash
docker run --rm --network host \
  -v $(pwd)/base_agent:/home/agent/agent_code \
  -v $(pwd)/benchmark_data:/home/agent/benchmark_data \
  -v $(pwd)/results/interactive_output:/home/agent/workdir \
  -e USE_LOCAL_LLM=true \
  -e OLLAMA_HOST=http://localhost:11434 \
  sica_sandbox \
  python -m agent_code.agent trading-learn -n 10 -c 1.0
```

**Parameters:**
- `-n, --iterations`: Number of LLM-guided mutation iterations (default: 5)
- `-c, --cost-threshold`: Maximum LLM cost in USD (default: None)
- `-s, --server`: Enable web visualization server

**Example with web server:**
```bash
docker run --rm --network host \
  -v $(pwd)/base_agent:/home/agent/agent_code \
  -v $(pwd)/benchmark_data:/home/agent/benchmark_data \
  -v $(pwd)/results/interactive_output:/home/agent/workdir \
  -e USE_LOCAL_LLM=true \
  -e OLLAMA_HOST=http://localhost:11434 \
  sica_sandbox \
  python -m agent_code.agent trading-learn -n 20 -c 5.0 -s
```

### 3. Trading-Test (Single Strategy Backtest)

Test a specific DSL strategy on historical data.

```bash
docker run --rm \
  -v $(pwd)/base_agent:/home/agent/agent_code \
  -v $(pwd)/benchmark_data:/home/agent/benchmark_data \
  -v $(pwd)/results/interactive_output:/home/agent/workdir \
  sica_sandbox \
  python -m agent_code.agent trading-test \
    --strategy "IF DELTA(0) > DELTA(20) THEN BUY ELSE SELL"
```

### 4. Trading-Demo (System Overview)

Shows DSL parsing, mutation, and evolutionary concept.

```bash
docker run --rm \
  -v $(pwd)/base_agent:/home/agent/agent_code \
  -v $(pwd)/benchmark_data:/home/agent/benchmark_data \
  -v $(pwd)/results/interactive_output:/home/agent/workdir \
  sica_sandbox \
  python -m agent_code.agent trading-demo
```

## Database Queries

Query the cell database from the host machine:

**View top cells:**
```bash
sqlite3 results/interactive_output/evolution/cells.db \
  'SELECT cell_id, generation, fitness, status FROM cells ORDER BY fitness DESC LIMIT 10;'
```

**View lineage of best cell:**
```bash
sqlite3 results/interactive_output/evolution/cells.db \
  'SELECT cell_id, generation, fitness, dsl_genome FROM cells WHERE cell_id <= 10 ORDER BY cell_id;'
```

**View evolution runs:**
```bash
sqlite3 results/interactive_output/evolution/cells.db \
  'SELECT * FROM evolution_runs ORDER BY run_id DESC LIMIT 5;'
```

**Count cells by status:**
```bash
sqlite3 results/interactive_output/evolution/cells.db \
  'SELECT status, COUNT(*) FROM cells GROUP BY status;'
```

## Recommended Workflow

### Phase 1: Build Cell Library (FREE)
```bash
# Run 100 generations of pure mutation to build genetic diversity
docker run --rm \
  -v $(pwd)/base_agent:/home/agent/agent_code \
  -v $(pwd)/benchmark_data:/home/agent/benchmark_data \
  -v $(pwd)/results/interactive_output:/home/agent/workdir \
  sica_sandbox \
  python -m agent_code.agent trading-evolve -g 100 -f 50.0
```

### Phase 2: LLM Pattern Discovery (Local LLM - FREE)
```bash
# Analyze cells and propose intelligent mutations
docker run --rm --network host \
  -v $(pwd)/base_agent:/home/agent/agent_code \
  -v $(pwd)/benchmark_data:/home/agent/benchmark_data \
  -v $(pwd)/results/interactive_output:/home/agent/workdir \
  -e USE_LOCAL_LLM=true \
  -e OLLAMA_HOST=http://localhost:11434 \
  sica_sandbox \
  python -m agent_code.agent trading-learn -n 20 -c 1.0
```

### Phase 3: Analyze Results
```bash
# Query database for insights
sqlite3 results/interactive_output/evolution/cells.db \
  'SELECT cell_id, generation, fitness, dsl_genome FROM cells ORDER BY fitness DESC LIMIT 5;'
```

## Ollama Setup (for Local LLM)

Install and start Ollama on the host machine:

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Gemma 3 27B model (or your preferred model)
ollama pull gemma2:27b

# Verify Ollama is running
ollama list

# Ollama automatically runs on http://localhost:11434
# No additional configuration needed for Docker with --network host
```

## Troubleshooting

**Issue**: Cannot connect to Ollama from Docker
- **Solution**: Use `--network host` flag and `OLLAMA_HOST=http://localhost:11434`

**Issue**: Permission denied on volumes
- **Solution**: Ensure directories exist: `mkdir -p results/interactive_output benchmark_data`

**Issue**: Cell database not found
- **Solution**: Run `trading-evolve` first to create the database before running `trading-learn`

**Issue**: Out of memory
- **Solution**: Reduce batch size in LLM analysis or use smaller LLM model

## Output Locations

All outputs are stored in the mounted volumes:

- **Cell Database**: `results/interactive_output/evolution/cells.db`
- **Evolution Logs**: `results/interactive_output/evolution/gen_*/`
- **Best Strategy**: `results/interactive_output/evolution/best_strategy.txt`
- **Summary**: `results/interactive_output/evolution/evolution_summary.txt`

## Performance Tips

1. **Use local LLM (Ollama)** for cost-free experimentation
2. **Run trading-evolve first** to build a cell library (100+ generations recommended)
3. **Use background mode** for long runs (append `&` and redirect output)
4. **Query database** to inspect progress without interrupting evolution
5. **Set fitness goal** to terminate early when target is reached

## Example: Complete Sprint 3 Test

```bash
# Step 1: Build cell library (100 generations, ~10-15 minutes)
docker run --rm \
  -v $(pwd)/base_agent:/home/agent/agent_code \
  -v $(pwd)/benchmark_data:/home/agent/benchmark_data \
  -v $(pwd)/results/interactive_output:/home/agent/workdir \
  sica_sandbox \
  python -m agent_code.agent trading-evolve -g 100 -f 50.0

# Step 2: LLM-guided learning (10 iterations with local LLM)
docker run --rm --network host \
  -v $(pwd)/base_agent:/home/agent/agent_code \
  -v $(pwd)/benchmark_data:/home/agent/benchmark_data \
  -v $(pwd)/results/interactive_output:/home/agent/workdir \
  -e USE_LOCAL_LLM=true \
  -e OLLAMA_HOST=http://localhost:11434 \
  sica_sandbox \
  python -m agent_code.agent trading-learn -n 10 -c 1.0

# Step 3: Review results
sqlite3 results/interactive_output/evolution/cells.db \
  'SELECT cell_id, generation, fitness, status FROM cells ORDER BY fitness DESC LIMIT 10;'
```

---

**Note**: All Python commands must run inside the Docker container. Do not run Python directly on the host if your system Python is broken.
