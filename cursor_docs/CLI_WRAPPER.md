# Trading System CLI Documentation

This document provides a comprehensive guide to the command-line interface (CLI) for the Trading Evolution System. The CLI is the primary way to interact with the system for evolving, testing, and managing trading strategies.

## Overview

The CLI is implemented in `trading_cli.py` and provides a clean, user-friendly interface built with Python's `argparse` library. It is designed to be self-documenting; you can always get help by running a command with the `-h` or `--help` flag.

**General Usage:**
```bash
./trading_cli.py [COMMAND] [OPTIONS]
```
or
```bash
python trading_cli.py [COMMAND] [OPTIONS]
```

## Global Options

*   `--version`: Displays the version of the Trading Evolution System.
*   `-h`, `--help`: Shows a help message and exits.

## Commands

The CLI is organized into several subcommands, each responsible for a specific function within the system.

### 1. `evolve`

The `evolve` command is the core of the system. It runs a genetic algorithm to discover and optimize trading strategies based on historical data.

**Usage:**
```bash
./trading_cli.py evolve [OPTIONS]
```

**Options:**

*   `-g, --generations INT`: The number of generations the evolution process will run for. (Default: 100)
*   `-f, --fitness-goal FLOAT`: The target fitness score the evolution aims to achieve. The process will stop if this goal is met. (Default: 50.0)
*   `-s, --symbol STR`: The trading symbol to use for the simulation (e.g., 'PURR'). (Default: "PURR")
*   `-c, --initial-capital FLOAT`: The amount of virtual capital to start with for each simulation. (Default: 1000.0)
*   `--stagnation-limit INT`: The number of generations without any improvement in the best fitness before the evolution stops. (Default: 20)
*   `--lenient-cells INT`: The number of new, randomly generated cells (strategies) to introduce at the beginning for diversity. (Default: 100)
*   `--dish STR`: **[NEW]** Named experiment dish (e.g., 'baseline_purr'). Creates a persistent experiment in `experiments/{dish_name}/` that can be resumed later. See **Petri Dish Architecture** section below.
*   `--resume`: **[NEW]** Resume evolution in an existing dish (requires `--dish`). Continues from the last generation.
*   `--output-dir STR`: Specifies a directory to save the results of the evolution run. If not provided, a directory is automatically created with a timestamp in the `results/` folder. **Note:** Not used when `--dish` is specified.

**Examples:**
```bash
# Traditional evolution (timestamp-based)
./trading_cli.py evolve --generations 200 --fitness-goal 150.0 --symbol BTCUSD

# Create a new named experiment dish
./trading_cli.py evolve --dish "baseline_purr" --generations 100

# Resume an existing experiment
./trading_cli.py evolve --dish "baseline_purr" --resume --generations 100
```

### 2. `learn`

The `learn` command uses a Large Language Model (LLM) to analyze and improve existing trading strategies.

**Usage:**
```bash
./trading_cli.py learn [OPTIONS]
```

**Options:**

*   `-n, --iterations INT`: The number of learning cycles to perform. (Default: 10)
*   `-c, --cost-limit FLOAT`: Sets a maximum cost (in USD) for using the cloud-based LLM to prevent unexpected expenses. (Default: 1.0)
*   `--use-local-llm`: A flag to use a local LLM (via Ollama) instead of the default cloud-based one.
*   `--min-cells INT`: The minimum number of cells that must exist in the database before the learning process can start. (Default: 50)
*   `--output-dir STR`: Specifies a directory to save the learning results. An auto-generated directory is created if this is not specified.

**Example:**
```bash
./trading_cli.py learn --iterations 5 --use-local-llm
```

### 3. `test`

The `test` command allows you to backtest a specific trading strategy written in the system's Domain-Specific Language (DSL).

**Usage:**
```bash
./trading_cli.py test "STRATEGY_STRING" [OPTIONS]
```

**Arguments:**

*   `strategy`: A string containing the trading strategy in DSL format. It's important to enclose the strategy in quotes.

**Options:**

*   `-s, --symbol STR`: The trading symbol to test the strategy against. (Default: "PURR")
*   `-c, --initial-capital FLOAT`: The starting capital for the backtest. (Default: 1000.0)
*   `--output-dir STR`: A directory to save the test results, including performance metrics and trade logs.

**Example:**
```bash
./trading_cli.py test "IF DELTA(0) > DELTA(20) THEN BUY ELSE SELL" --symbol ETHUSD
```

### 4. `demo`

The `demo` command runs a pre-configured demonstration of the trading system, showcasing some predefined strategies.

**Usage:**
```bash
./trading_cli.py demo [OPTIONS]
```

**Options:**

*   `--output-dir STR`: Directory to save the demonstration results.

**Example:**
```bash
./trading_cli.py demo
```

### 5. `query`

The `query` command provides tools to inspect the database of evolved trading strategies (cells).

**Usage:**
```bash
./trading_cli.py query [QUERY_TYPE] [OPTIONS]
```

**Query Types:**

*   `summary`: **[NEW]** Shows a summary of a database including total cells, max generation, and best strategy.
*   `top-cells`: Retrieves a list of the top-performing cells based on their fitness score.
*   `lineage`: Traces the evolutionary history of a specific cell.
*   `patterns`: (Placeholder) Intended for discovering common patterns in successful strategies.
*   `runs`: (Placeholder) Intended for querying information about past evolution runs.

**Options:**

*   `--limit INT`: Limits the number of results returned by the query. (Default: 10)
*   `--cell-id INT`: The ID of the cell to use for a `lineage` query.
*   `--min-trades INT`: Filters cells to only include those that have executed a minimum number of trades. (Default: 0)
*   `--dish STR`: **[NEW]** Filter query by dish name. If not specified, uses the most recent database.

**Examples:**
```bash
# Query most recent database
./trading_cli.py query top-cells --limit 5 --min-trades 10
./trading_cli.py query lineage --cell-id 123

# Query specific dish
./trading_cli.py query summary --dish "baseline_purr"
./trading_cli.py query top-cells --dish "baseline_purr" --limit 10
```

### 6. `list-dishes`

**[NEW]** The `list-dishes` command displays all experiment dishes with their statistics.

**Usage:**
```bash
./trading_cli.py list-dishes
```

**Output includes:**
*   Dish name
*   Total cells
*   Total generations
*   Best fitness achieved
*   Creation date
*   Description

**Example:**
```bash
./trading_cli.py list-dishes
```

**Sample Output:**
```
🧫 Experiment Dishes (3 total):

Dish Name                 Cells    Gens   Best Fitness    Created      Description
----------------------------------------------------------------------------------------------------
baseline_purr             156      114    $65.35          2025-10-10   Standard evolution...
aggressive_mut            89       50     $42.12          2025-10-11   High mutation rate...
dsl_v2_test               45       30     $38.50          2025-10-12   Testing DSL v2...
```

### 7. `web`

The `web` command is intended to start a web-based user interface for interacting with the trading system.

**Note:** This feature is currently under development.

**Usage:**
```bash
./trading_cli.py web [OPTIONS]
```

**Options:**

*   `-p, --port INT`: The port number for the web server. (Default: 8081)
*   `--host STR`: The hostname or IP address to bind the web server to. (Default: "localhost")

**Example:**
```bash
./trading_cli.py web --port 8888
```

## Petri Dish Architecture

**[NEW]** The Petri Dish Architecture is a new organizational system for managing multiple evolutionary experiments. Each "dish" represents a named, persistent experiment that can be created, resumed, and compared independently.

### Key Features

1. **Named Experiments**: Each dish has a unique name (e.g., "baseline_purr", "aggressive_mutations")
2. **Persistent Structure**: All data is stored in `experiments/{dish_name}/` with:
   - `dish_config.json` - Experiment metadata
   - `evolution/cells.db` - Cell database with all evolved strategies
   - `runs/` - Directory for individual evolution runs
3. **Resumable**: Continue evolution from any point using `--resume`
4. **Cell Naming**: Each cell has a unique, self-documenting name: `{dish_name}_g{generation}_c{counter}`
   - Example: `baseline_purr_g114_c001` (dish: baseline_purr, generation 114, 1st cell in that generation)
5. **Isolated Databases**: Each dish has its own SQLite database, preventing cross-contamination

### Workflow Example

```bash
# 1. Create a baseline experiment
./trading_cli.py evolve --dish "baseline_purr" --generations 100

# 2. Create a variant with different parameters
./trading_cli.py evolve --dish "aggressive_mut" --generations 100 --lenient-cells 200

# 3. List all experiments
./trading_cli.py list-dishes

# 4. Continue baseline experiment
./trading_cli.py evolve --dish "baseline_purr" --resume --generations 400

# 5. Compare results
./trading_cli.py query top-cells --dish "baseline_purr" --limit 10
./trading_cli.py query top-cells --dish "aggressive_mut" --limit 10

# 6. View experiment summary
./trading_cli.py query summary --dish "baseline_purr"
```

### Directory Structure

```
experiments/
├── baseline_purr/
│   ├── dish_config.json
│   ├── evolution/
│   │   ├── cells.db
│   │   ├── gen_0/
│   │   ├── gen_1/
│   │   └── ...
│   └── runs/
│       ├── run_001_gen0-100/
│       └── run_002_gen101-500/
└── aggressive_mut/
    └── ...
```

### Benefits

- **Organization**: Clear separation between different experiment types
- **Reproducibility**: Each dish maintains complete configuration and history
- **Comparison**: Easy to compare different evolutionary strategies
- **Resume**: Continue experiments without losing context
- **Context Safety**: Dish-specific queries prevent loading entire databases into LLM context

## How it Works

The `trading_cli.py` script serves as a wrapper around the core logic contained in `base_agent/agent.py`. When a command is executed, the CLI parses the arguments and calls the corresponding asynchronous function (e.g., `run_trading_evolve`, `run_trading_learn`). This separation of concerns keeps the CLI clean and focused on user interaction, while the complex business logic resides in the agent module.

The Petri Dish Architecture is managed by `base_agent/src/dish_manager.py`, which handles dish creation, loading, and metadata management. Each dish maintains its own SQLite database via `base_agent/src/storage/cell_repository.py`, with automatic cell naming and generation tracking.
