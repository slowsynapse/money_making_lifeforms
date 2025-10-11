# Project Vision: The Autonomous Trading Agent

## 1. Overview

This document outlines the long-term vision for this project. The ultimate goal is to evolve the `base_agent` from a tool-assisted framework into a fully **autonomous agent** capable of managing the entire lifecycle of its own trading application.

In its final form, the agent will operate as a "meta-programmer" or "AI software developer." It will use its own capabilities—running command-line tools, editing code, analyzing results—to continuously and automatically improve its trading strategies and the underlying software that runs them.

> **Disclaimer:** This is a forward-looking vision document. The `base_agent` is currently under active development, and the functionalities described herein are not yet implemented. The core components exist, but they are not yet integrated into the autonomous loop described below.

---

## 2. The Autonomous Loop: Core Capabilities

The agent's operation will be based on an autonomous, self-driven loop. Instead of a human running scripts like `trading-evolve`, the `base_agent` itself will initiate and manage these processes. The key capabilities it will need are:

### 2.1. Self-Execution and Environment Control

-   **CLI as a Tool**: The agent will have a tool that allows it to execute shell commands. Its primary use for this will be to run the trading application's CLI (`trading-evolve`, `trading-learn`, `query-cells`, etc.).
-   **Process Management**: The agent will be able to start, monitor, and terminate the trading application processes, capturing their output (logs, errors, results) for analysis.

### 2.2. Performance Analysis and Hypothesis Generation

-   **Log and Data Analysis**: After a trading run, the agent will read the output logs, performance metrics, and the state of the `cells.db`.
-   **Hypothesis Formulation**: Based on this analysis, the agent will form hypotheses about how to improve performance. Examples of hypotheses include:
    -   *"The evolutionary process is stagnating. Modifying the mutation logic in `mutator.py` might introduce more novelty."*
    -   *"The DSL lacks a way to express time-based conditions. I should add a `TIME()` function to `dsl/grammar.py`."*
    -   *"The backtester is too slow. I can optimize the `_run_backtest` function in `trading_benchmark.py`."*

### 2.3. Self-Editing and Code Modification

-   **Code Manipulation Tools**: The agent will use its existing file editing and code writing tools (`edit_file`, `write_file`) to modify its own codebase.
-   **Targeted Changes**: It will make targeted, surgical changes based on its hypotheses. This could involve modifying the DSL, tweaking the evolutionary algorithm, or even fixing bugs it discovers in its own code.

### 2.4. Self-Testing and Validation

-   **Unit and Integration Testing**: After making code changes, the agent will run the project's test suite (`pytest`) to ensure its modifications haven't broken existing functionality.
-   **Validation Runs**: If tests pass, it will initiate a new, short `trading-evolve` run to validate whether its changes had the intended effect on trading performance.

---

## 3. A Day in the Life of the Autonomous Agent

Here is a narrative of how the agent would function in its ideal state:

1.  **Initiation**: The agent decides to start a new evolution run. It uses its CLI tool to execute: `python -m base_agent.trading_app.runner --mode evolve --generations 50`.
2.  **Monitoring**: It monitors the process, observing the fitness scores being logged to the console. It sees that the best fitness has plateaued at `$15.00` for the last 20 generations.
3.  **Analysis**: The run completes. The agent reads the final report and queries the `cells.db` to inspect the genomes of the top-performing cells. It notices that all the best strategies are simple variations of `ALPHA(10) > BETA(50)`.
4.  **Hypothesis**: The agent concludes: *"The mutation operator is not creating enough structural diversity. I need to add a mutation that combines two existing conditions with an AND operator."*
5.  **Self-Editing**:
    -   The agent reads `base_agent/trading_app/dsl/mutator.py`.
    -   It identifies the main mutation function and adds new logic to implement the "AND" mutation.
    -   It saves the file.
6.  **Self-Testing**:
    -   The agent runs `pytest base_agent/trading_app/dsl/`.
    -   All tests pass.
7.  **Validation**:
    -   The agent starts a new, shorter validation run: `python -m base_agent.trading_app.runner --mode evolve --generations 10`.
    -   It observes the output and sees new strategies appearing with compound conditions. The fitness quickly surpasses the previous `$15.00` plateau.
8.  **Conclusion**: The agent determines its change was successful. It logs the results of its experiment in `agent_change_log.md` and begins the cycle anew.

---

## 4. Current State vs. Future Vision

-   **Current**: The project contains two main components that are loosely coupled: the `base_agent` framework and the trading application logic. A human operator is required to run the CLI, interpret the results, and make code changes.
-   **Future**: The `base_agent` will be the operator. The trading application will be the "environment" that the agent acts upon, using its core tools to close the autonomous loop.

This vision provides a clear architectural goal: to continue refining the `base_agent`'s core capabilities (planning, tool use, code editing) and to solidify the `trading_app` as a well-defined, command-line-driven tool that the agent can operate.
