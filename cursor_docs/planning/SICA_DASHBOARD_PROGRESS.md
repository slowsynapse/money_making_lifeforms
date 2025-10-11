# SICA Dashboard Vision

## 1. Executive Summary

This document outlines the future vision for the SICA Dashboard, located in the `sica_web/` directory.

The ultimate goal is to create a **single, unified interface** that serves as the primary window into the operations of the Autonomous Trading Agent. It will provide a "mission control" view, combining real-time monitoring of the agent's internal reasoning (call graphs, event streams) with the tangible results of its work (cell evolution, fitness charts).

This vision supersedes the original `SICA_DASHBOARD_PROGRESS.md`, transforming it from a status report into a forward-looking plan.

> **Disclaimer**: This is a vision document. The features described are the end-goal. The current dashboard in `sica_web/` is a starting point, and the components from `trading_web/` will be merged into it.

---

## 2. Core Concept: A Tale of Two Views

The dashboard will be organized around two primary views, likely as tabs, that reflect the agent's dual nature: its thought process and its actions' consequences.

### View 1: Agent Control Center (The "Mind")

This view is for observing the agent's real-time decision-making process as it executes its autonomous loop.

**Key Components**:

-   **Autonomous Loop Control**: A simple "Start/Stop" control to initiate or halt the agent's entire autonomous operation.
-   **Orchestration Phase Tracker**: A visual indicator (e.g., a stepper or timeline) that highlights which phase of the "Ideal Orchestration Flow" the agent is currently in. This provides immediate insight into the agent's strategy.
    -   *Phase 1: Context Reconstruction*
    -   *Phase 2: Hypothesis Formation*
    -   *Phase 3: Multi-Source Investigation (Code, Data, Sim)*
    -   *Phase 4: Discrepancy Resolution*
    -   *Phase 5: Infrastructure Creation (Self-Editing)*
    -   *Phase 6: Synthesis & Communication*
    -   *Phase 7: Meta-Reflection*
-   **Agent's Internal Monologue**: A real-time, high-level event stream/log that shows the agent's "thoughts" and key decisions, directly sourced from the `EventBus`.
    -   *Example Log: "Hypothesis formulated: Mutation operator lacks diversity."*
    -   *Example Log: "Action: Editing `trading_app/dsl/mutator.py` to add new operator."*
-   **Live Call Graph**: The real-time execution tree from the SICA framework, showing the sub-agents and tools the agent is currently using (e.g., `CoderAgent` -> `edit_file`).
-   **Live Code Diff**: A panel that displays the `git diff` of any code changes the agent has just made, providing immediate insight into its self-editing process.

### View 2: Evolution Monitoring (The "World")

This view shows the results of the agent's work on the trading application. It will absorb and enhance the features currently in `trading_web/`.

**Key Components**:

-   **Interactive Cell Table**: A filterable, searchable table of all cells in the `cells.db`, showing status ("🟢 alive" / "💀 dead"), fitness, generation, and source (LLM or Evolution).
-   **D3.js Lineage Tree**: The interactive visualization of a cell's ancestry, showing how strategies have evolved.
-   **Live Fitness Chart**: A chart showing the best and average fitness over generations, updating in real-time as the agent's evolution process runs.
-   **LLM Analysis & Statistics**: Panels to display the pattern taxonomy discovered by the LLM and aggregate statistics on its performance.

---

## 3. A Day in the Life (Dashboard View)

This narrative illustrates how a user would experience the dashboard, based on the agent's lifecycle from `VISION.md`:

1.  **Initiation**: The user clicks "Start Autonomous Loop". The **Orchestration Phase Tracker** shows **Phase 1: Context Reconstruction** as the agent scans its environment and recent logs.
2.  **Monitoring**: The **Internal Monologue** shows: *"Starting new evolution run."* The **Live Call Graph** visualizes the agent calling `execute_command`. As the process runs, the **Evolution Monitoring** view comes alive with new cells and fitness data.
3.  **Hypothesis**: After the run, the agent's analysis begins. The **Phase Tracker** updates to **Phase 2: Hypothesis Formation**. The **Internal Monologue** displays: *"Analysis complete. Hypothesis: Mutation operator is stale."*
4.  **Self-Editing**: The agent decides to act. The **Phase Tracker** moves to **Phase 5: Infrastructure Creation**. The **Call Graph** shows the `CoderAgent` being invoked. A few moments later, a diff appears in the **Live Code Diff** panel, showing the changes to `mutator.py`.
5.  **Validation**: The agent starts a new validation run. The **Phase Tracker** might cycle back through **Phase 3 (Investigation)** and **Phase 4 (Discrepancy Resolution)** as it compares the new results to the old, allowing the user to see the immediate impact of the agent's self-improvement.

---

## 4. Architectural Goal & Path Forward

The primary challenge, as noted in the original progress report, is backend integration. The vision requires a unified backend that can serve the UI while accessing data from two different sources.

**The Path Forward**:

1.  **Single Backend**: Solidify `sica_api.py` as the single source for the dashboard. It will be responsible for serving the `sica_web/` frontend.
2.  **Unify Data Sources**: The API must be able to:
    -   Query the `cells.db` directly for all evolution-related data.
    -   Communicate with the running SICA agent framework to get real-time `EventBus` and `CallGraphManager` data. The agent will need to emit new, high-level events (e.g., `ORCHESTRATION_PHASE_CHANGED`) for the dashboard to consume. (This could be achieved via a shared state layer like Redis or a dedicated internal API).
3.  **Merge Frontends**: Incrementally migrate the feature-complete components (lineage tree, cell table) from `trading_web/` into the `sica_web/` dashboard, connecting them to the unified `sica_api.py`.

This plan provides a clear vision for a powerful, integrated dashboard and a strategy to overcome the architectural hurdles to build it.
