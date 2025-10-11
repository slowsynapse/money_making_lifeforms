# Trading Evolution Web Dashboard

This document describes the real-time, interactive web dashboard for monitoring and analyzing the cell-based trading strategy evolution. The dashboard is a purpose-built interface for understanding the complex dynamics of the evolutionary process.

**Note**: This dashboard, located in `trading_web/`, replaces the generic agent visualization that was inherited from the original SICA framework.

## Architecture

The dashboard is a modern, decoupled web application:

-   **Backend (FastAPI)**: A dedicated Python server (`trading_api.py`) that runs on port `8081`. It serves the frontend and provides a rich REST API to query the `cells.db` database.
-   **Frontend (HTML/JS/D3.js)**: A single-page application located in `trading_web/`. It uses TailwindCSS for styling, Chart.js for graphing, and D3.js for advanced visualizations.

```
trading_web/
├── dashboard.js          # Main application logic, data fetching
├── index.html            # UI structure and layout
└── lineage-viz.js        # D3.js component for lineage graphs
```

## How to Use

1.  **Start the API Server**:
    ```bash
    python trading_api.py
    ```
2.  **Open the Dashboard**:
    Navigate to **http://localhost:8081** in your browser.

The dashboard will automatically connect to the API and display the latest data from the most recent evolution run.

## Current Features

The dashboard provides a rich, multi-faceted view into the evolutionary process.

### 1. Main Dashboard View

A high-level overview of the cell population, including:
-   **Header Stats**: Real-time count of total cells, best fitness, and average fitness.
-   **Top Cells Table**: A sortable and filterable list of the most successful cells. You can filter to see all cells, only LLM-generated cells, or only randomly evolved cells.
-   **Fitness Evolution Chart**: An interactive Chart.js line graph showing the progression of best and average fitness across generations.

### 2. Detailed Cell Inspector

Clicking on any cell in the main table opens a detailed inspector panel with deep insights:
-   **Strategy Genome**: The full, readable DSL of the cell's strategy.
-   **Multi-Timeframe Performance**: A breakdown of the strategy's performance (profit, Sharpe ratio, trades) across different timeframes (e.g., 1H, 4H, 1D).
-   **LLM Analysis**: If an LLM has analyzed the cell, this section displays the human-readable name, category, and hypothesis for the discovered pattern.

### 3. Interactive Lineage & Genome Diff

This is one of the most powerful features. When a cell is selected, two new panels appear:
-   **Evolutionary Lineage Tree**: A stunning, animated D3.js visualization that shows the cell's direct ancestry. Each node represents an ancestor, showing its fitness and how it connects to the next generation. This allows you to trace the entire history of a successful strategy.
-   **Genome Evolution**: A precise, color-coded "diff" view that shows the exact mutations that occurred between each generation in the selected lineage, making it easy to pinpoint the specific changes that led to a breakthrough.

### 4. Live `trading-learn` Terminal

The dashboard includes an integrated terminal that streams the live output from the `trading-cli.py learn` process. This allows you to monitor the LLM's analysis and mutation process in real-time without leaving the browser.

---

## Future Features: The Global Analysis Dashboard

While the current interface is excellent for inspecting individual cells, the next evolution is to build a **Global Analysis Dashboard** that provides a strategic, high-level overview of the entire evolutionary run, turning the insightful mockups from the old documentation into a real, interactive feature.

This new dashboard tab would include:

### 1. Run-Level Statistics

-   **Mutation Success Analysis**: A visualization showing the ratio of successful vs. failed mutations, with a breakdown of failure reasons (e.g., worse fitness, syntax errors). This is critical for tuning the mutator.
-   **Survival Analysis**: Key metrics on the population dynamics, such as the number of cells currently "online" vs. "deprecated," the average cell lifespan, and the fitness level at which cells are typically superseded by their offspring.

### 2. Interactive Pattern Taxonomy Explorer

-   A new UI component that allows you to browse all the patterns discovered by the LLM, grouped by category (e.g., "Volume Analysis," "Mean Reversion").
-   Clicking a pattern would show its average fitness, the number of cells that use it, and a list of those cells, effectively creating a library of proven, named strategies.

### 3. Cost & ROI Analysis

-   A dedicated view for `learn` mode that tracks total LLM API costs, token usage, and calculates the **Return on Investment (ROI)** of the Intelligence Engine.
-   It would answer key questions like: "What is the average cost per successful LLM-guided mutation?" and "How long until a strategy's profits pay for the cost of its discovery?"

### 4. Lineage of Best Cell

-   A prominent section dedicated to visualizing the full lineage of the single best cell from the entire run, annotated with the key mutations and conceptual breakthroughs at each step (e.g., "Arithmetic expression added," "Compound condition added").
