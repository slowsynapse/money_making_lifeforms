# Trading Dashboard - Final Polish Plan

## Objective
Finalize the `trading-learn` integration in the web dashboard by implementing a few remaining backend refinements and UI features. The majority of the frontend and backend work is already complete.

---

## Remaining Tasks

### Task 1: Refine Cell Status Logic (Backend)

**Status**: Frontend is implemented, backend needs improvement.

**Current Behavior**: The frontend in `dashboard.js` determines if a cell is "alive" or "dead" based on a simple `fitness >= 0` check.

**Required Change**:
-   In `trading_api.py`, enhance the `/cells/top/{limit}` endpoint. Before returning the cell data, implement a server-side `is_cell_viable()` function that sets a definitive status (e.g., "🟢 alive" or "💀 dead"). This function should incorporate more robust logic, such as checking for `total_trades > 0` and a reasonable fitness score.

---

### Task 2: Implement LLM Statistics Panel (Frontend)

**Status**: Not yet implemented.

**Current Behavior**: The dashboard displays LLM details for individual cells but lacks an aggregate view.

**Required Change**:
-   In `index.html`, add a new section for "LLM Statistics".
-   In `trading_api.py`, create a new endpoint, `/api/stats/llm`, that calculates and returns key metrics (e.g., total LLM-created cells, success rate by category, average fitness of LLM cells).
-   In `dashboard.js`, fetch from this new endpoint and render the statistics in the new panel.

---

### Task 3: Activate the "LLM Learn" Modal

**Status**: The modal exists in HTML but the main "Run" button is hardcoded.

**Current Behavior**: The "▶ Run" button in the header is hardcoded in `dashboard.js` to start a 100-iteration `trading-learn` session and does not use the existing configuration modal.

**Required Change**:
-   In `dashboard.js`, modify the event listener for the `#run-btn`. Instead of directly calling the `/learn/run` API, it should first **display the `#llm-modal`**.
-   The "Start Learning" button (`#modal-start`) within the modal will then be responsible for reading the configuration values from the form and making the API call to start the session.

---

## Files to Modify

-   **Backend**: `trading_api.py` (Refine cell status logic, add LLM stats endpoint).
-   **Frontend**: `trading_web/index.html` (Add stats panel structure), `trading_web/dashboard.js` (Activate modal, fetch and render stats).
