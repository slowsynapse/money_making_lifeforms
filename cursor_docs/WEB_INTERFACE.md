# Web Interface for Agent Visualization

## Overview

The agent includes a **real-time web visualization** that displays:
- **Execution tree** of agent/subagent calls
- **Event stream** of all agent activities
- **Metrics** (duration, tokens, costs)

## Where is the Code?

### Backend (FastAPI Server)
**Location**: `base_agent/src/web_server/server.py`

Key endpoints:
- `GET /` - Main HTML interface
- `GET /api/callgraph` - JSON data for visualization
- `WebSocket /ws` - Real-time event streaming

### Frontend (HTML/JS)
**Location**: `base_agent/src/web_server/`

```
web_server/
├── templates/
│   └── index.html          # Main page template
├── static/
│   ├── styles.css          # Tailwind-based styling
│   ├── visualizer.js       # Main entry point
│   ├── core.js             # Web component base
│   ├── store.js            # State management
│   ├── components/
│   │   ├── execution-tree.js      # Tree visualization
│   │   ├── event-stream.js        # Event log display
│   │   └── metrics-display.js     # Header metrics
│   └── utils/
│       ├── formatters.js   # Display formatters
│       └── event-utils.js  # Event processing
```

## How to Use with Trading

### 1. Run with Web Interface Enabled

```bash
# Inside Docker container:
docker run --rm \
  -p 8080:8080 \
  -v "$(pwd)/base_agent":/home/agent/agent_code \
  -v "$(pwd)/benchmark_data":/home/agent/benchmark_data \
  --env-file .env \
  sica_sandbox \
  python -m agent_code.agent trading-learn \
    --iterations 5 \
    --server  # This enables the web interface!
```

### 2. Open Your Browser

Navigate to: **http://localhost:8080**

You'll see:
- **Header**: Real-time metrics (duration, tokens, cost)
- **Execution Tree**: Hierarchical view of agent calls
- **Event Stream**: Live log of all activities

### 3. What You'll See During Trading Learning

The web interface displays the **internal agent execution**:

```
Execution Tree:
├─ Agent (Root)
   ├─ DslInterpreter.parse()
   ├─ DslInterpreter.execute()
   ├─ Backtest simulation
   └─ Result scoring

Event Stream:
[12:34:56] Agent started
[12:34:57] Tool: read_file (ohlcv.csv)
[12:34:58] LLM: Generated DSL strategy
[12:34:59] Tool: write_file (answer.txt)
[12:35:00] Backtest complete: $12.34 profit
[12:35:01] Agent completed (1234 tokens, $0.0165)
```

**Important**: The web interface shows the **agent's internal execution**, NOT the multi-iteration learning loop. Each iteration runs one agent execution.

### 4. Trading State Display

The **trading state** is shown in the **terminal output**, not the web interface:

```
💰 AGENT STATE AFTER ITERATION 1:
   Current Balance: $87.45
   Net Result: -$12.55
   Total Attempts: 1
```

The web interface shows:
- Token usage per attempt
- LLM cost per attempt
- Event timeline per attempt

## Cell Visualization (Future Enhancement)

The web interface could be extended to show:

### Cell Lineage Graph

```javascript
// Future feature: Interactive lineage tree
<div id="lineage-graph">
  Cell #1 (Gen 0) ─┬─ Cell #5 (Gen 12) ─── Cell #15 (Gen 34) ─── Cell #47 (Gen 89)
                    │
                    └─ Cell #8 (Gen 15) [extinct]
</div>
```

### Pattern Taxonomy View

```
Volume Analysis (12 cells):
  • Volume Spike Reversal - $18.45 avg
  • Contrarian Volume Fade - $8.23 avg

Mean Reversion (8 cells):
  • Overbought Correction - $15.32 avg
```

### Cell Details Panel

When clicking a cell in the execution tree:
```
Cell #47
Generation: 89
Fitness: $23.31
Parent: Cell #15
Status: online

Genome:
IF (EPSILON(0) / EPSILON(20)) > 1.5 AND DELTA(0) < DELTA(10) THEN BUY ELSE HOLD

LLM Analysis:
Name: Volume Spike Reversal
Category: Volume Analysis
Confidence: 0.85
Hypothesis: Detects institutional accumulation during local dips...

Phenotypes:
  1H: 45 trades, 62.2% win rate, Sharpe 1.47
  4H: 12 trades, 58.3% win rate, Sharpe 1.12
  1D: 3 trades, 66.7% win rate, Sharpe 0.89
```

### Implementation Notes

To add cell visualization to the web interface:

1. **Backend** (`server.py`):
```python
@app.get("/api/cells")
async def get_cells():
    repo = CellRepository(db_path)
    cells = repo.get_top_cells(limit=100)
    return [cell.to_dict() for cell in cells]

@app.get("/api/cells/{cell_id}/lineage")
async def get_cell_lineage(cell_id: int):
    repo = CellRepository(db_path)
    lineage = repo.get_lineage(cell_id)
    return [cell.to_dict() for cell in lineage]
```

2. **Frontend** (`static/components/cell-viewer.js`):
```javascript
class CellViewer extends BaseComponent {
    async loadCells() {
        const response = await fetch('/api/cells');
        this.cells = await response.json();
        this.render();
    }

    renderCell(cell) {
        return `
            <div class="cell" data-cell-id="${cell.cell_id}">
                <div class="cell-id">Cell #${cell.cell_id}</div>
                <div class="cell-fitness">$${cell.fitness.toFixed(2)}</div>
                <div class="cell-name">${cell.llm_name || 'Unnamed'}</div>
                <div class="cell-genome">${cell.dsl_genome}</div>
            </div>
        `;
    }
}
```

See `CELL_ARCHITECTURE.md` and `CELL_STORAGE_API.md` for data structures.

## Customizing the Display

### Add Trading Metrics to Web Interface

To show trading-specific data in the web UI, modify:

**1. Backend** (`server.py`):
```python
class NodeData(BaseModel):
    # ... existing fields ...
    trading_profit: Optional[float] = None
    agent_balance: Optional[float] = None
```

**2. Frontend** (`metrics-display.js`):
```javascript
render() {
  // Add trading metrics to header
  this.container.innerHTML = `
    <div class="metric">
      <span class="label">Agent Balance</span>
      <span class="value">$${data.agent_balance}</span>
    </div>
    // ... existing metrics ...
  `;
}
```

## Architecture

### Real-Time Communication

```
Agent Execution
    ↓
EventBus (publishes events)
    ↓
WebSocket (broadcasts to clients)
    ↓
Browser (updates UI in real-time)
```

### State Management

The web interface uses a **reactive store pattern**:

```javascript
// store.js
class Store {
  setState(key, value) {
    this.state[key] = value;
    // Notify all components
    document.dispatchEvent(new CustomEvent('state-change', {
      detail: { property: key, value }
    }));
  }
}
```

Components listen for `state-change` events and re-render automatically.

## Running Without Web Interface

If you don't need visualization:

```bash
python -m agent_code.agent trading-learn --iterations 5
# No --server flag = no web interface
```

This saves a small amount of resources.

## Troubleshooting

### Port Already in Use
```bash
# Error: Address already in use
# Solution: Kill existing process or change port
```

The server runs on port `8080` by default (hardcoded in `server.py`).

### WebSocket Connection Failed
```bash
# In browser console:
# WebSocket connection to 'ws://localhost:8080/ws' failed
```

**Cause**: Server not started or Docker port not mapped.

**Fix**: Ensure `-p 8080:8080` in docker run command.

### No Data Displayed
```bash
# Web interface loads but shows no execution tree
```

**Cause**: Agent hasn't started execution yet, or completed too quickly.

**Fix**: The interface populates as the agent runs. For trading-learn, each iteration is a separate agent execution.

## Summary

| Feature | Location | Purpose |
|---------|----------|---------|
| **Web Server** | `server.py` | FastAPI backend, WebSocket |
| **Main Page** | `templates/index.html` | UI structure |
| **Components** | `static/components/` | Reusable UI widgets |
| **Styles** | `static/styles.css` | Tailwind CSS styling |
| **State** | `static/store.js` | Reactive state management |

The web interface is **already built and working** - just add `--server` flag to any agent mode!
