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
