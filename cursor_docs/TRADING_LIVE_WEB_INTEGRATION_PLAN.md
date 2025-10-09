# Trading Live Web Interface Integration Plan

## Current Status (as of 2025-10-10)

### Cell Data Analysis
- **Total cells in database**: 100 cells (generations 0-100)
- **LLM involvement**: ✗ None of the cells have been worked on by LLM yet
  - All cells have `llm_name = null`
  - No LLM-generated hypotheses or categories
  - These cells were generated through pure evolutionary mutation
- **Cell Status Issue**: All cells currently marked as "online" - need to mark unviable cells as "dead"
  - Should use: "🟢 alive" for viable cells
  - Should use: "💀 dead" for unviable/failed cells

### Recent Commits
- **Latest commit (1fa52aa)**: "Add D3.js lineage visualization and responsive dashboard layout"
  - Added interactive D3.js evolutionary lineage tree
  - Implemented genome evolution diff viewer
  - Restructured dashboard layout (lineage top, cell table left, details/genome/fitness right)
  - Responsive height calculation for lineage visualization

## Objective
Add real-time trading capabilities to the web dashboard, allowing users to:
1. Select top-performing cells and deploy them to live trading
2. Monitor live trading performance in real-time
3. View live P&L, position data, and trade history
4. Compare live performance vs backtest results
5. **Allow LLM to create new cells directly from the web interface**
6. **Properly mark unviable cells as "💀 dead" instead of all being "online"**

---

## Phase 0: Prerequisites (Cell Status & LLM Creation)

### 0.1 Fix Cell Status Display

**Current Issue**: All cells showing as "online" regardless of viability

**Changes Needed**:

1. **Update Cell Status Logic** (`base_agent/src/storage/models.py`):
```python
# Change status field to use emoji constants
CELL_STATUS_ALIVE = "🟢 alive"
CELL_STATUS_DEAD = "💀 dead"
```

2. **Add Cell Viability Checker**:
```python
def is_cell_viable(cell):
    """Determine if cell should be marked as alive or dead"""
    # Dead if: fitness < threshold, no trades, syntax errors, etc.
    if cell.fitness < -10.0:
        return False
    if cell.total_trades == 0:
        return False
    # Add more criteria as needed
    return True
```

3. **Update API Response** (`trading_api.py`):
```python
# When returning cells, set proper status
for cell in cells:
    cell.status = CELL_STATUS_ALIVE if is_cell_viable(cell) else CELL_STATUS_DEAD
```

4. **Update Frontend Display** (`trading_web/dashboard.js`):
```javascript
// Render status with emoji
<span class="inline-flex px-2 py-1 text-xs font-medium rounded-full ${
    cell.status === '🟢 alive'
        ? 'bg-green-100 text-green-800'
        : 'bg-red-100 text-red-800'
}">
    ${cell.status}
</span>
```

### 0.2 LLM Cell Creation Interface

**Goal**: Allow LLM to create new cells from web dashboard

**New API Endpoints**:

```python
@app.post("/cells/create")
async def create_cell_from_llm(
    dsl_genome: str,
    llm_name: str,
    llm_hypothesis: str,
    llm_category: str,
    parent_cell_id: Optional[int] = None
):
    """Create a new cell with LLM-generated strategy"""
    # 1. Validate DSL syntax
    # 2. Run backtest to get fitness
    # 3. Create cell in database with LLM metadata
    # 4. Return new cell_id and results

@app.post("/cells/{cell_id}/fork")
async def fork_cell(
    cell_id: int,
    mutation_description: str,
    llm_name: str
):
    """Fork existing cell with LLM-guided mutation"""
    # 1. Get parent cell
    # 2. Apply mutation
    # 3. Backtest mutated strategy
    # 4. Create new cell with lineage
```

**UI Components**:

1. **"Create Strategy" Button** in header
2. **Strategy Editor Modal** with:
   - DSL textarea with syntax highlighting
   - LLM hypothesis input
   - Category selector (momentum, mean-reversion, etc.)
   - "Test Strategy" button (runs backtest)
   - "Create Cell" button

3. **"Fork with LLM" Button** on cell details panel

**Example UI Flow**:
```
User clicks "Create Strategy"
→ Modal opens with DSL editor
→ User writes DSL or asks LLM for help
→ Click "Test Strategy" → Shows backtest results
→ If satisfied, click "Create Cell" → Cell added to database
→ Dashboard refreshes showing new cell
```

---

## Phase 1: Backend API Extensions

### 1.1 New API Endpoints (trading_api.py)

Add these endpoints to `trading_api.py`:

```python
@app.post("/trading/start")
async def start_live_trading(
    cell_id: int,
    allocation: float = 1000.0,  # USD allocation
    timeframe: str = "1h",       # Trading timeframe
    dry_run: bool = True         # Paper trading vs live
):
    """Start live trading for a specific cell"""
    # 1. Load cell from repository
    # 2. Initialize trading session
    # 3. Start monitoring thread/task
    # 4. Return session ID and initial status

@app.post("/trading/stop/{session_id}")
async def stop_live_trading(session_id: str):
    """Stop an active trading session"""

@app.get("/trading/sessions")
async def get_trading_sessions():
    """Get all active and recent trading sessions"""

@app.get("/trading/session/{session_id}")
async def get_session_details(session_id: str):
    """Get detailed info for a specific session"""
    # Returns: session_id, cell_id, status, start_time,
    #          current_pnl, open_positions, trade_count

@app.get("/trading/session/{session_id}/trades")
async def get_session_trades(session_id: str):
    """Get trade history for a session"""

@app.get("/trading/session/{session_id}/positions")
async def get_session_positions(session_id: str):
    """Get current open positions for a session"""

@app.websocket("/ws/trading/{session_id}")
async def trading_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time updates"""
    # Stream: price updates, P&L changes, new trades, position updates
```

### 1.2 Trading Session Manager

Create `base_agent/src/trading/live_session_manager.py`:

```python
class LiveTradingSession:
    """Manages a single live trading session"""
    def __init__(self, cell_id, dsl_genome, allocation, timeframe, dry_run):
        self.session_id = str(uuid.uuid4())
        self.cell_id = cell_id
        self.genome = dsl_genome
        self.allocation = allocation
        self.timeframe = timeframe
        self.dry_run = dry_run
        self.status = "initializing"  # initializing, running, stopped, error
        self.start_time = datetime.now()
        self.pnl = 0.0
        self.trades = []
        self.positions = []

    async def start(self):
        """Initialize connection and start trading loop"""

    async def stop(self):
        """Gracefully stop trading and close positions"""

    async def update_tick(self, price_data):
        """Process new price tick"""

class LiveTradingManager:
    """Manages all active trading sessions"""
    def __init__(self):
        self.sessions: Dict[str, LiveTradingSession] = {}

    async def create_session(self, cell_id, genome, allocation, timeframe, dry_run):
        """Create and start new trading session"""

    async def stop_session(self, session_id):
        """Stop a specific session"""

    def get_active_sessions(self):
        """Get all active sessions"""
```

---

## Phase 2: Frontend Dashboard Extensions

### 2.1 New UI Components

#### 2.1.1 Live Trading Panel (Right Column Addition)
Add below Fitness Evolution chart in `trading_web/index.html`:

```html
<!-- Live Trading Control -->
<div id="live-trading-panel" class="bg-white rounded-lg shadow mt-6">
    <div class="px-6 py-4 border-b border-gray-200">
        <h2 class="text-lg font-semibold text-gray-900">Live Trading</h2>
        <p class="text-sm text-gray-600">Deploy cells to live trading</p>
    </div>
    <div class="p-6">
        <div id="live-trading-content">
            <!-- Trading controls will be inserted here -->
        </div>
    </div>
</div>
```

#### 2.1.2 Active Sessions Table (New Section Above Main Grid)
Add below lineage section:

```html
<!-- Active Trading Sessions -->
<div id="active-sessions-section" class="mb-6 bg-white rounded-lg shadow" style="display: none;">
    <div class="px-6 py-4 border-b border-gray-200">
        <h2 class="text-lg font-semibold text-gray-900">Active Trading Sessions</h2>
        <p class="text-sm text-gray-600">Real-time monitoring of deployed cells</p>
    </div>
    <div class="p-6">
        <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
                <tr>
                    <th>Session ID</th>
                    <th>Cell</th>
                    <th>Timeframe</th>
                    <th>Status</th>
                    <th>P&L</th>
                    <th>Trades</th>
                    <th>Uptime</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody id="sessions-table-body">
                <!-- Sessions will be rendered here -->
            </tbody>
        </table>
    </div>
</div>
```

### 2.2 New JavaScript Files

#### 2.2.1 trading_web/live-trading.js
```javascript
// Live Trading Controls and Management

class LiveTradingManager {
    constructor() {
        this.activeSessions = new Map();
        this.websockets = new Map();
    }

    async startTrading(cellId, allocation, timeframe, dryRun) {
        // POST to /trading/start
        // Open WebSocket connection
        // Update UI
    }

    async stopTrading(sessionId) {
        // POST to /trading/stop/{session_id}
        // Close WebSocket
        // Update UI
    }

    connectWebSocket(sessionId) {
        // Connect to /ws/trading/{session_id}
        // Handle real-time updates
    }

    renderSessionCard(session) {
        // Render live session card with P&L, status, controls
    }
}
```

#### 2.2.2 trading_web/realtime-charts.js
```javascript
// Real-time chart updates for live trading

class RealtimePnLChart {
    constructor(canvasId) {
        this.chart = new Chart(canvasId, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'P&L',
                    data: [],
                    borderColor: 'rgb(34, 197, 94)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                animation: false,  // Disable for real-time
                scales: {
                    x: { type: 'time' }
                }
            }
        });
    }

    addDataPoint(timestamp, pnl) {
        this.chart.data.labels.push(timestamp);
        this.chart.data.datasets[0].data.push(pnl);
        this.chart.update('none');  // No animation
    }
}
```

---

## Phase 3: Integration with Existing trading-live Command

### 3.1 Adapt Existing Code
The `trading-live` command already exists in `base_agent/src/commands/trading_live.py`:
- Uses Kraken API for live data
- Implements DSL execution
- Handles position management

**Modifications needed**:
1. Extract core trading logic into reusable class
2. Add session tracking and state management
3. Add WebSocket broadcast for UI updates
4. Store trade history in database

### 3.2 Database Schema Extensions

Add to `base_agent/src/storage/models.py`:

```python
class TradingSession(Base):
    __tablename__ = 'trading_sessions'

    session_id = Column(String, primary_key=True)
    cell_id = Column(Integer, ForeignKey('cells.cell_id'))
    allocation = Column(Float)
    timeframe = Column(String)
    dry_run = Column(Boolean)
    status = Column(String)  # running, stopped, error
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True)
    final_pnl = Column(Float, nullable=True)
    trade_count = Column(Integer, default=0)

class LiveTrade(Base):
    __tablename__ = 'live_trades'

    trade_id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('trading_sessions.session_id'))
    cell_id = Column(Integer, ForeignKey('cells.cell_id'))
    timestamp = Column(DateTime)
    side = Column(String)  # buy/sell
    quantity = Column(Float)
    price = Column(Float)
    pnl = Column(Float)
    position_after = Column(Float)
```

---

## Phase 4: Testing Plan

### 4.1 Unit Tests
- Test session creation/stopping
- Test WebSocket connections
- Test trade logging
- Test P&L calculations

### 4.2 Integration Tests
- Start session via API
- Verify WebSocket receives updates
- Stop session and verify cleanup
- Test multiple concurrent sessions

### 4.3 Manual Testing
1. Deploy Cell #29 (best performer) to paper trading
2. Monitor for 1 hour
3. Verify UI updates correctly
4. Stop session and verify final P&L saved

---

## Phase 5: Implementation Order

### Sprint 0: Prerequisites (MUST DO FIRST)
1. **Fix cell status display**
   - Update status field to use "🟢 alive" / "💀 dead"
   - Implement `is_cell_viable()` checker
   - Update API to set proper status
   - Update frontend to display emoji status
2. **Add LLM cell creation endpoints**
   - POST `/cells/create` for new cells
   - POST `/cells/{cell_id}/fork` for mutations
   - Add backtest runner integration
3. **Add Strategy Editor Modal UI**
   - Create button in header
   - Modal with DSL editor
   - Test & create workflow

### Sprint 1: Backend Foundation
1. Create LiveTradingSession and LiveTradingManager classes
2. Add database models for sessions and trades
3. Implement basic API endpoints (start/stop/list)
4. Add session storage to database

### Sprint 2: Real-time Communication
1. Implement WebSocket endpoint
2. Add real-time price feed integration
3. Broadcast trade events to WebSocket
4. Test WebSocket with multiple clients

### Sprint 3: Frontend UI
1. Create live-trading.js with session management
2. Add Live Trading panel to dashboard
3. Add Active Sessions table
4. Implement start/stop controls

### Sprint 4: Real-time Charts
1. Create realtime-charts.js
2. Add P&L chart component
3. Add position chart component
4. Connect charts to WebSocket updates

### Sprint 5: Polish & Testing
1. Add error handling and recovery
2. Add session persistence (survive server restart)
3. Add trading session history view
4. Performance testing with multiple sessions

---

## Technical Considerations

### Security
- Require authentication for trading endpoints (future)
- Validate cell_id exists before starting session
- Limit concurrent sessions per user
- Add trading permissions system

### Performance
- Use async/await for all I/O operations
- Limit WebSocket broadcast rate (max 1 update/sec)
- Add session cleanup for stale connections
- Implement connection pooling for Kraken API

### Data Management
- Store all trades in database for audit trail
- Implement session recovery after server restart
- Add session archival (move old sessions to archive table)
- Keep last 100 sessions in main table

### Monitoring
- Add logging for all trading operations
- Track API rate limits (Kraken has limits)
- Monitor WebSocket connection health
- Alert on session errors

---

## Files to Modify/Create

### New Files
- `base_agent/src/trading/live_session_manager.py`
- `base_agent/src/storage/models.py` (extend with new tables)
- `trading_web/live-trading.js`
- `trading_web/realtime-charts.js`

### Modified Files
- `trading_api.py` (add new endpoints + WebSocket)
- `trading_web/index.html` (add UI sections)
- `trading_web/dashboard.js` (integrate live trading controls)
- `base_agent/src/storage/cell_repository.py` (add session queries)

---

## Success Criteria

1. ✓ User can click "Start Live Trading" on any cell
2. ✓ Session appears in Active Sessions table
3. ✓ Real-time P&L updates visible
4. ✓ Trade history populates as trades occur
5. ✓ User can stop session and see final results
6. ✓ Session data persists in database
7. ✓ Multiple concurrent sessions work correctly
8. ✓ WebSocket reconnects on disconnect

---

## Future Enhancements (Post-MVP)

- Add risk management controls (max drawdown, stop loss)
- Add trading schedule (only trade during certain hours)
- Add multi-asset support (beyond XBTUSD)
- Add performance comparison (live vs backtest)
- Add LLM-powered strategy suggestions based on live performance
- Add automated strategy rotation (stop losers, boost winners)
- Add social features (share successful strategies)

---

## Notes for Future Claude Code Sessions

When resuming this work:
1. Check `git log` for latest commits
2. Verify API server is running: `./trade-api`
3. Check which background tasks are running: `/bashes`
4. Review this plan and pick up at current sprint
5. **Current cells have NO LLM involvement** - pure evolutionary strategies
6. Dashboard is at http://localhost:8081
7. Use Puppeteer MCP to test UI changes visually
