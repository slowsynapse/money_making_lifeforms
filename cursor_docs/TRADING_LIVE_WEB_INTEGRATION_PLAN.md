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
Integrate existing `trading-learn` functionality into the web dashboard, allowing users to:
1. **Allow LLM to create new cells directly from the web interface** (uses existing `trading-learn` API)
2. **Properly mark unviable cells as "💀 dead" instead of all being "online"**
3. Monitor LLM learning sessions in real-time
4. View which cells were created by LLM vs evolution
5. Track LLM hypotheses and their success rates

**Note**: The `trading-learn` command already exists and has API implementation. We just need to expose it in the web UI.

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

### 0.2 Integrate Existing `trading-learn` Command

**Current Implementation**: The `trading-learn` command already exists in `base_agent/src/commands/trading_learn.py`
- Takes `-n` (number of strategies) and `-c` (confidence threshold)
- Uses LLM to analyze top cells and create new strategies
- Already stores cells with `llm_name`, `llm_hypothesis`, `llm_category`

**What We Need**: Web UI to trigger and monitor `trading-learn` sessions

**Check Existing API Endpoints** in `trading_api.py`:
```bash
# Need to verify if these exist or add them:
POST /llm/learn/start - Start a trading-learn session
GET /llm/learn/status - Get current learning status
GET /llm/learn/sessions - Get learning session history
GET /cells?has_llm=true - Filter cells created by LLM
```

**If endpoints don't exist, add**:
```python
@app.post("/llm/learn/start")
async def start_llm_learning(
    num_strategies: int = 5,
    confidence: float = 1.0,
    use_local: bool = False
):
    """Start a trading-learn session (wrapper around existing command)"""
    # Execute: trading-learn -n {num_strategies} -c {confidence}
    # Return session_id for monitoring

@app.get("/llm/learn/status/{session_id}")
async def get_learning_status(session_id: str):
    """Get status of running learning session"""
    # Check if process is running, how many cells created, etc.

@app.get("/cells")
async def get_cells(
    limit: int = 100,
    has_llm: Optional[bool] = None,
    llm_category: Optional[str] = None
):
    """Get cells with optional LLM filtering"""
```

**UI Components**:

1. **"LLM Learn" Button** in header
   - Opens modal to configure learning session
   - Inputs: num_strategies (5), confidence (1.0), use_local LLM toggle
   - "Start Learning" button triggers session

2. **Learning Session Monitor Panel** (appears when session active)
   - Shows: "Learning... 3/5 strategies created"
   - Progress bar
   - List of newly created cells as they appear
   - "Stop Learning" button

3. **LLM Filter Toggle** in cell table
   - "Show All" / "LLM Only" / "Evolution Only"
   - Adds icon/badge to LLM-created cells (e.g., 🤖)

4. **LLM Cell Details** in cell details panel
   - Show LLM name, hypothesis, category
   - "This cell was created by LLM analyzing Cell #29"

---

## Future Enhancement: Live Trading Feature

**Note**: Live trading with real-time sessions is a separate feature from `trading-learn`.
This would require additional implementation beyond the LLM learning integration.

See separate plan document if/when live trading feature is needed.

---

## Implementation Order

### Sprint 1: Fix Cell Status (Priority 1)
1. **Update cell status to use emojis**
   - Implement `is_cell_viable()` function
   - Update API to return "🟢 alive" or "💀 dead"
   - Update frontend display with proper styling
2. **Test with current 100 cells**
   - Mark cells with fitness < threshold as dead
   - Verify UI displays correctly

### Sprint 2: Add `trading-learn` API Endpoints (Priority 2)
1. **Check existing trading_api.py for any LLM endpoints**
2. **Add missing endpoints**:
   - `POST /llm/learn/start`
   - `GET /llm/learn/status/{session_id}`
   - `GET /cells?has_llm=true` (filter parameter)
3. **Test endpoints**:
   - Start a learning session via API
   - Monitor status
   - Verify new cells appear in database

### Sprint 3: LLM Learning UI (Priority 3)
1. **Add "LLM Learn" button** to dashboard header
2. **Create learning configuration modal**
   - Num strategies input
   - Confidence threshold
   - Use local LLM toggle
3. **Add learning session monitor**
   - Progress bar
   - Real-time cell creation updates
4. **Add LLM filter toggle** to cell table
   - Show all / LLM only / Evolution only
   - Add 🤖 badge to LLM cells

### Sprint 4: Enhanced LLM Cell Display (Priority 4)
1. **Update cell details panel**
   - Show LLM metadata (name, hypothesis, category)
   - Show parent cell if applicable
   - Highlight LLM cells in lineage tree
2. **Add LLM statistics panel**
   - Total LLM cells created
   - Success rate by category
   - Best performing LLM strategies

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
