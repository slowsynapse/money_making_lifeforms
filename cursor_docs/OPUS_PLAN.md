# OPUS PLAN: Architectural Refactoring for Trading Evolution System

**Created by**: Claude Opus 4.1
**Date**: 2025-10-10
**Status**: PLANNING

## Executive Summary

The trading evolution system is **90% functionally complete** but suffers from architectural debt due to being grafted onto the SICA framework. This plan separates the trading system into its own clean interface while preserving all existing functionality.

---

## ✅ COMPLETED WORK (From IMPLEMENTATION_TODO.md)

### Sprint 1-3: Core Infrastructure ✅
- ✅ **Cell Storage System**: Full SQLite database with 8 tables
- ✅ **Multi-timeframe Backtesting**: 1H/4H/1D support
- ✅ **Evolution Integration**: Cell birth, lineage tracking, fitness storage
- ✅ **LLM Integration**: Pattern discovery, intelligent mutations
- ✅ **DSL V1**: Basic comparisons and indicators

### Sprint 4: DSL V2 Enhancements ✅
- ✅ **Phase 1: Arithmetic Operations** (+, -, *, /)
- ✅ **Phase 2: Aggregation Functions** (AVG, SUM, MAX, MIN, STD)
- ✅ **Phase 3: Logical Operators** (AND, OR, NOT)
- ✅ **Phase 4: Multi-Timeframe Syntax** (DELTA_1H, DELTA_4H, etc.)

### Sprint 6: Web Visualization (Partial) ✅
- ✅ **Event Integration**: 8 new event types for evolution
- ✅ **Cell Table Display**: Live updates via `/api/cells`
- ✅ **Tabbed Interface**: Callgraph + Evolution Cells tabs

### Latest Improvements ✅
- ✅ **Zero-Trade Penalty**: Prevents inactive strategies from ranking high
- ✅ **Adaptive Mutation Rates**: Temperature-based complexity control
- ✅ **Zero-Trade Filtering**: Excludes inactive cells from LLM analysis

---

## 🎯 THE CORE PROBLEM

### Current Architecture (Tangled)
```
SICA Framework (agent orchestration)
    └── Trading System (grafted on)
        ├── Evolution functions in agent.py (1500+ lines)
        ├── Events mixed with SICA events
        ├── Web UI fights callgraph paradigm
        └── No clean CLI interface
```

### Target Architecture (Separated)
```
Trading Evolution System (standalone)
    ├── trading_cli.py (clean entry point)
    ├── trading_evolution.py (core logic)
    ├── trading_api.py (web API)
    └── trading_web/ (dedicated UI)

SICA Framework (preserved for benchmarks)
    └── Remains untouched for compatibility
```

---

## 📋 IMPLEMENTATION PLAN

### Phase 1: Create Standalone Trading CLI 🆕
**Goal**: User-friendly command-line interface without Docker complexity

#### 1.1 Create `trading_cli.py` (Root Level)
```python
# Clean argparse interface with subcommands
python trading_cli.py evolve --generations 100 --fitness-goal 50
python trading_cli.py learn --iterations 10 --use-local-llm
python trading_cli.py query top-cells --limit 10
python trading_cli.py web --port 8081
```

**Features**:
- Direct Python execution (no Docker required for local)
- Clear parameter names (not cryptic flags)
- Built-in help system
- Progress bars and status updates

#### 1.2 Create `run_trading.sh` (Docker Wrapper)
```bash
#!/bin/bash
# Simplifies Docker commands
./run_trading.sh evolve --generations 100
./run_trading.sh query lineage --cell-id 47
```

**Status**: 🆕 NEW WORK

---

### Phase 2: Extract Trading Logic from agent.py 🆕

#### 2.1 Create `base_agent/src/trading/trading_evolution.py`
Move from `agent.py`:
- `run_trading_evolve()` → `TradingEvolution.evolve()`
- `run_trading_learn()` → `TradingEvolution.learn()`
- `run_trading_test()` → `TradingEvolution.test()`
- All trading-specific helper functions

#### 2.2 Create `base_agent/src/trading/trading_config.py`
Centralize configuration:
```python
DEFAULT_SYMBOL = "PURR"
DEFAULT_TIMEFRAMES = ["1H", "4H", "1D"]
DEFAULT_INITIAL_CAPITAL = 1000.0
DEFAULT_FEE_RATE = 0.00045
LENIENT_CELL_COUNT = 100
STAGNATION_LIMIT = 20
```

**Status**: 🆕 NEW WORK
**Benefit**: Reduces agent.py from 1500+ to ~500 lines

---

### Phase 3: Create Cell Query Interface 🆕

#### 3.1 Create `base_agent/src/trading/cell_queries.py`
High-level query functions:
```python
class CellQueries:
    def get_top_performers(limit=10, min_trades=1)
    def get_lineage_tree(cell_id)
    def get_pattern_taxonomy()
    def export_strategy(cell_id, format='dsl')
    def compare_cells(cell_ids)
    def get_evolution_summary(run_id)
```

#### 3.2 Integrate with CLI
```bash
python trading_cli.py query top-cells --min-trades 5
python trading_cli.py query lineage --cell-id 47 --format tree
python trading_cli.py query patterns --category "Volume Analysis"
```

**Status**: 🆕 NEW WORK

---

### Phase 4: Build Dedicated Trading Web Interface 🆕

#### 4.1 Create `trading_api.py` (Standalone FastAPI)
```python
# Separate from SICA web server
@app.get("/api/cells/top/{limit}")
@app.get("/api/cell/{cell_id}")
@app.get("/api/cell/{cell_id}/phenotypes")
@app.get("/api/evolution/runs")
@app.websocket("/ws/evolution")  # Real-time updates
```

#### 4.2 Create `trading_web/` Directory
```
trading_web/
├── index.html         # Trading-specific UI
├── js/
│   ├── cell-table.js  # Cell browser
│   ├── lineage-tree.js # D3.js ancestry viz
│   ├── fitness-chart.js # Evolution progress
│   └── pattern-browser.js # Pattern taxonomy
└── css/
    └── trading.css    # Clean, focused styling
```

**Features**:
- Direct database queries (no EventBus routing)
- WebSocket for real-time evolution updates
- Purpose-built visualizations for cells
- No callgraph paradigm interference

**Status**: 🆕 NEW WORK
**Port**: 8081 (separate from SICA's 8080)

---

### Phase 5: Add Progress Monitoring 🆕

#### 5.1 Real-time Evolution Display
```
Generation 47/100 [████████████░░░░░░] 47%
Cells Birthed: 23 | Best Fitness: $31.45 | Temperature: 0.325
Current: IF AVG(DELTA, 20) > DELTA(0) THEN BUY ELSE SELL
```

#### 5.2 LLM Analysis Progress
```
Analyzing Cells [████████░░░░░░░░░░] 30/100
Pattern Found: "Volume Spike Reversal" (5 cells)
Proposing Mutation... [EPSILON focus detected]
```

**Status**: 🆕 NEW WORK

---

### Phase 6: Documentation & Migration Guide 🆕

#### 6.1 Create `MIGRATION.md`
- How to run existing experiments with new CLI
- Mapping old Docker commands to new CLI
- Database compatibility notes

#### 6.2 Update `README.md`
- New quick start with `trading_cli.py`
- Architecture diagram showing separation
- Performance improvements achieved

**Status**: 🆕 NEW WORK

---

## 📊 COMPARISON: Current vs New

| Aspect | Current (Tangled) | New (Separated) |
|--------|------------------|-----------------|
| **Entry Point** | `docker run ... python -m agent_code.agent trading-evolve` | `python trading_cli.py evolve` |
| **Code Organization** | 1500+ lines in agent.py | Modular files < 500 lines each |
| **Web Interface** | Fighting SICA paradigm | Purpose-built for cells |
| **Database Access** | Through event system | Direct repository queries |
| **Configuration** | Scattered constants | Centralized config file |
| **Testing** | Mixed with SICA tests | Isolated trading tests |
| **Documentation** | Assumes SICA knowledge | Standalone trading docs |

---

## 🚀 IMPLEMENTATION SEQUENCE

### Week 1: CLI & Core Extraction
1. Create `trading_cli.py` with basic commands
2. Extract logic to `trading_evolution.py`
3. Create `trading_config.py`
4. Test all modes work via new CLI

### Week 2: Query Interface & Monitoring
1. Implement `cell_queries.py`
2. Add query commands to CLI
3. Add progress monitoring
4. Create Docker wrapper script

### Week 3: Dedicated Web Interface
1. Create `trading_api.py` with FastAPI
2. Build `trading_web/` UI components
3. Implement WebSocket for real-time updates
4. Test with live evolution runs

### Week 4: Polish & Documentation
1. Write migration guide
2. Update all documentation
3. Add comprehensive tests
4. Performance optimization

---

## ✅ SUCCESS METRICS

### Immediate Wins (Week 1)
- ✅ Can run evolution without complex Docker commands
- ✅ agent.py reduced to < 500 lines
- ✅ All existing functionality preserved

### Medium Term (Week 2-3)
- ✅ Query cells without writing SQL
- ✅ Web UI shows cells without callgraph confusion
- ✅ Real-time evolution monitoring

### Long Term (Week 4+)
- ✅ New users can start evolving in < 5 minutes
- ✅ Clean codebase enables rapid feature development
- ✅ Performance improvements from direct DB access

---

## 🎯 KEY BENEFITS

1. **Clean Separation**: Trading system independent of SICA
2. **User Friendly**: Simple CLI replaces Docker complexity
3. **Maintainable**: Modular code instead of monolithic files
4. **Performant**: Direct database access, no event routing
5. **Extensible**: Easy to add new features without breaking SICA

---

## 🚦 RISK MITIGATION

### Risk: Breaking existing functionality
**Mitigation**: Keep old code until new system proven, run in parallel

### Risk: Database compatibility issues
**Mitigation**: No schema changes, only access patterns change

### Risk: Web UI complexity
**Mitigation**: Start with simple table view, add visualizations incrementally

---

## 📝 NOTES

- This plan **preserves all existing work** - nothing is thrown away
- The SICA framework remains intact for benchmark compatibility
- Can be implemented incrementally without breaking current system
- Each phase delivers immediate value

---

**Ready to untangle the web and set your trading evolution free! 🚀**