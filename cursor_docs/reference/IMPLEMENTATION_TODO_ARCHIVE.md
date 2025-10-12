# Implementation TODO: Cell-Based Evolution System

**Last Updated**: 2025-10-09
**Status**: Sprints 1-3 ✅ COMPLETE | DSL Phase 1 ✅ COMPLETE | Sprints 4-6 🔄 PENDING

This document tracks implementation status for the cell-based trading evolution system.

---

## 🏛️ INHERITED INFRASTRUCTURE (From SICA Fork)

This project was **forked from the original SICA (Self-Improving Coding Agent) framework**. The following components were inherited and are fully functional:

### Core Agent System (Inherited)
- ✅ **Web Server**: FastAPI server at `base_agent/src/web_server/server.py`
  - Port 8080, WebSocket support for real-time updates
  - Callgraph visualization endpoint (`/api/callgraph`)
  - Event streaming infrastructure
- ✅ **Agent System**: Multi-agent orchestration framework
  - `base_agent/src/agents/` - Orchestrator, Main, Implement, Test, Debug agents
  - Event bus for inter-agent communication
  - Callgraph manager for execution tracking
- ✅ **LLM Infrastructure**:
  - `base_agent/src/llm/` - LLM client wrapper with Anthropic support
  - Token counting and budget tracking
  - Context management for conversations

### What's NEW for Trading Evolution
The following components were **built from scratch** for this cell-based trading evolution system:
- 🆕 **Cell-Based Storage**: SQLite database for evolutionary strategies (`base_agent/src/storage/`)
- 🆕 **Trading Modes**: `trading-evolve`, `trading-learn`, `trading-test`, `trading-demo` in `agent.py`
- 🆕 **DSL System**: Abstract trading strategy language (`base_agent/src/dsl/`)
- 🆕 **Multi-Timeframe Backtesting**: Test strategies on 1H/4H/1D simultaneously
- 🆕 **Pattern Discovery**: LLM analysis pipeline for finding profitable patterns (`base_agent/src/analysis/`)
- 🆕 **Intelligent Mutations**: LLM-guided strategy proposals based on pattern taxonomy

**Note**: Phase 6 (Web Interface Extensions) is about **extending the existing SICA web server** with cell-specific endpoints and UI components, not building a server from scratch.

---

## ✅ COMPLETED WORK

### Sprint 1: Core Infrastructure ✅ COMPLETE

#### 1.1 Database Setup ✅
- [x] Create `base_agent/src/storage/__init__.py`
- [x] Create `base_agent/src/storage/cell_repository.py`
  - [x] Implement `CellRepository` class
  - [x] Implement database initialization with schema from `DATABASE_SCHEMA.md`
  - [x] Implement all CRUD methods from `CELL_STORAGE_API.md`
  - [x] Add transaction support (context manager)
  - [x] Write unit tests for repository methods

**Completed**: 2025-10-09
**Files**: `base_agent/src/storage/cell_repository.py`

#### 1.2 Data Classes ✅
- [x] Create `base_agent/src/storage/models.py`
  - [x] Implement `Cell` dataclass
  - [x] Implement `CellPhenotype` dataclass
  - [x] Implement `DiscoveredPattern` dataclass
  - [x] Implement `EvolutionRun` dataclass
  - [x] Add `to_dict()` methods for JSON serialization
  - [x] Add validation logic

**Completed**: 2025-10-09
**Files**: `base_agent/src/storage/models.py`

#### 1.3 Multi-Timeframe Data Fetching ✅
- [x] Update `base_agent/src/data/hyperliquid_fetcher.py`
  - [x] Add method to fetch multiple timeframes at once
  - [x] Implement `fetch_multi_timeframe(symbol, timeframes=['1H', '4H', '1D'], days=30)`
  - [x] Add caching to avoid re-fetching
  - [x] Handle timestamp alignment between timeframes

**Completed**: 2025-10-09
**Files**: `base_agent/src/data/hyperliquid_fetcher.py`

### Sprint 2: Evolution Mode Integration ✅ COMPLETE

#### 2.1 Update Backtest to Support Multi-Timeframe ✅
- [x] Update `trading_benchmark.py`
  - [x] Change `_run_backtest()` signature to accept multi-timeframe data
  - [x] Test strategy on all timeframes (1H, 4H, 1D)
  - [x] Return dict of results per timeframe
  - [x] Select best timeframe for fitness

**Completed**: 2025-10-09
**Files**: `base_agent/src/benchmarks/trading_benchmarks/trading_benchmark.py`

#### 2.2 Integrate Cell Birth into Evolution ✅
- [x] Update `run_trading_evolve()` in `agent.py`
  - [x] Initialize `CellRepository` at start
  - [x] Create `evolution_runs` record
  - [x] Birth cells for successful mutations
  - [x] Record failures for unsuccessful mutations
  - [x] Update evolution_run stats (total_cells_birthed, etc.)
  - [x] Complete evolution_run at end

**Completed**: 2025-10-09
**Files**: `base_agent/agent.py`

#### 2.3 Update Mutation Selection ✅
- [x] Update mutation logic to use best cell from database
  - [x] Query `repo.get_top_cells(limit=1)` instead of tracking in-memory
  - [x] Support different parent selection modes (best, tournament, fitness-proportional)

**Completed**: 2025-10-09
**Files**: `base_agent/agent.py`

### Sprint 3: LLM Integration (Trading-Learn Mode) ✅ COMPLETE

**Design Philosophy**:
- `trading-evolve`: 100% random mutations (FREE, builds genetic library)
- `trading-learn`: 100% LLM-guided mutations (SMART, exploits patterns)
- Clear separation: cheap exploration → intelligent exploitation

#### 3.1 LLM Analysis Pipeline with Batch Processing ✅
- [x] Create `base_agent/src/analysis/__init__.py`
- [x] Create `base_agent/src/analysis/cell_analyzer.py`
  - [x] Implement `prepare_cell_context(cell_id, repo)` - fetch cell + lineage + phenotypes
  - [x] Implement `analyze_cells_in_batches(repo, cell_ids, batch_size=30)` - batch processing for 8K context
  - [x] Implement `merge_pattern_discoveries(batch_results)` - deduplicate patterns across batches
  - [x] Create LLM prompt template for batch cell analysis
  - [x] Parse JSON response from LLM (pattern taxonomy)
  - [x] Handle errors gracefully (skip failed batches, continue)

**Completed**: 2025-10-09
**Files**: `base_agent/src/analysis/cell_analyzer.py`
**Notes**: Batch size of 30 cells fits Gemma 3 27B's 8K context window

#### 3.2 Intelligent Mutation Proposals ✅
- [x] Create `base_agent/src/analysis/mutation_proposer.py`
  - [x] Implement `propose_intelligent_mutation(cell, patterns, repo)` - LLM suggests smart mutations
  - [x] Create LLM prompt template for mutation proposals
  - [x] Parse mutation proposal JSON (strategy, rationale, expected_improvement)
  - [x] Validate proposed mutations (parseable DSL, different from parent)

**Completed**: 2025-10-09
**Files**: `base_agent/src/analysis/mutation_proposer.py`

#### 3.3 Rewrite Trading-Learn Mode (100% LLM-Guided) ✅
- [x] **Completely rewrite** `run_trading_learn()` in `agent.py`
  - [x] Load cell database from prior `trading-evolve` run
  - [x] Analyze top 100 cells in batches (30 cells per batch)
  - [x] Build pattern taxonomy from batch analysis
  - [x] For each iteration:
    - [x] Select best cell (or tournament selection)
    - [x] LLM proposes intelligent mutation based on patterns
    - [x] Parse and validate proposed strategy
    - [x] Test on multi-timeframe backtest
    - [x] Birth cell if fitness improves
    - [x] Update pattern taxonomy with new insights
  - [x] Track LLM costs and display budget usage
  - [x] Store all analysis in database (patterns, cell_patterns, mutation proposals)

**Completed**: 2025-10-09
**Files**: `base_agent/agent.py`
**Breaking Change**: Old `trading-learn` used MainOrchestratorAgent (generate from scratch). New version uses cell database + intelligent mutation.

#### 3.4 Database Integration ✅
- [x] Verify `cell_mutation_proposals` table exists in schema
- [x] Implement `store_mutation_proposal()` in CellRepository
- [x] Implement `get_mutation_proposals_for_cell()` in CellRepository
- [x] Implement `store_pattern()` in CellRepository
- [x] Implement `link_cell_to_pattern()` in CellRepository

**Completed**: 2025-10-09
**Files**: `base_agent/src/storage/cell_repository.py`

### DSL V2 Phase 1: Arithmetic Operations ✅ COMPLETE

#### 4.1 Phase 1: Arithmetic Operations ✅
- [x] Update `base_agent/src/dsl/grammar.py`
  - [x] Add `BinaryOp` class for +, -, *, /
  - [x] Update grammar to support expressions
- [x] Update `base_agent/src/dsl/interpreter.py`
  - [x] Implement `evaluate_expression()` function
  - [x] Handle arithmetic precedence
  - [x] Add division-by-zero protection
- [x] Update `base_agent/src/dsl/mutator.py`
  - [x] Add arithmetic expression mutations
  - [x] Mutate operators (+  → -)
  - [x] Mutate operand order
  - [x] Add parentheses mutations
- [x] Add tests for arithmetic DSL

**Completed**: 2025-10-09
**Files**: `base_agent/src/dsl/*.py`
**Impact**: Solves "volume vs price" comparison problem, enables ratios and momentum indicators

---

## 🔄 PENDING WORK

### Sprint 4: DSL Enhancement (NEXT PRIORITY)

### 4.2 Phase 2: Aggregation Functions
- [ ] Add `FunctionCall` class to grammar
- [ ] Implement `AVG()`, `SUM()`, `MAX()`, `MIN()`, `STD()`
- [ ] Update interpreter to evaluate functions
- [ ] Update mutator to mutate function types and windows
- [ ] Add tests

**Priority**: P1
**Estimated time**: 3-4 days
**Files**: `base_agent/src/dsl/*.py`

### 4.3 Phase 3: Logical Operators
- [ ] Add `AND`, `OR`, `NOT` to grammar
- [ ] Implement in interpreter
- [ ] Update mutator
- [ ] Add tests

**Priority**: P2
**Estimated time**: 2 days
**Files**: `base_agent/src/dsl/*.py`

### 4.4 Phase 4: Multi-Timeframe Syntax
- [ ] Add `SYMBOL_TIMEFRAME(N)` syntax to grammar
- [ ] Update interpreter to handle multi-timeframe data dict
- [ ] Implement timestamp alignment logic
- [ ] Update mutator to mutate timeframes
- [ ] Add tests

**Priority**: P3 (Advanced feature)
**Estimated time**: 3-4 days
**Files**: `base_agent/src/dsl/*.py`

## Phase 5: Query Tools and Analysis (⏸️ DEFERRED)

**Status**: ⏸️ **DEFERRED** - Not needed immediately

**Rationale**:
- **CLI query tool is redundant**: `sqlite3` command-line tool + `DATABASE_SCHEMA.md` (lines 382-523) already document all queries
- **Analysis scripts duplicate Phase 6**: Web interface (real-time analytics) is more valuable than post-mortem CLI reports
- **Better alternatives exist**:
  - Use `sqlite3` directly for quick queries
  - Use Python REPL with `CellRepository` for programmatic access
  - Create Jupyter notebook for exploratory analysis (10 minutes to set up)
- **Vision captured in docs**: See `WEB_INTERFACE.md` "Evolution Analysis Dashboard" section for what this analysis should look like when built into Phase 6

**Recommendation**: Skip Phase 5 entirely, or defer to "Phase 9: Nice-to-Haves" after Phases 4, 6, 7 are complete. Focus on **DSL enhancements** (Sprint 4) and **web interface extensions** (Phase 6) instead.

---

### 5.1 CLI Query Tool (DEFERRED)
- [ ] Create `query_cells.py` script
  - [ ] Command: `python query_cells.py top --limit 10`
  - [ ] Command: `python query_cells.py lineage --cell-id 47`
  - [ ] Command: `python query_cells.py unanalyzed --limit 50`
  - [ ] Command: `python query_cells.py patterns`
  - [ ] Command: `python query_cells.py pattern --name "Volume Analysis"`

**Priority**: ~~P2~~ → P9 (Deferred)
**Estimated time**: 1 day
**Files**: `query_cells.py`

**Alternative**: Use `sqlite3` directly or Python REPL:
```bash
# CLI query example
sqlite3 cells.db "SELECT cell_id, fitness, dsl_genome FROM cells ORDER BY fitness DESC LIMIT 10"

# Python REPL example
python -c "from base_agent.src.storage.cell_repository import CellRepository; \
  repo = CellRepository('cells.db'); print(repo.get_top_cells(10))"
```

### 5.2 Analysis Scripts (DEFERRED)
- [ ] Create `analyze_evolution.py` script
  - [ ] Plot fitness over generations
  - [ ] Show survival rate statistics
  - [ ] Display pattern taxonomy
  - [ ] Generate lineage graphs
  - [ ] Export reports

**Priority**: ~~P2~~ → P9 (Deferred)
**Estimated time**: 1-2 days
**Files**: `analyze_evolution.py`

**Alternative**: Build into Phase 6 web interface (real-time, interactive) or use Jupyter notebook for ad-hoc analysis. See `WEB_INTERFACE.md:202-358` for full vision of what this should become.

## Phase 6: Web Interface Extensions

**Note**: The web server **already exists** (inherited from original SICA framework). It shows agent execution trees, not cell evolution. This phase extends it with cell-specific features.

### 6.1 Cell API Endpoints (Extend Existing Server)
- [x] ✅ **Web server infrastructure exists** (from SICA: `base_agent/src/web_server/`)
- [x] ✅ FastAPI server running on port 8080
- [x] ✅ WebSocket for real-time updates
- [x] ✅ Existing endpoints: `/`, `/ws`, `/api/callgraph`
- [ ] **Add cell-specific endpoints** to existing `server.py`:
  - [ ] `GET /api/cells` - List cells with filtering
  - [ ] `GET /api/cells/{cell_id}` - Cell details (genome, fitness, status)
  - [ ] `GET /api/cells/{cell_id}/lineage` - Ancestry chain
  - [ ] `GET /api/cells/{cell_id}/phenotypes` - Performance across timeframes
  - [ ] `GET /api/patterns` - Pattern taxonomy list
  - [ ] `GET /api/patterns/{pattern_id}/cells` - Cells using a pattern

**Priority**: P3
**Estimated time**: 2 days (extending existing server, not building from scratch)
**Files**: `base_agent/src/web_server/server.py` (extend existing)

### 6.2 Cell Visualization Components (New UI for Cells)
- [x] ✅ **Existing UI** shows agent execution (callgraph, events, metrics)
- [ ] **Add new cell-specific components** to existing UI:
  - [ ] Create `static/components/cell-viewer.js` - Cell list/grid view
  - [ ] Create `static/components/cell-details.js` - Individual cell panel
  - [ ] Create `static/components/lineage-graph.js` - Ancestry visualization
  - [ ] Create `static/components/pattern-taxonomy.js` - Pattern browser
  - [ ] Add navigation tab to existing `index.html` for cell view

**Priority**: P3
**Estimated time**: 3-4 days
**Files**: `base_agent/src/web_server/static/components/*.js` (new files)

## Phase 7: Testing and Validation

### 7.1 Unit Tests
- [ ] Test `CellRepository` methods
- [ ] Test multi-timeframe backtesting
- [ ] Test DSL V2 parsing and evaluation
- [ ] Test LLM analysis pipeline
- [ ] Test mutation generation

**Priority**: P1
**Estimated time**: Ongoing throughout implementation
**Files**: `base_agent/tests/**/*.py`

### 7.2 Integration Tests
- [ ] Test full evolution run with cell storage
- [ ] Test trading-learn mode end-to-end
- [ ] Test database queries under load
- [ ] Test multi-timeframe data fetching

**Priority**: P1
**Estimated time**: 2-3 days
**Files**: `base_agent/tests/integration/*.py`

### 7.3 Validation Runs
- [ ] Run 50-generation evolution with DSL V2
- [ ] Compare vs V1 results (should break $6.17 barrier)
- [ ] Run trading-learn for 10 iterations
- [ ] Verify pattern taxonomy emerges
- [ ] Test on multiple symbols (PURR, HFUN, BTC)

**Priority**: P1
**Estimated time**: 1-2 days (mostly runtime)
**Files**: N/A (testing)

## Phase 8: Documentation and Polish

### 8.1 Code Documentation
- [ ] Add docstrings to all new classes
- [ ] Add type hints throughout
- [ ] Create usage examples in docstrings
- [ ] Generate API documentation

**Priority**: P2
**Estimated time**: 1-2 days
**Files**: All implementation files

### 8.2 User Guides
- [ ] Create step-by-step tutorial for cell queries
- [ ] Document LLM analysis workflow
- [ ] Create troubleshooting guide
- [ ] Add performance tuning tips

**Priority**: P2
**Estimated time**: 1 day
**Files**: `cursor_docs/*.md`

## Implementation Status Summary

### ✅ Sprint 1: Core Infrastructure - COMPLETE (2025-10-09)
1. ✅ Database setup (1.1)
2. ✅ Data classes (1.2)
3. ✅ Multi-timeframe data fetching (1.3)
4. ✅ Update backtest for multi-timeframe (2.1)

**Goal**: ✅ ACHIEVED - Can fetch data and store cells in database

### ✅ Sprint 2: Evolution Integration - COMPLETE (2025-10-09)
1. ✅ Integrate cell birth into evolution (2.2)
2. ✅ Update mutation selection (2.3)
3. ✅ DSL V2 Phase 1: Arithmetic (4.1)

**Goal**: ✅ ACHIEVED - Evolution mode works with cell storage and arithmetic DSL

### ✅ Sprint 3: LLM Integration - COMPLETE (2025-10-09)
1. ✅ LLM analysis pipeline with batch processing (3.1)
2. ✅ Intelligent mutation proposals (3.2)
3. ✅ Rewrite trading-learn mode for 100% LLM-guided evolution (3.3)
4. ✅ Database integration for patterns and proposals (3.4)

**Goal**: ✅ ACHIEVED - Trading-learn mode analyzes cell library and uses 100% LLM-guided mutations

### 🔄 Sprint 4: DSL Enhancement - NEXT PRIORITY
1. DSL V2 Phase 2: Aggregations (4.2) - AVG, SUM, MAX, MIN, STD
2. DSL V2 Phase 3: Logical operators (4.3) - AND, OR, NOT
3. DSL V2 Phase 4: Multi-timeframe syntax (4.4) - SYMBOL_TIMEFRAME(N)

**Goal**: Complete DSL V2 with aggregations, logic, and multi-timeframe support
**Note**: Phase 5 (CLI tools) deferred - focus on DSL completeness first

### 🔄 Sprint 5: Validation - FUTURE
1. Comprehensive testing (7.1, 7.2)
2. Validation runs (7.3)
3. Performance optimization

**Goal**: System proven to work, breaks V1 limitations, ready for production

### 🔄 Sprint 6: Polish and Extensions - FUTURE
1. Web interface extensions (6.1, 6.2) - Real-time evolution analytics dashboard
2. Documentation (8.1, 8.2)
3. Multi-symbol validation

**Goal**: Production-ready system with visualization and comprehensive docs

## Success Metrics

### MVP Success (After Sprint 2):
- ✅ Cells stored in database with lineage
- ✅ Evolution runs for 50 generations without errors
- ✅ Arithmetic DSL expressions work
- ✅ Best fitness > $6.17 (breaks V1 stagnation)

### Full Success (After Sprint 5):
- ✅ Trading-learn discovers and names 5+ patterns
- ✅ Pattern taxonomy builds over 100 generations
- ✅ LLM-guided mutations show better convergence than random
- ✅ Strategies work across multiple timeframes
- ✅ System handles 1000+ cells without performance issues

### Production Ready (After Sprint 6):
- ✅ Web interface visualizes cells and patterns (real-time evolution analytics)
- ✅ Interactive lineage graphs and pattern taxonomy browser
- ✅ Documentation complete and tested
- ✅ Ready for multi-symbol, walk-forward validation

## Estimated Total Time

- **Minimum (MVP)**: 2 weeks (Sprints 1-2)
- **Full Feature Set**: 5 weeks (Sprints 1-5)
- **Production Ready**: 6 weeks (All sprints)

## Dependencies

**External**:
- SQLite (already available)
- pandas (already installed)
- anthropic SDK (already installed)

**Internal**:
- Existing DSL infrastructure
- Existing trading benchmark
- Existing LLM client

**No major external dependencies needed!**

## Risk Mitigation

**Risk 1**: DSL V2 too complex to implement
- **Mitigation**: Implement incrementally (Phase 1 → 2 → 3 → 4)
- **Fallback**: Can ship with just arithmetic (Phase 1)

**Risk 2**: LLM analysis too expensive
- **Mitigation**: Make analysis optional, use budget limits
- **Fallback**: Run evolution mode (free), analyze cells manually later

**Risk 3**: Database performance with large cell count
- **Mitigation**: Indexes already designed, tested to 10K+ cells
- **Fallback**: Add archiving for old cells

**Risk 4**: Multi-timeframe alignment issues
- **Mitigation**: Use timestamp matching, well-tested approach
- **Fallback**: Start with single timeframe, add multi-TF later

## Next Steps

1. **Review this plan** with stakeholders
2. **Set up development environment** (database tools, testing framework)
3. **Start Sprint 1** with database setup
4. **Create GitHub issues** for each task (if using issue tracking)
5. **Set up CI/CD** for automated testing

---

**This implementation plan brings all the documentation to life. Ready to build the Money Making Lifeforms! 🧬💰**
