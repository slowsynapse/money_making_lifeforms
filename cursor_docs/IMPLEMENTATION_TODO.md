# Implementation TODO: Cell-Based Evolution System

This document outlines all tasks needed to implement the cell-based architecture described in the documentation.

## Phase 1: Core Infrastructure (CRITICAL)

### 1.1 Database Setup
- [ ] Create `base_agent/src/storage/__init__.py`
- [ ] Create `base_agent/src/storage/cell_repository.py`
  - [ ] Implement `CellRepository` class
  - [ ] Implement database initialization with schema from `DATABASE_SCHEMA.md`
  - [ ] Implement all CRUD methods from `CELL_STORAGE_API.md`
  - [ ] Add transaction support (context manager)
  - [ ] Write unit tests for repository methods

**Priority**: P0 (Blocker for all other work)
**Estimated time**: 2-3 days
**Files**: `base_agent/src/storage/cell_repository.py`

### 1.2 Data Classes
- [ ] Create `base_agent/src/storage/models.py`
  - [ ] Implement `Cell` dataclass
  - [ ] Implement `CellPhenotype` dataclass
  - [ ] Implement `DiscoveredPattern` dataclass
  - [ ] Implement `EvolutionRun` dataclass
  - [ ] Add `to_dict()` methods for JSON serialization
  - [ ] Add validation logic

**Priority**: P0
**Estimated time**: 1 day
**Files**: `base_agent/src/storage/models.py`

### 1.3 Multi-Timeframe Data Fetching
- [ ] Update `base_agent/src/data/hyperliquid_fetcher.py`
  - [ ] Add method to fetch multiple timeframes at once
  - [ ] Implement `fetch_multi_timeframe(symbol, timeframes=['1H', '4H', '1D'], days=30)`
  - [ ] Add caching to avoid re-fetching
  - [ ] Handle timestamp alignment between timeframes

**Priority**: P0
**Estimated time**: 1 day
**Files**: `base_agent/src/data/hyperliquid_fetcher.py`

## Phase 2: Evolution Mode Integration

### 2.1 Update Backtest to Support Multi-Timeframe
- [ ] Update `trading_benchmark.py`
  - [ ] Change `_run_backtest()` signature to accept multi-timeframe data
  - [ ] Test strategy on all timeframes (1H, 4H, 1D)
  - [ ] Return dict of results per timeframe
  - [ ] Select best timeframe for fitness

**Priority**: P0
**Estimated time**: 2 days
**Files**: `base_agent/src/benchmarks/trading_benchmarks/trading_benchmark.py`

### 2.2 Integrate Cell Birth into Evolution
- [ ] Update `run_trading_evolve()` in `agent.py`
  - [ ] Initialize `CellRepository` at start
  - [ ] Create `evolution_runs` record
  - [ ] Birth cells for successful mutations
  - [ ] Record failures for unsuccessful mutations
  - [ ] Update evolution_run stats (total_cells_birthed, etc.)
  - [ ] Complete evolution_run at end

**Priority**: P0
**Estimated time**: 2 days
**Files**: `base_agent/agent.py`

### 2.3 Update Mutation Selection
- [ ] Update mutation logic to use best cell from database
  - [ ] Query `repo.get_top_cells(limit=1)` instead of tracking in-memory
  - [ ] Support different parent selection modes (best, tournament, fitness-proportional)

**Priority**: P1
**Estimated time**: 1 day
**Files**: `base_agent/agent.py`

## Phase 3: LLM Integration (Trading-Learn Mode)

**Design Philosophy**:
- `trading-evolve`: 100% random mutations (FREE, builds genetic library)
- `trading-learn`: 100% LLM-guided mutations (SMART, exploits patterns)
- Clear separation: cheap exploration → intelligent exploitation

### 3.1 LLM Analysis Pipeline with Batch Processing
- [ ] Create `base_agent/src/analysis/__init__.py`
- [ ] Create `base_agent/src/analysis/cell_analyzer.py`
  - [ ] Implement `prepare_cell_context(cell_id, repo)` - fetch cell + lineage + phenotypes
  - [ ] Implement `analyze_cells_in_batches(repo, cell_ids, batch_size=30)` - batch processing for 8K context
  - [ ] Implement `merge_pattern_discoveries(batch_results)` - deduplicate patterns across batches
  - [ ] Create LLM prompt template for batch cell analysis
  - [ ] Parse JSON response from LLM (pattern taxonomy)
  - [ ] Handle errors gracefully (skip failed batches, continue)

**Priority**: P1
**Estimated time**: 2-3 days
**Files**: `base_agent/src/analysis/cell_analyzer.py`
**Notes**: Batch size of 30 cells fits Gemma 3 27B's 8K context window

### 3.2 Intelligent Mutation Proposals
- [ ] Create `base_agent/src/analysis/mutation_proposer.py`
  - [ ] Implement `propose_intelligent_mutation(cell, patterns, repo)` - LLM suggests smart mutations
  - [ ] Create LLM prompt template for mutation proposals
  - [ ] Parse mutation proposal JSON (strategy, rationale, expected_improvement)
  - [ ] Validate proposed mutations (parseable DSL, different from parent)

**Priority**: P1
**Estimated time**: 1-2 days
**Files**: `base_agent/src/analysis/mutation_proposer.py`

### 3.3 Rewrite Trading-Learn Mode (100% LLM-Guided)
- [ ] **Completely rewrite** `run_trading_learn()` in `agent.py`
  - [ ] Load cell database from prior `trading-evolve` run
  - [ ] Analyze top 100 cells in batches (30 cells per batch)
  - [ ] Build pattern taxonomy from batch analysis
  - [ ] For each iteration:
    - [ ] Select best cell (or tournament selection)
    - [ ] LLM proposes intelligent mutation based on patterns
    - [ ] Parse and validate proposed strategy
    - [ ] Test on multi-timeframe backtest
    - [ ] Birth cell if fitness improves
    - [ ] Update pattern taxonomy with new insights
  - [ ] Track LLM costs and display budget usage
  - [ ] Store all analysis in database (patterns, cell_patterns, mutation proposals)

**Priority**: P1 (CRITICAL)
**Estimated time**: 2-3 days
**Files**: `base_agent/agent.py`
**Breaking Change**: Old `trading-learn` used MainOrchestratorAgent (generate from scratch). New version uses cell database + intelligent mutation.

### 3.4 Database Integration
- [ ] Verify `cell_mutation_proposals` table exists in schema
- [ ] Implement `store_mutation_proposal()` in CellRepository
- [ ] Implement `get_mutation_proposals_for_cell()` in CellRepository
- [ ] Implement `store_pattern()` in CellRepository
- [ ] Implement `link_cell_to_pattern()` in CellRepository

**Priority**: P1
**Estimated time**: 1 day
**Files**: `base_agent/src/storage/cell_repository.py`
**Note**: Most methods already exist from Sprint 1, just verify completeness

## Phase 4: DSL V2 Implementation

### 4.1 Phase 1: Arithmetic Operations (CRITICAL for breaking stagnation)
- [ ] Update `base_agent/src/dsl/grammar.py`
  - [ ] Add `BinaryOp` class for +, -, *, /
  - [ ] Update grammar to support expressions
- [ ] Update `base_agent/src/dsl/interpreter.py`
  - [ ] Implement `evaluate_expression()` function
  - [ ] Handle arithmetic precedence
  - [ ] Add division-by-zero protection
- [ ] Update `base_agent/src/dsl/mutator.py`
  - [ ] Add arithmetic expression mutations
  - [ ] Mutate operators (+  → -)
  - [ ] Mutate operand order
  - [ ] Add parentheses mutations
- [ ] Add tests for arithmetic DSL

**Priority**: P0 (Critical - solves "volume vs price" problem)
**Estimated time**: 3-4 days
**Files**: `base_agent/src/dsl/*.py`

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

## Phase 5: Query Tools and Analysis

### 5.1 CLI Query Tool
- [ ] Create `query_cells.py` script
  - [ ] Command: `python query_cells.py top --limit 10`
  - [ ] Command: `python query_cells.py lineage --cell-id 47`
  - [ ] Command: `python query_cells.py unanalyzed --limit 50`
  - [ ] Command: `python query_cells.py patterns`
  - [ ] Command: `python query_cells.py pattern --name "Volume Analysis"`

**Priority**: P2
**Estimated time**: 1 day
**Files**: `query_cells.py`

### 5.2 Analysis Scripts
- [ ] Create `analyze_evolution.py` script
  - [ ] Plot fitness over generations
  - [ ] Show survival rate statistics
  - [ ] Display pattern taxonomy
  - [ ] Generate lineage graphs
  - [ ] Export reports

**Priority**: P2
**Estimated time**: 1-2 days
**Files**: `analyze_evolution.py`

## Phase 6: Web Interface Extensions

### 6.1 Cell API Endpoints
- [ ] Add to `base_agent/src/web_server/server.py`
  - [ ] `GET /api/cells` - List cells
  - [ ] `GET /api/cells/{cell_id}` - Cell details
  - [ ] `GET /api/cells/{cell_id}/lineage` - Ancestry
  - [ ] `GET /api/cells/{cell_id}/phenotypes` - Performance data
  - [ ] `GET /api/patterns` - Pattern taxonomy
  - [ ] `GET /api/patterns/{pattern_id}/cells` - Cells by pattern

**Priority**: P3
**Estimated time**: 2 days
**Files**: `base_agent/src/web_server/server.py`

### 6.2 Cell Visualization Components
- [ ] Create `static/components/cell-viewer.js`
  - [ ] Cell list view
  - [ ] Cell details panel
  - [ ] Lineage graph visualization
- [ ] Create `static/components/pattern-taxonomy.js`
  - [ ] Pattern category browser
  - [ ] Pattern performance metrics
- [ ] Update main page to include cell views

**Priority**: P3
**Estimated time**: 3-4 days
**Files**: `base_agent/src/web_server/static/components/*.js`

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

## Implementation Order (Recommended)

### Sprint 1 (Week 1): Core Infrastructure
1. Database setup (1.1)
2. Data classes (1.2)
3. Multi-timeframe data fetching (1.3)
4. Update backtest for multi-timeframe (2.1)

**Goal**: Can fetch data and store cells in database

### Sprint 2 (Week 2): Evolution Integration
1. Integrate cell birth into evolution (2.2)
2. Update mutation selection (2.3)
3. DSL V2 Phase 1: Arithmetic (4.1)

**Goal**: Evolution mode works with cell storage and arithmetic DSL

### Sprint 3 (Week 3): LLM Integration
1. LLM analysis pipeline with batch processing (3.1)
2. Intelligent mutation proposals (3.2)
3. Rewrite trading-learn mode for 100% LLM-guided evolution (3.3)
4. Database integration for patterns and proposals (3.4)

**Goal**: Trading-learn mode analyzes cell library and uses 100% LLM-guided mutations

### Sprint 4 (Week 4): DSL Enhancement
1. DSL V2 Phase 2: Aggregations (4.2)
2. DSL V2 Phase 3: Logical operators (4.3)
3. CLI query tool (5.1)

**Goal**: Rich DSL with query capabilities

### Sprint 5 (Week 5): Validation
1. Comprehensive testing (7.1, 7.2)
2. Validation runs (7.3)
3. Analysis scripts (5.2)

**Goal**: System proven to work, breaks V1 limitations

### Sprint 6 (Week 6): Polish and Extensions
1. Web interface extensions (6.1, 6.2)
2. Documentation (8.1, 8.2)
3. Performance optimization

**Goal**: Production-ready system with visualization

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
- ✅ Web interface visualizes cells and patterns
- ✅ Query tools make analysis easy
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
