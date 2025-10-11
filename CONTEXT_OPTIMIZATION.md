# Context Optimization Summary

This document summarizes the changes made to reduce Claude Code context consumption.

## Problem

Claude Code conversations were frequently running out of context mid-feature due to:
1. Very large Python files (agent.py at 1,862 lines)
2. Large documentation files being auto-loaded (~237KB total)
3. Results and database files being scanned

## Changes Made

### 1. Code Refactoring

**Extracted trading logic from monolithic agent.py:**
- Created `base_agent/src/trading/` module
- Moved 1,136 lines to `trading_evolution.py`
- Created `trading_config.py` for centralized config
- **Result**: agent.py reduced from 1,862 → 734 lines (60% reduction)

### 2. Documentation Reorganization

**Created directory structure:**
```
cursor_docs/
├── QUICK_REFERENCE.md          (NEW - concise overview)
├── IMPLEMENTATION_TODO.md       (kept - active tasks)
├── DOCKER_COMMANDS.md           (kept - quick reference)
├── TESTING.md                   (kept - how to test)
├── VISION.md                    (kept - project overview)
├── CLI_WRAPPER.md               (kept - small file)
├── .claudeignore/               (planning docs - 86KB)
│   ├── agent_orchestration_data_needs.md
│   ├── agent_orchestration_ideal_flow.md
│   ├── future_on_chain_intelligence.md
│   └── OPUS_PLAN.md
└── reference/                   (technical docs - 97KB)
    ├── CELL_ARCHITECTURE.md
    ├── CELL_STORAGE_API.md
    ├── DATABASE_SCHEMA.md
    ├── EVOLUTION_WORKFLOW.md
    ├── EVOLUTIONARY_LOOP.md
    └── LLM_INTEGRATION.md
```

**Result**: 183KB (77%) of docs excluded from auto-load

### 3. Git Ignore Updates

Added to `.gitignore`:
```gitignore
# Trading evolution results and databases
results/
*.db
*.db-journal
*.db-shm
*.db-wal

# Claude Code - exclude large docs from auto-loading
cursor_docs/.claudeignore/
cursor_docs/reference/
```

### 4. Updated CLAUDE.md

Changed from verbose to ultra-concise:
```markdown
## Project Context
- This runs in a Docker container (local Python is broken, avoid root)
- Primary tasklist: `cursor_docs/IMPLEMENTATION_TODO.md`
- Test with 100 generations
- Restart server after code changes to flush old code

## Key Files
- Trading logic: `base_agent/src/trading/`
- Cell storage: `base_agent/src/storage/cell_repository.py`
- DSL system: `base_agent/src/dsl/`

## Documentation Structure
- Quick reference: `cursor_docs/` (DOCKER_COMMANDS.md, TESTING.md, etc.)
- Deep reference: `cursor_docs/reference/` (only read when needed)
- Planning docs: `cursor_docs/.claudeignore/` (only read when explicitly requested)
```

## Impact

### Before
- agent.py: 1,862 lines (read on most operations)
- Documentation: ~237KB auto-loaded
- Database files scanned
- **Result**: Frequent context exhaustion

### After
- agent.py: 734 lines (60% smaller)
- Documentation: ~54KB auto-loaded (77% reduction)
- Database files ignored
- **Estimated context savings: ~70%**

## Verification

All changes verified:
- ✅ Python files compile without errors
- ✅ Import structure validated (`verify_refactor.py`)
- ✅ All 4 trading functions properly exported
- ✅ Directory structure created correctly

## Files Added

- `base_agent/src/trading/__init__.py`
- `base_agent/src/trading/trading_config.py`
- `base_agent/src/trading/trading_evolution.py`
- `cursor_docs/QUICK_REFERENCE.md`
- `cursor_docs/reference/README.md`
- `cursor_docs/.claudeignore/README.md`
- `verify_refactor.py`
- `CONTEXT_OPTIMIZATION.md` (this file)

## Next Steps (Recommended)

1. ✅ Test trading modes work in Docker
2. Create `trading_cli.py` wrapper (referenced in DOCKER_COMMANDS.md)
3. Consider further modularization:
   - Extract LLM provider code (~2,300 lines across 4 files)
   - Split large storage/repository file (981 lines)
   - Modularize utility files (871 lines in archive_analysis.py)

## Maintenance

When adding new documentation:
- **Quick reference** (< 300 lines, frequently needed): Put in `cursor_docs/`
- **Deep reference** (technical details, rarely needed): Put in `cursor_docs/reference/`
- **Planning/vision** (future ideas): Put in `cursor_docs/.claudeignore/`

When writing code:
- Keep files under 800 lines when possible
- Extract coherent subsystems into dedicated modules
- Use relative imports correctly (`..` for parent package)
