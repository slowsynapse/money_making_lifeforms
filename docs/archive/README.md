# Documentation Archive

This directory contains historical documentation that has been archived to reduce clutter while preserving the project's history.

## Archive Structure

```
docs/archive/
├── sprints/      # Historical sprint completion reports
└── analysis/     # Old progress reports and evolution analyses
```

## Archived Files

### Sprint Reports (`sprints/`)

Historical sprint completion documentation:

- **`sprint_1_complete.md`** (2025-10-09) - Sprint 1: Core Infrastructure completion report
- **`sprint_2_complete.md`** (2025-10-09) - Sprint 2: Evolution Integration completion report
- **`sprint_2_started.md`** (2025-10-09) - Sprint 2 kickoff notes

These have been superseded by:
- Current status: `SPRINT_3_COMPLETE.md` (root directory)
- Implementation tracking: `cursor_docs/IMPLEMENTATION_TODO.md`

### Analysis Reports (`analysis/`)

Early evolution analysis and progress reports:

- **`progress_report_1.md`** (~22K) - Detailed early progress analysis
- **`progress_report_2.md`** (~9.5K) - Follow-up progress analysis
- **`evolution_analysis.md`** (~10.3K) - Evolution behavior analysis

These contain valuable insights from early development but are no longer actively maintained.

## Current Active Documentation

For up-to-date documentation, see:

### Root Directory
- **`README.md`** - Project overview and quick start
- **`SPRINT_3_COMPLETE.md`** - Latest sprint completion status
- **`DOCKER_COMMANDS.md`** - Comprehensive Docker usage guide
- **`TRADING_QUICKSTART.md`** - Quick reference for trading modes

### Documentation (`cursor_docs/`)
- **`IMPLEMENTATION_TODO.md`** - Implementation status and TODO tracking
- **`CELL_ARCHITECTURE.md`** - Cell-based evolution system design
- **`CELL_STORAGE_API.md`** - Database API reference
- **`DATABASE_SCHEMA.md`** - Database schema documentation
- **`DSL_DESIGN.md`** - DSL philosophy and rationale
- **`DSL_V2_SPEC.md`** - DSL V2 specification
- **`EVOLUTIONARY_LOOP.md`** - Evolution cycle overview
- **`EVOLUTION_WORKFLOW.md`** - Complete evolution workflow
- **`LLM_INTEGRATION.md`** - LLM integration guide
- **`WEB_INTERFACE.md`** - Web visualization guide
- **`TROUBLESHOOTING.md`** - Common issues and solutions
- **`TESTING.md`** - Testing guidelines

## Why These Files Were Archived

1. **Sprint Reports**: Replaced by consolidated `SPRINT_3_COMPLETE.md` and tracked in `IMPLEMENTATION_TODO.md`
2. **Analysis Reports**: Historical snapshots from early development, insights preserved but no longer actively referenced
3. **Redundancy**: Multiple files describing the same completed work
4. **Outdated Information**: Some content no longer accurate after subsequent sprints

## Accessing Archived Files

All files are preserved in version control. To view:

```bash
# View sprint completion history
ls docs/archive/sprints/

# View analysis reports
ls docs/archive/analysis/

# Read specific archived file
cat docs/archive/sprints/sprint_1_complete.md
```

## Notes

- Files are archived, not deleted - full git history is preserved
- Archived files are read-only references
- For current implementation status, always refer to `cursor_docs/IMPLEMENTATION_TODO.md`
- For latest sprint status, see `SPRINT_3_COMPLETE.md` in root directory

---

**Archive Created**: 2025-10-09
**Last Updated**: 2025-10-09
