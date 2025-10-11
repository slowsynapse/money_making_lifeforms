## Project Context
- This runs in a Docker container (local Python is broken, avoid root)
- Primary tasklist: `cursor_docs/IMPLEMENTATION_TODO.md`
- Test with 100 generations
- Restart server after code changes to flush old code

## Key Files
- Trading logic: `base_agent/src/trading/` (or `agent.py` if not yet refactored)
- Cell storage: `base_agent/src/storage/cell_repository.py`
- DSL system: `base_agent/src/dsl/`

## Notes
- Documentation in `cursor_docs/` - read specific files as needed, don't load all
- Planning docs in `cursor_docs/.claudeignore/` - only read when explicitly needed