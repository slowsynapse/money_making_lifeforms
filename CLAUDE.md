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