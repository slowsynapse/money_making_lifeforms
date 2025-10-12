# Current Tasks (Active TODO)

**Last Updated**: 2025-10-12

## 🎯 Active Sprint: Context Optimization

### In Progress
- [ ] Verify trading module refactoring works in Docker
- [ ] Test evolution modes with 100 generations

### Next Up
- [ ] DSL Phase 2: Aggregation functions (AVG, SUM, MAX, MIN, STD)
- [ ] DSL Phase 3: Logical operators (AND, OR, NOT)
- [ ] DSL Phase 4: Multi-timeframe syntax

## Quick Reference

**Run evolution**:
```bash
docker run --rm -v $(pwd):/app sica_sandbox \
  ./trading_cli.py evolve --generations 100 --fitness-goal 50.0
```

**Check results**:
```bash
sqlite3 results/evolve_*/evolution/cells.db \
  "SELECT * FROM cells ORDER BY fitness DESC LIMIT 5;"
```

## Completed Recently
- ✅ Sprint 1-3: Core infrastructure, evolution mode, LLM integration
- ✅ DSL Phase 1: Arithmetic operations
- ✅ Trading module refactoring (agent.py: 1,862 → 734 lines)
- ✅ Documentation reorganization

## See Also
- Full sprint history: `cursor_docs/reference/IMPLEMENTATION_TODO_ARCHIVE.md`
- Docker commands: `DOCKER_COMMANDS.md`
- Quick reference: `QUICK_REFERENCE.md`
