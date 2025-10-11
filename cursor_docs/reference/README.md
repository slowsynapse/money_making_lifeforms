# Reference Documentation

This directory contains detailed technical reference documentation about the SICA system internals.

## Files in this Directory

### Architecture & Design
- **CELL_ARCHITECTURE.md** - Cell-based evolutionary architecture design
- **EVOLUTIONARY_LOOP.md** - How the evolutionary loop operates
- **EVOLUTION_WORKFLOW.md** - Complete workflow documentation

### Implementation Details
- **CELL_STORAGE_API.md** - Cell repository API reference
- **DATABASE_SCHEMA.md** - SQLite database schema details
- **LLM_INTEGRATION.md** - LLM provider integration guide

## Purpose

These files provide deep technical details useful for:
- Understanding internal implementation
- Modifying core systems
- API reference when extending functionality
- Debugging complex issues

They are **NOT** auto-loaded by Claude Code to save context. Claude will read them only when specifically needed.

## Quick Reference

For everyday tasks, see the main `cursor_docs/` files:
- `IMPLEMENTATION_TODO.md` - Current task list
- `DOCKER_COMMANDS.md` - How to run the system
- `TESTING.md` - Test instructions
- `VISION.md` - Project overview
