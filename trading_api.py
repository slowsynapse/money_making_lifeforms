#!/usr/bin/env python3
"""
Trading Evolution System API
Provides a FastAPI server for the trading evolution system.
This runs INSIDE Docker and provides endpoints the web UI can call.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio
import os
import sys
import json
from datetime import datetime

# Add base_agent to path
sys.path.insert(0, str(Path(__file__).parent))

from base_agent.src.storage.cell_repository import CellRepository
from base_agent.src.storage.models import Cell, CellPhenotype

app = FastAPI(
    title="Trading Evolution API",
    description="API for cell-based trading strategy evolution",
    version="1.0.0"
)

# Enable CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your web UI origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the web UI
web_dir = Path(__file__).parent / "trading_web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

# Global repository instance
repo: Optional[CellRepository] = None


def get_repository() -> CellRepository:
    """Get or create repository instance."""
    global repo
    if repo is None:
        # Find the most recent cells.db
        db_paths = []
        results_dir = Path("results")
        if results_dir.exists():
            all_dbs = list(results_dir.glob("**/cells.db"))
            all_dbs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            db_paths.extend(all_dbs)

        if Path("cells.db").exists():
            db_paths.append(Path("cells.db"))

        if not db_paths:
            raise HTTPException(status_code=404, detail="No cell database found")

        repo = CellRepository(db_paths[0])
    return repo


@app.get("/")
async def root():
    """Serve the dashboard HTML."""
    dashboard_path = Path(__file__).parent / "trading_web" / "index.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)

    # Fallback to API info if dashboard doesn't exist
    return {
        "name": "Trading Evolution API",
        "version": "1.0.0",
        "endpoints": [
            "/cells/top/{limit}",
            "/cell/{cell_id}",
            "/cell/{cell_id}/phenotypes",
            "/cell/{cell_id}/lineage",
            "/patterns",
            "/evolution/status",
            "/evolution/start"
        ]
    }


@app.get("/cells/top/{limit}")
async def get_top_cells(limit: int = 10, min_trades: int = 0, has_llm: Optional[bool] = None):
    """Get top cells by fitness.

    Args:
        limit: Maximum number of cells to return
        min_trades: Minimum number of trades required
        has_llm: Filter by LLM involvement (True=LLM only, False=evolution only, None=all)
    """
    try:
        repo = get_repository()
        cells = repo.get_top_cells(limit=limit, min_trades=min_trades)

        result = []
        for cell in cells:
            # Filter by LLM involvement if requested
            if has_llm is not None:
                if has_llm and cell.llm_name is None:
                    continue  # Skip evolution-only cells
                elif not has_llm and cell.llm_name is not None:
                    continue  # Skip LLM cells

            # Get phenotypes for trade count
            phenotypes = repo.get_phenotypes(cell.cell_id)
            total_trades = sum(p.total_trades for p in phenotypes) if phenotypes else 0

            result.append({
                "cell_id": cell.cell_id,
                "generation": cell.generation,
                "fitness": cell.fitness,
                "total_trades": total_trades,
                "status": cell.status,
                "dsl_genome": cell.dsl_genome,
                "llm_name": cell.llm_name,
                "parent_cell_id": cell.parent_cell_id,
                "created_at": cell.created_at.isoformat() if cell.created_at else None
            })

        return {"cells": result, "count": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cell/{cell_id}")
async def get_cell(cell_id: int):
    """Get a specific cell by ID."""
    try:
        repo = get_repository()
        cell = repo.get_cell(cell_id)

        if not cell:
            raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found")

        return cell.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cell/{cell_id}/phenotypes")
async def get_cell_phenotypes(cell_id: int):
    """Get phenotypes for a cell."""
    try:
        repo = get_repository()
        phenotypes = repo.get_phenotypes(cell_id)

        return {
            "cell_id": cell_id,
            "phenotypes": [p.to_dict() for p in phenotypes],
            "count": len(phenotypes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cell/{cell_id}/lineage")
async def get_cell_lineage(cell_id: int):
    """Get lineage (ancestry) for a cell."""
    try:
        repo = get_repository()
        lineage = repo.get_lineage(cell_id)

        return {
            "cell_id": cell_id,
            "lineage": [cell.to_dict() for cell in lineage],
            "depth": len(lineage)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patterns")
async def get_patterns(category: Optional[str] = None):
    """Get discovered patterns."""
    try:
        repo = get_repository()

        if category:
            patterns = repo.get_patterns_by_category(category)
        else:
            # For now, return empty as we need to implement get_all_patterns
            patterns = []

        return {
            "patterns": [p.to_dict() for p in patterns],
            "count": len(patterns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evolution/status")
async def get_evolution_status():
    """Get current evolution status."""
    # This would check if evolution is running
    # For now, return a simple status
    return {
        "running": False,
        "current_generation": None,
        "best_fitness": None,
        "cells_birthed": None
    }


@app.post("/evolution/start")
async def start_evolution(
    background_tasks: BackgroundTasks,
    generations: int = 100,
    fitness_goal: float = 50.0
):
    """Start evolution process."""
    # This would trigger evolution in the background
    # For now, return a placeholder response
    return {
        "message": "Evolution starting...",
        "parameters": {
            "generations": generations,
            "fitness_goal": fitness_goal
        }
    }


@app.get("/stats")
async def get_statistics():
    """Get database statistics."""
    try:
        repo = get_repository()
        total_cells = repo.get_cell_count()

        # Get failure statistics
        failure_stats = repo.get_failure_statistics()

        return {
            "total_cells": total_cells,
            "failure_statistics": failure_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8081,
        log_level="info"
    )