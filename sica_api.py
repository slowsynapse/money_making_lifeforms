#!/usr/bin/env python3
"""
SICA-Style Trading Dashboard API
Provides a FastAPI server with CallGraphManager integration.
Runs on port 8082 (trading_api.py uses 8081)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from pydantic import BaseModel
import asyncio
import os
import sys
import json
import subprocess
import logging
from datetime import datetime
import httpx

# Add base_agent to path
sys.path.insert(0, str(Path(__file__).parent))

from base_agent.src.storage.cell_repository import CellRepository
from base_agent.src.storage.models import Cell

logger = logging.getLogger(__name__)


# Pydantic models for callgraph data
class NodeData(BaseModel):
    """Node data for visualization."""
    id: str
    name: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    token_count: Optional[int]
    num_cached_tokens: Optional[int]
    cost: Optional[float]
    success: Optional[bool]
    events: List[Dict]
    children: List[str]


class CallGraphData(BaseModel):
    """Complete callgraph data for visualization."""
    nodes: Dict[str, NodeData]
    root_id: Optional[str]
    total_duration: Optional[float]
    total_tokens: Optional[int]
    num_cached_tokens: Optional[int]
    total_cost: Optional[float]

app = FastAPI(
    title="SICA Trading Dashboard API",
    description="SICA-style API with callgraph visualization",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
web_dir = Path(__file__).parent / "sica_web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

# Global state
repo: Optional[CellRepository] = None
active_websockets: Set[WebSocket] = set()
current_execution: Optional[Dict[str, Any]] = None


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
    """Serve the SICA-style dashboard HTML."""
    dashboard_path = Path(__file__).parent / "sica_web" / "index.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)

    return {
        "name": "SICA Trading Dashboard API",
        "version": "1.0.0",
        "port": 8082,
        "endpoints": [
            "/api/callgraph",
            "/api/command/trading-evolve",
            "/api/command/trading-learn",
            "/api/command/trading-test",
            "/api/command/query-cells",
        ]
    }


@app.get("/api/callgraph")
async def get_callgraph():
    """Proxy callgraph data from trading_api.py on port 8081.

    The original trading_api.py has the real CallGraphManager integration.
    This endpoint simply forwards the request to avoid code duplication.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8081/api/callgraph", timeout=5.0)
            if response.status_code == 200:
                return response.json()
            else:
                # Return empty callgraph if port 8081 is not responding
                return {
                    "nodes": {},
                    "root_id": None,
                    "total_duration": None,
                    "total_tokens": None,
                    "num_cached_tokens": None,
                    "total_cost": None,
                }
    except Exception as e:
        logger.warning(f"Failed to proxy callgraph from port 8081: {e}")
        # Return empty callgraph on error
        return {
            "nodes": {},
            "root_id": None,
            "total_duration": None,
            "total_tokens": None,
            "num_cached_tokens": None,
            "total_cost": None,
        }


@app.post("/api/command/trading-evolve")
async def run_trading_evolve(
    background_tasks: BackgroundTasks,
    generations: int = 1000,
    fitness_goal: float = 50.0
):
    """Execute trading-evolve command (offline evolution)."""
    global current_execution

    try:
        # Create execution record
        current_execution = {
            "command": "trading-evolve",
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "parameters": {
                "generations": generations,
                "fitness_goal": fitness_goal
            }
        }

        # Broadcast update
        await broadcast_event({
            "type": "command_started",
            "command": "trading-evolve",
            "params": current_execution["parameters"]
        })

        # TODO: Actually run the docker command in background
        # For now, return success

        return {
            "message": f"Trading-evolve started: {generations} generations, fitness goal {fitness_goal}",
            "execution_id": "evolve_1"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/command/trading-learn")
async def run_trading_learn(
    background_tasks: BackgroundTasks,
    num_strategies: int = 5,
    confidence: float = 1.0,
    use_local: bool = True
):
    """Execute trading-learn command (LLM-guided evolution)."""
    global current_execution

    try:
        current_execution = {
            "command": "trading-learn",
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "parameters": {
                "num_strategies": num_strategies,
                "confidence": confidence,
                "use_local": use_local
            }
        }

        await broadcast_event({
            "type": "command_started",
            "command": "trading-learn",
            "params": current_execution["parameters"]
        })

        return {
            "message": f"Trading-learn started: {num_strategies} strategies using {'local LLM' if use_local else 'cloud LLM'}",
            "execution_id": "learn_1"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/command/trading-test")
async def run_trading_test(
    background_tasks: BackgroundTasks,
    cell_id: Optional[int] = None
):
    """Execute trading-test command (backtest a strategy)."""
    global current_execution

    try:
        current_execution = {
            "command": "trading-test",
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "parameters": {
                "cell_id": cell_id
            }
        }

        await broadcast_event({
            "type": "command_started",
            "command": "trading-test",
            "params": current_execution["parameters"]
        })

        return {
            "message": f"Trading-test started for cell {cell_id}" if cell_id else "Trading-test started for best cell",
            "execution_id": "test_1"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/command/query-cells")
async def query_cells(
    limit: int = 20,
    min_trades: int = 0,
    has_llm: Optional[bool] = None
):
    """Query cells from database."""
    try:
        repo = get_repository()
        cells = repo.get_top_cells(limit=limit, min_trades=min_trades)

        result = []
        for cell in cells:
            if has_llm is not None:
                if has_llm and cell.llm_name is None:
                    continue
                elif not has_llm and cell.llm_name is not None:
                    continue

            phenotypes = repo.get_phenotypes(cell.cell_id)
            total_trades = sum(p.total_trades for p in phenotypes) if phenotypes else 0

            result.append({
                "cell_id": cell.cell_id,
                "generation": cell.generation,
                "fitness": cell.fitness,
                "total_trades": total_trades,
                "dsl_genome": cell.dsl_genome[:100] + "..." if len(cell.dsl_genome) > 100 else cell.dsl_genome,
                "llm_name": cell.llm_name,
            })

        return {
            "cells": result,
            "count": len(result)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get overall system statistics."""
    try:
        repo = get_repository()
        total_cells = repo.get_cell_count()
        top_cells = repo.get_top_cells(limit=1)

        best_fitness = top_cells[0].fitness if top_cells else 0.0

        # Calculate average fitness
        all_cells = repo.get_top_cells(limit=100)
        avg_fitness = sum(c.fitness for c in all_cells) / len(all_cells) if all_cells else 0.0

        return {
            "total_cells": total_cells,
            "best_fitness": round(best_fitness, 2),
            "avg_fitness": round(avg_fitness, 2),
            "active_execution": current_execution is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time event streaming."""
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)


async def broadcast_event(event_data: Dict[str, Any]):
    """Broadcast an event to all connected WebSocket clients."""
    message = json.dumps(event_data)
    disconnected = set()

    for websocket in active_websockets:
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"Error broadcasting to websocket: {e}")
            disconnected.add(websocket)

    for ws in disconnected:
        active_websockets.discard(ws)


@app.on_event("startup")
async def startup_event():
    """Startup tasks."""
    print("✓ SICA-style Trading Dashboard API started on port 8082")
    print("  Original dashboard (trading_api.py) runs on port 8081")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8082,
        log_level="info"
    )
