#!/usr/bin/env python3
"""
Trading Evolution System API
Provides a FastAPI server for the trading evolution system.
This runs INSIDE Docker and provides endpoints the web UI can call.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
import asyncio
import os
import sys
import json
import subprocess
import signal
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

# Active WebSocket connections for real-time event streaming
active_websockets: Set[WebSocket] = set()

# Global process tracker for trading-learn
current_process: Optional[subprocess.Popen] = None
process_output_file: Optional[Path] = None


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


@app.post("/llm/learn/start")
async def start_llm_learning(
    background_tasks: BackgroundTasks,
    num_strategies: int = 5,
    confidence: float = 1.0,
    use_local: bool = False
):
    """Start a trading-learn session to create new LLM-guided strategies.

    This now calls run_trading_learn directly with proper CallGraphManager integration.
    """
    try:
        # Import run_trading_learn from agent module
        from base_agent.agent import run_trading_learn

        # Set environment variables for LLM configuration
        if use_local:
            os.environ["USE_LOCAL_LLM"] = "true"
            os.environ["OLLAMA_HOST"] = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        # Run trading-learn as a background task with proper async integration
        # This maintains CallGraphManager and EventBus integration for real-time updates
        async def run_learn_task():
            try:
                await run_trading_learn(
                    iterations=num_strategies,
                    workdir=Path("/workdir"),  # Docker workdir
                    logdir=None,
                    server_enabled=True,  # Enable CallGraphManager/EventBus
                    cost_threshold=confidence * 10.0,  # Convert confidence to cost threshold
                )
            except Exception as e:
                print(f"Error in trading-learn: {e}")
                import traceback
                traceback.print_exc()

        # Start the task in the background
        background_tasks.add_task(run_learn_task)

        llm_type = "local LLM (Ollama)" if use_local else "cloud LLM"
        return {
            "message": f"LLM learning started for {num_strategies} strategies using {llm_type}",
            "parameters": {
                "num_strategies": num_strategies,
                "confidence": confidence,
                "use_local": use_local
            },
            "note": "Learning is running in background. Watch the System Activity log for updates."
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time callgraph event streaming."""
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        # Keep connection alive and listen for client messages
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

    # Remove disconnected websockets
    for ws in disconnected:
        active_websockets.discard(ws)


@app.post("/learn/run")
async def run_trading_learn():
    """Start trading-learn process with 100 iterations using local LLM."""
    global current_process, process_output_file

    try:
        # Check if already running
        if current_process and current_process.poll() is None:
            return {"status": "error", "message": "Trading-learn is already running"}

        # Create output file
        process_output_file = Path("/tmp") / f"trading_learn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Build docker command (matching your working command)
        cmd = [
            "docker", "run", "--rm", "--network", "host",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "USE_LOCAL_LLM=true",
            "-v", f"{Path.cwd()}/base_agent:/home/agent/agent_code",
            "-v", f"{Path.cwd()}/benchmark_data:/home/agent/benchmark_data",
            "-v", f"{Path.cwd()}/results/evolution_viz:/home/agent/workdir",
            "sica_sandbox",
            "python", "-m", "agent_code.agent", "trading-learn", "-n", "100"
        ]

        # Start process with output redirection
        with open(process_output_file, 'w') as f:
            current_process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Create new process group for easier killing
            )

        return {
            "status": "success",
            "message": "Trading-learn started with 100 iterations (local LLM)",
            "pid": current_process.pid,
            "output_file": str(process_output_file)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/learn/stop")
async def stop_trading_learn():
    """Stop the running trading-learn process."""
    global current_process

    try:
        if not current_process or current_process.poll() is not None:
            return {"status": "error", "message": "No trading-learn process is running"}

        # Kill the entire process group (docker and its children)
        try:
            os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
        except:
            current_process.terminate()

        # Wait for process to actually stop
        try:
            current_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(current_process.pid), signal.SIGKILL)

        current_process = None

        return {"status": "success", "message": "Trading-learn process stopped"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/learn/status")
async def get_learn_status():
    """Get status of trading-learn process."""
    global current_process

    if not current_process:
        return {"running": False, "pid": None}

    poll_result = current_process.poll()
    is_running = poll_result is None

    return {
        "running": is_running,
        "pid": current_process.pid if is_running else None,
        "exit_code": poll_result if not is_running else None
    }


@app.get("/learn/output")
async def get_learn_output(lines: int = 50):
    """Get recent output from trading-learn process."""
    global process_output_file

    try:
        if not process_output_file or not process_output_file.exists():
            return {"output": "No output file available"}

        # Read last N lines using tail
        result = subprocess.run(
            ["tail", "-n", str(lines), str(process_output_file)],
            capture_output=True,
            text=True
        )

        return {"output": result.stdout, "file": str(process_output_file)}
    except Exception as e:
        return {"output": f"Error reading output: {e}"}


@app.on_event("startup")
async def startup_event():
    """Subscribe to EventBus on startup to stream events to WebSocket clients."""
    try:
        from base_agent.src.events import EventBus
        from base_agent.src.types.event_types import EventType, Event

        event_bus = await EventBus.get_instance()

        # Define event callback
        async def event_callback(event: Event):
            """Forward events to WebSocket clients."""
            try:
                event_data = {
                    "type": event.type.value if hasattr(event.type, "value") else str(event.type),
                    "content": str(event.content) if event.content else "",
                    "timestamp": event.timestamp.isoformat() if event.timestamp else datetime.now().isoformat(),
                    "metadata": event.metadata if event.metadata else {}
                }
                await broadcast_event(event_data)
            except Exception as e:
                print(f"Error processing event: {e}")

        # Subscribe to all event types
        event_bus.subscribe(set(EventType), event_callback)
        print("✓ Subscribed to EventBus for real-time event streaming")
    except Exception as e:
        print(f"Warning: Could not subscribe to EventBus: {e}")


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8081,
        log_level="info"
    )