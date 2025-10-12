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


def get_repository(dish_name: Optional[str] = None) -> CellRepository:
    """Get or create repository instance.

    Args:
        dish_name: Optional dish name to load specific dish database
    """
    global repo

    # If dish_name is specified, always load that specific dish
    if dish_name:
        from base_agent.src.dish_manager import DishManager
        dm = DishManager(Path("experiments"))
        try:
            dish_path, config = dm.load_dish(dish_name)
            db_path = dish_path / "evolution" / "cells.db"
            return CellRepository(db_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Dish '{dish_name}' not found")

    # Otherwise, use the cached global repo or find most recent
    if repo is None:
        # Find the most recent cells.db
        db_paths = []

        # First check experiments directory (new dish architecture)
        experiments_dir = Path("experiments")
        if experiments_dir.exists():
            all_dbs = list(experiments_dir.glob("**/cells.db"))
            all_dbs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            db_paths.extend(all_dbs)

        # Then check results directory (legacy)
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
async def get_top_cells(
    limit: int = 10,
    min_trades: int = 0,
    has_llm: Optional[bool] = None,
    dish: Optional[str] = None
):
    """Get top cells by fitness.

    Args:
        limit: Maximum number of cells to return
        min_trades: Minimum number of trades required
        has_llm: Filter by LLM involvement (True=LLM only, False=evolution only, None=all)
        dish: Optional dish name to filter by
    """
    try:
        repo = get_repository(dish_name=dish)
        cells = repo.get_top_cells(limit=limit, min_trades=min_trades, dish_name=dish)

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
                "cell_name": cell.cell_name,
                "dish_name": cell.dish_name,
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
async def get_cell(cell_id: int, dish: Optional[str] = None):
    """Get a specific cell by ID."""
    try:
        repo = get_repository(dish_name=dish)
        cell = repo.get_cell(cell_id)

        if not cell:
            raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found")

        return cell.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cell/{cell_id}/phenotypes")
async def get_cell_phenotypes(cell_id: int, dish: Optional[str] = None):
    """Get phenotypes for a cell."""
    try:
        repo = get_repository(dish_name=dish)
        phenotypes = repo.get_phenotypes(cell_id)

        return {
            "cell_id": cell_id,
            "phenotypes": [p.to_dict() for p in phenotypes],
            "count": len(phenotypes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cell/{cell_id}/lineage")
async def get_cell_lineage(cell_id: int, dish: Optional[str] = None):
    """Get lineage (ancestry) for a cell."""
    try:
        repo = get_repository(dish_name=dish)
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


from pydantic import BaseModel

class LearnRequest(BaseModel):
    dish: Optional[str] = None
    iterations: int = 30

@app.post("/learn/run")
async def run_trading_learn(request: LearnRequest):
    """Start trading-learn process with local LLM on a specific dish.

    Args:
        request: LearnRequest with dish and iterations
    """
    global current_process, process_output_file

    try:
        # Check if already running
        if current_process and current_process.poll() is None:
            return {"status": "error", "message": "Trading-learn is already running"}

        # Create output file
        process_output_file = Path("/tmp") / f"trading_learn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Build command to run trading-learn directly on host
        cmd = ["./trade", "learn", "-n", str(request.iterations), "--use-local-llm"]

        # Add dish parameter if specified
        if request.dish:
            cmd.extend(["--dish", request.dish])

        # Start process with output redirection
        with open(process_output_file, 'w') as f:
            current_process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # Create new process group for easier killing
                cwd=str(Path.cwd())  # Run from project root
            )

        dish_msg = f" on dish '{dish}'" if dish else ""
        return {
            "status": "success",
            "message": f"Trading-learn started with {iterations} iterations (local LLM){dish_msg}",
            "pid": current_process.pid,
            "dish": dish,
            "iterations": iterations,
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


@app.get("/dishes")
async def list_dishes():
    """List all experiment dishes."""
    try:
        from base_agent.src.dish_manager import DishManager
        dm = DishManager(Path("experiments"))
        dishes = dm.list_dishes()

        return {
            "dishes": dishes,
            "count": len(dishes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dish/{dish_name}")
async def get_dish(dish_name: str):
    """Get details for a specific dish."""
    try:
        from base_agent.src.dish_manager import DishManager
        dm = DishManager(Path("experiments"))
        dish_path, config = dm.load_dish(dish_name)

        # Get additional stats from database
        db_path = dish_path / "evolution" / "cells.db"
        if db_path.exists():
            repo = CellRepository(db_path)
            top_cells = repo.get_top_cells(limit=5, dish_name=dish_name)

            config["top_cells"] = [
                {
                    "cell_id": cell.cell_id,
                    "cell_name": cell.cell_name,
                    "generation": cell.generation,
                    "fitness": cell.fitness,
                    "dsl_genome": cell.dsl_genome
                }
                for cell in top_cells
            ]

        return config
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dish '{dish_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dish/{dish_name}/summary")
async def get_dish_summary(dish_name: str):
    """Get summary statistics for a dish."""
    try:
        repo = get_repository(dish_name=dish_name)

        total_cells = repo.get_cell_count()
        max_gen = repo.get_max_generation(dish_name)

        summary = {
            "dish_name": dish_name,
            "total_cells": total_cells,
            "max_generation": max_gen,
            "best_fitness": None,
            "best_cell": None
        }

        if total_cells > 0:
            top_cells = repo.get_top_cells(limit=1, dish_name=dish_name)
            if top_cells:
                best_cell = top_cells[0]
                summary["best_fitness"] = best_cell.fitness
                summary["best_cell"] = {
                    "cell_id": best_cell.cell_id,
                    "cell_name": best_cell.cell_name,
                    "generation": best_cell.generation,
                    "dsl_genome": best_cell.dsl_genome
                }

        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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