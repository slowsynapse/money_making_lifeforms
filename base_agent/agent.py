# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
The main entrypoint to the system.
"""

import time
import signal
import asyncio
import logging

from pathlib import Path

from .src.events import EventBus
from .src.web_server import run_server
from .src.llm.metering import get_total_cost, get_total_usage, budget_info
from .src.callgraph.manager import CallGraphManager
from .src.callgraph.reporting import generate_execution_report, generate_execution_tree
from .src.oversight.overseer import Overseer
from .src.agents.agent_calling import await_agent_task
from .src.agents.implementations.main_orchestrator import MainOrchestratorAgent
from .src.events.event_bus_utils import log_to_stdout
from .src.config import settings
from .src.types.event_types import EventType, Event
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


async def cost_monitor(cost_threshold: float, shutdown_event: asyncio.Event):
    """Monitor the cost and trigger shutdown if threshold is exceeded, with notifications at thresholds"""
    # Notification thresholds as percentage of remaining budget
    thresholds = [0.50, 0.80, 0.90, 0.95]  # 50%, 20%, 10%, 5% remaining
    triggered_thresholds = set()

    # Get the event bus for sending notifications
    event_bus = await EventBus.get_instance()
    callgraph = await CallGraphManager.get_instance()

    # Start time for duration calculation
    start_time = time.time()

    while not shutdown_event.is_set():
        current_cost = get_total_cost()

        # Check for budget thresholds
        if cost_threshold > 0:
            usage_ratio = current_cost / cost_threshold
            remaining_percentage = 100 * (1 - usage_ratio)

            # Check each threshold
            for threshold in thresholds:
                if usage_ratio >= threshold and threshold not in triggered_thresholds:
                    triggered_thresholds.add(threshold)
                    remaining_budget = cost_threshold - current_cost
                    remaining_percent = 100 * (1 - threshold)

                    # Get main agent ID from callgraph
                    callgraph = await CallGraphManager.get_instance()
                    root_node = callgraph.graph.root
                    if root_node is not None:
                        main_agent_id = root_node.id
                    else:
                        main_agent_id = "root"

                    # Send notification
                    # logger.warning(f"Budget alert: {remaining_percent:.1f}% of cost budget remaining (${remaining_budget:.4f} of ${cost_threshold:.4f})")
                    print(
                        f"Budget alert: {remaining_percent:.1f}% of cost budget remaining (${remaining_budget:.4f} of ${cost_threshold:.4f})"
                    )

                    await event_bus.publish(
                        Event(
                            type=EventType.APPLICATION_WARNING,
                            content=f"BUDGET ALERT: You have approximately {remaining_percent:.1f}% of cost budget remaining (${remaining_budget:.4f} of ${cost_threshold:.4f})",
                            metadata={
                                "warning_type": "cost_budget",
                                "threshold": threshold,
                                "remaining_budget": remaining_budget,
                                "total_budget": cost_threshold,
                                "remaining_percentage": remaining_percent,
                                "elapsed_time": time.time() - start_time,
                            },
                        ),
                        callgraph.current_function_id or main_agent_id,
                    )

        # If threshold exceeded, trigger shutdown
        if current_cost >= cost_threshold:
            logger.warning(
                f"Cost threshold of ${cost_threshold:.2f} exceeded (current: ${current_cost:.2f})"
            )
            shutdown_event.set()
            return

        await asyncio.sleep(1)


async def time_monitor(timeout: int, start_time: float, shutdown_event: asyncio.Event):
    """Monitor the time and trigger shutdown if timeout is exceeded, with notifications at thresholds"""
    # Notification thresholds as percentage of remaining time
    thresholds = [0.50, 0.80, 0.90, 0.95]  # 50%, 20%, 10%, 5% remaining
    triggered_thresholds = set()

    # Get the event bus for sending notifications
    event_bus = await EventBus.get_instance()
    callgraph = await CallGraphManager.get_instance()

    while not shutdown_event.is_set():
        elapsed_time = time.time() - start_time

        # Check for time thresholds
        if timeout > 0:
            usage_ratio = elapsed_time / timeout

            # Check each threshold
            for threshold in thresholds:
                if usage_ratio >= threshold and threshold not in triggered_thresholds:
                    triggered_thresholds.add(threshold)
                    remaining_time = timeout - elapsed_time
                    remaining_percent = 100 * (1 - threshold)

                    # Get main agent ID from callgraph
                    callgraph = await CallGraphManager.get_instance()
                    root_node = callgraph.graph.root
                    if root_node is not None:
                        main_agent_id = root_node.id
                    else:
                        main_agent_id = "root"

                    # Send notification
                    logger.warning(
                        f"Time alert: {remaining_percent:.1f}% of time budget remaining ({remaining_time:.1f}s of {timeout}s)"
                    )

                    await event_bus.publish(
                        Event(
                            type=EventType.APPLICATION_WARNING,
                            content=f"TIME ALERT: You have approximately {remaining_percent:.1f}% of time budget remaining ({remaining_time:.1f}s of {timeout}s)",
                            metadata={
                                "warning_type": "time_budget",
                                "threshold": threshold,
                                "remaining_time": remaining_time,
                                "total_time": timeout,
                                "remaining_percentage": remaining_percent,
                                "elapsed_cost": get_total_cost(),
                            },
                        ),
                        callgraph.current_function_id or main_agent_id,
                    )

        # If threshold exceeded, trigger shutdown
        if elapsed_time >= timeout:
            logger.warning(
                f"Time threshold of {timeout}s exceeded (elapsed: {elapsed_time:.2f}s)"
            )
            shutdown_event.set()
            return

        await asyncio.sleep(1)


class Agent:
    """
    The Agent class acts as the 'root' of the application state. Since no state
    is persisted between different 'exec' calls, this class could just as
    easily be a function.
    """

    def __init__(
        self,
        workdir: Path = Path.home() / "workdir",
        logdir: Path | None = None,
        server_enabled: bool = True,
        debug_mode: bool = False,
    ):
        # Setup directories
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

        self.logdir = logdir if logdir else workdir / "agent_outputs"
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.server_enabled = server_enabled
        self.debug_mode = debug_mode

        self._shutdown_event = asyncio.Event()
        self._main_task: asyncio.Task | None = None
        self._register_signal_handlers()

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                logger.debug(f"Agent registering handler for {sig.name}")
                loop.add_signal_handler(
                    sig, lambda s=sig: asyncio.create_task(self._signal_handler(s))
                )
            logger.debug("Agent signal handlers registered successfully")
        except Exception as e:
            logger.error(f"Error registering agent signal handlers: {e}")

    async def _signal_handler(self, sig: signal.Signals):
        """Handle shutdown signals gracefully"""
        try:
            if self._shutdown_event.is_set():
                # If we get a second signal, force immediate stop
                logger.warning("Forced shutdown requested")
                # Get the current event loop and stop it
                loop = asyncio.get_running_loop()
                loop.stop()
                return

            logger.info(f"Agent received signal {sig.name}, shutting down...")
            self._shutdown_event.set()

            # Cancel main execution task if running
            if self._main_task and not self._main_task.done():
                self._main_task.cancel()
                try:
                    await self._main_task
                except asyncio.CancelledError:
                    pass

            logger.info("Agent shutdown complete")
        except Exception as e:
            logger.error(f"Error during agent signal handling shutdown: {e}")
            # If we hit an error during shutdown, stop the loop
            loop = asyncio.get_running_loop()
            loop.stop()

    async def exec(
        self,
        problem: str,
        timeout: int | None = None,
        cost_threshold: float | None = None,
    ) -> tuple[int, int, float, float]:
        """
        Starts the main agent off on a problem statement.

        Before starting the agent, this function records the start time, sets
        of the observability web server (if enabled) and is also responsible
        for starting and keeping track of the asyncio task in which the
        asynchronous overseer is running.

        After the main agent returns, the overseer and server are stopped
        before this function returns.

        Args:
            problem: The problem statement or request, formatted as a string.
            timeout: (optional) the time, in seconds, before the agent execution is cancelled
            cost_threshold: (optional) the cost threshoold, in USD, before the agent execution is cancelled

        Returns:
            Tuple of (token_count, of_which_cached, total_cost, duration_seconds)
        """
        exec_start = time.time()
        overseer = Overseer(model=settings.OVERSIGHT_MODEL)
        overseer.start()

        # Add logging to all events
        event_bus = await EventBus.get_instance()
        event_bus.subscribe(set(EventType), log_to_stdout)

        # Start the visualisation of the oversight
        if self.server_enabled:
            self.web_server_task = asyncio.create_task(run_server())

        try:
            # Create the main agent instance
            main = MainOrchestratorAgent(
                workdir=self.workdir, logdir=self.logdir, debug_mode=self.debug_mode
            )

            # Publish the initial problem statement
            await event_bus.publish(
                Event(
                    type=EventType.PROBLEM_STATEMENT,
                    content=problem,
                ),
                main._id,
            )
            budget_info["start_time"] = time.time()
            budget_info["cost_budget"] = cost_threshold
            budget_info["time_budget"] = timeout

            # Register the agent with the callgraph
            callgraph = await CallGraphManager.get_instance()
            await callgraph.start_agent(
                agent_name=main.AGENT_NAME,
                node_id=main._id,
                args=main.model_dump(),
            )

            # Create and store the main execution task
            self._main_task = asyncio.create_task(main.execute())
            await callgraph.register_agent_task(main._id, self._main_task)

            # Create shutdown wait task
            shutdown_task = asyncio.create_task(self._shutdown_event.wait())

            # Initialise task list with main and shutdown tasks
            task_list = [self._main_task, shutdown_task]

            # If given a timeout, create a timeout task
            time_monitor_task = None
            if timeout:
                time_monitor_task = asyncio.create_task(
                    time_monitor(timeout, exec_start, self._shutdown_event)
                )
                task_list.append(time_monitor_task)

            # If given a cost threshold, create a cost monitor task
            cost_monitor_task = None
            if cost_threshold:
                cost_monitor_task = asyncio.create_task(
                    cost_monitor(cost_threshold, self._shutdown_event)
                )
                task_list.append(cost_monitor_task)

            # Wait for either completion or shutdown
            done, pending = await asyncio.wait(
                task_list, return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await callgraph.cancel_all_agents("Main agent timeout")
                    await task
                except asyncio.CancelledError:
                    pass

            # Detect reason for termination
            if shutdown_task in done:
                if time_monitor_task in task_list and time_monitor_task not in pending:
                    # Time threshold was exceeded
                    elapsed_time = time.time() - exec_start
                    logger.warning(
                        f"Execution cancelled due to time threshold ({timeout}s) being exceeded (elapsed: {elapsed_time:.2f}s)"
                    )
                    await event_bus.publish(
                        Event(
                            type=EventType.TIMEOUT,
                            content=f"Agent time threshold of {timeout}s exceeded (elapsed: {elapsed_time:.2f}s)",
                            metadata={"timeout": timeout, "elapsed_time": elapsed_time},
                        ),
                        main._id,
                    )
                if cost_monitor_task in task_list and cost_monitor_task not in pending:
                    # Cost threshold was exceeded
                    logger.warning(
                        f"Execution cancelled due to cost threshold (${cost_threshold:.2f}) being exceeded"
                    )
                    await event_bus.publish(
                        Event(
                            type=EventType.COST_LIMIT,
                            content=f"Agent cost threshold of ${cost_threshold:.2f} exceeded (current: ${get_total_cost():.2f})",
                            metadata={
                                "cost_threshold": cost_threshold,
                                "current_cost": get_total_cost(),
                            },
                        ),
                        main._id,
                    )
                else:
                    # Manual cancellation
                    raise KeyboardInterrupt("Execution cancelled")

            await await_agent_task(self._main_task, main)

            # Print the global token usage informatio
            usage = get_total_usage()
            cost = get_total_cost()
            return (
                usage.total_tokens,
                usage.cached_prompt_tokens,
                cost,
                time.time() - exec_start,
            )

        except KeyboardInterrupt:
            logger.info("Execution cancelled by user")
            raise
        except Exception as e:
            logger.error(f"Error in execution: {str(e)}", exc_info=True)
            raise

        finally:
            overseer.stop()

            # Remove the logging callback
            event_bus.unsubscribe(set(EventType), log_to_stdout)

            # To allow last events to show up in web interface
            await asyncio.sleep(5)

            # await event_bus.save_state(self.logdir)

            # Cancel the web server task
            if self.server_enabled and self.web_server_task:
                self.web_server_task.cancel()
                try:
                    await self.web_server_task
                except asyncio.CancelledError:
                    pass
                self.web_server_task = None

    async def create_report(self):
        """
        Generates the trace.txt and execution_tree.txt reports from the global
        call graph manager instance. This must only be called after `exec` has
        run.
        """
        cg = await CallGraphManager.get_instance()
        report = await generate_execution_report(cg)
        with open(self.logdir / "trace.txt", "w") as f:
            f.write(report)
        execution_tree = await generate_execution_tree(
            cg, truncate_assistant_events=500, include_all_events=True
        )
        with open(self.logdir / "execution_tree.txt", "w") as f:
            f.write(execution_tree)


async def run_trading_demo():
    """Run the trading evolution demo"""
    print("\n" + "="*60)
    print("TRADING EVOLUTION SYSTEM - DEMO")
    print("="*60)
    print("\nLoading trading components...")
    
    try:
        from .src.benchmarks.trading_benchmarks.trading_benchmark import TradingBenchmark
        from .src.dsl.interpreter import DslInterpreter
        from .src.dsl.mutator import DslMutator
        
        interpreter = DslInterpreter()
        mutator = DslMutator()
        benchmark = TradingBenchmark()
        
        print("✓ Trading benchmark loaded")
        print("✓ DSL interpreter initialized")
        print("✓ DSL mutator initialized")
        
        # Demo 1: Parse some strategies
        print("\n" + "="*60)
        print("1. DSL PARSING")
        print("="*60)
        strategies = [
            "IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL",
            "IF GAMMA(14) < PSI() THEN BUY ELSE HOLD",
            "IF OMEGA() >= EPSILON(100) THEN HOLD ELSE SELL",
        ]
        for strategy in strategies:
            program = interpreter.parse(strategy)
            status = "✓" if program else "✗"
            print(f"{status} {strategy}")
        
        # Demo 2: Show mutations
        print("\n" + "="*60)
        print("2. DSL MUTATION")
        print("="*60)
        base = "IF DELTA(20) > ZETA(50) THEN BUY ELSE SELL"
        print(f"Base: {base}")
        print("\nMutations:")
        program = interpreter.parse(base)
        for i in range(3):
            mutated = mutator.mutate(program)
            print(f"  {i+1}. {interpreter.to_string(mutated)}")
            program = interpreter.parse(base)  # Reset for next mutation
        
        # Demo 3: Explain the system
        print("\n" + "="*60)
        print("3. EVOLUTIONARY CONCEPT")
        print("="*60)
        print("""
The system evolves trading strategies through natural selection:

1. Generate a strategy (DSL string)
2. Backtest on historical data
3. Calculate fitness: Profit - Costs
4. If fitness > 0: Strategy survives
5. Mutate the best survivor
6. Repeat

Over many generations, profitable patterns emerge without
any human bias about what "should" work in markets.

Symbols (ALPHA, BETA, etc.) are abstract placeholders.
The market determines which combinations are profitable.
""")
        
        print("="*60)
        print("DEMO COMPLETE!")
        print("="*60)
        print("\nTo test a specific strategy:")
        print('  python -m agent_code.agent trading-test --strategy "IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL"')
        print("\nTo run the full evolution loop (requires runner.py on host):")
        print("  python3 runner.py --evolution-mode --iterations 10")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def run_trading_test(strategy_dsl: str, output_dir: str | None):
    """Test a DSL strategy through a backtest"""
    import tempfile
    import pandas as pd
    
    print("\n" + "="*60)
    print("TRADING STRATEGY BACKTEST")
    print("="*60)
    print(f"\nStrategy: {strategy_dsl}")
    
    try:
        from .src.benchmarks.trading_benchmarks.trading_benchmark import TradingBenchmark
        from .src.benchmarks.base import Problem
        
        benchmark = TradingBenchmark()
        problem = benchmark.problems[0]
        
        # Create temporary directories for the test
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            workdir = tmppath / "workdir"
            answer_dir = tmppath / "answer"
            workdir.mkdir()
            answer_dir.mkdir()
            
            # Write the strategy to answer.txt
            answer_file = answer_dir / "answer.txt"
            answer_file.write_text(strategy_dsl + "\n")
            
            # Setup the problem (copies OHLCV data)
            await benchmark.setup_problem(problem, workdir, "test_container")
            
            # Run the backtest
            print("\nRunning backtest...")
            score, error, discussion = await benchmark.score_problem(
                problem, str(workdir), str(answer_dir), "test_container"
            )
            
            # Display results
            print("\n" + "="*60)
            print("RESULTS")
            print("="*60)
            if error:
                print(f"❌ Error: {error}")
            else:
                print(f"Fitness Score: ${score:.2f}")
                print(f"\nDetails:\n{discussion}")
                
                if score > 0:
                    print(f"\n✓ Strategy SURVIVED (fitness > 0)")
                else:
                    print(f"\n✗ Strategy DIED (fitness ≤ 0)")
            
            # Optionally save results
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                result_file = output_path / "backtest_result.txt"
                with open(result_file, 'w') as f:
                    f.write(f"Strategy: {strategy_dsl}\n")
                    f.write(f"Score: ${score:.2f}\n")
                    f.write(f"Details: {discussion}\n")
                print(f"\n✓ Results saved to: {result_file}")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def run_trading_learn(
    iterations: int,
    workdir: Path,
    logdir: Path | None,
    server_enabled: bool,
    cost_threshold: float | None,
):
    """
    LLM-Guided Evolution: Intelligent mutation of existing cell library.

    This mode requires a cell database from a prior `trading-evolve` run.
    It uses the LLM to:
    1. Analyze top cells and discover patterns
    2. Propose intelligent mutations based on pattern analysis
    3. Test proposed mutations and birth new cells

    This is 100% LLM-guided (no random mutations), designed for exploitation
    of discovered patterns after exploration via trading-evolve.

    Args:
        iterations: Number of LLM-guided mutation iterations
        workdir: Working directory (expects cells.db in workdir/evolution/)
        logdir: Optional log directory
        server_enabled: Whether to run the web server
        cost_threshold: Optional cost limit for LLM usage
    """
    print("\n" + "="*70)
    print("TRADING STRATEGY LEARNING MODE (LLM-Guided Evolution)")
    print("="*70)
    print(f"\n🧠 LLM will analyze cells and propose {iterations} intelligent mutations.")
    print("💡 Requires cell database from prior trading-evolve run.\n")

    # Start web server if enabled
    web_server_task = None
    if server_enabled:
        print(f"🌐 Starting web visualization server on http://localhost:8080")
        web_server_task = asyncio.create_task(run_server())

    try:
        from .src.benchmarks.trading_benchmarks.trading_benchmark import TradingBenchmark
        from .src.dsl.interpreter import DslInterpreter
        from .src.storage.cell_repository import CellRepository
        from .src.data.hyperliquid_fetcher import HyperliquidDataFetcher
        from .src.analysis import (
            analyze_cells_in_batches,
            merge_pattern_discoveries,
            propose_intelligent_mutation,
        )

        # Initialize EventBus for web UI updates
        event_bus = await EventBus.get_instance() if server_enabled else None

        # Create root callgraph node for learn run (enables web UI event display)
        callgraph = None
        if server_enabled:
            callgraph = await CallGraphManager.get_instance()
            await callgraph.start_agent(
                agent_name="trading-learn",
                node_id="trading-learn",
                args={"iterations": iterations, "cost_threshold": cost_threshold},
            )

        # Create results directory
        results_dir = workdir / "evolution"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Load cell database
        db_path = results_dir / "cells.db"
        if not db_path.exists():
            print(f"\n❌ Error: Cell database not found at {db_path}")
            print(f"   Please run 'trading-evolve' first to build a cell library.")
            return

        repo = CellRepository(db_path)
        print(f"📊 Loaded cell database: {db_path}")

        # Check if database has cells
        total_cells = repo.get_cell_count()
        if total_cells == 0:
            print(f"\n❌ Error: Cell database is empty (0 cells found)")
            print(f"   Please run 'trading-evolve' first to populate the database.")
            return

        print(f"✓ Found {total_cells} cells in database")

        # Initialize components
        benchmark = TradingBenchmark()
        interpreter = DslInterpreter()

        # Start evolution run tracking for this learn session
        run_id = repo.start_evolution_run(
            run_type='trading-learn',
            max_generations=iterations,
            fitness_goal=None,  # No specific goal for learn mode
            symbol='PURR',
            timeframe='multi',
            initial_capital=100.0,
            transaction_fee_rate=0.00045,
        )
        print(f"🧬 LLM-guided run #{run_id} started")

        # Fetch multi-timeframe data for testing
        print(f"\n📈 Fetching multi-timeframe market data...")
        fetcher = HyperliquidDataFetcher()
        multi_tf_data = fetcher.fetch_multi_timeframe('PURR', lookback_days=30)

        # PHASE 1: Pattern Discovery
        print(f"\n{'='*70}")
        print(f"PHASE 1: PATTERN DISCOVERY")
        print(f"{'='*70}")

        # Get top cells for analysis (up to 100 cells, batch size 30 for 8K context)
        top_cells = repo.get_top_cells(limit=100, status='online')
        if not top_cells:
            print(f"❌ No online cells found - cannot analyze patterns")
            return

        print(f"📊 Analyzing top {len(top_cells)} cells for pattern discovery...")
        cell_ids = [cell.cell_id for cell in top_cells]

        # Publish analysis start event
        if event_bus:
            await event_bus.publish(
                Event(
                    type=EventType.CELL_ANALYSIS_START,
                    content=f"Loading {len(top_cells)} cells for pattern analysis..."
                ),
                "trading-learn"
            )

        # Analyze cells in batches (30 cells per batch for Gemma 8K context)
        batch_results = await analyze_cells_in_batches(
            repo=repo,
            cell_ids=cell_ids,
            batch_size=30,
            use_json=True,
        )

        # Merge and deduplicate patterns
        patterns = await merge_pattern_discoveries(batch_results)

        print(f"\n✓ Pattern discovery complete!")
        print(f"   Unique patterns identified: {len(patterns['patterns'])}")

        # Display discovered patterns
        if patterns['patterns']:
            print(f"\n🔍 DISCOVERED PATTERNS:")
            for pattern in patterns['patterns'][:5]:  # Show top 5
                print(f"   - {pattern['pattern_name']} ({pattern['pattern_category']})")
                print(f"     {pattern['explanation']}")
                print(f"     Used by {pattern['num_cells']} cells")
                print()

                # Publish pattern discovery event
                if event_bus:
                    await event_bus.publish(
                        Event(
                            type=EventType.PATTERN_DISCOVERED,
                            content=f"Pattern: {pattern['pattern_name']} ({pattern['pattern_category']})\n{pattern['explanation']}\nUsed by {pattern['num_cells']} cells"
                        ),
                        "trading-learn"
                    )

        # PHASE 2: Intelligent Mutation Loop
        print(f"\n{'='*70}")
        print(f"PHASE 2: INTELLIGENT MUTATION ({iterations} iterations)")
        print(f"{'='*70}")

        # Track statistics
        cells_birthed = 0
        mutations_failed = 0
        total_cost = 0.0
        history = []
        best_fitness = repo.get_top_cells(limit=1, status='online')[0].fitness if top_cells else float('-inf')
        best_strategy = None
        best_cell_id = None

        for iteration in range(iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{iterations}")
            print(f"{'='*70}")

            # Publish iteration progress
            if event_bus:
                await event_bus.publish(
                    Event(
                        type=EventType.CELL_ANALYSIS_PROGRESS,
                        content=f"Iteration {iteration + 1}/{iterations}: Generating intelligent mutation..."
                    ),
                    "trading-learn"
                )

            # Check cost threshold
            if cost_threshold and total_cost >= cost_threshold:
                print(f"\n⚠️  Cost threshold reached: ${total_cost:.4f} >= ${cost_threshold:.4f}")
                print(f"Terminating early to stay within budget")
                break

            # Select parent cell (best cell)
            parent_cell = repo.get_top_cells(limit=1, status='online')[0]
            print(f"\n📍 Parent: Cell #{parent_cell.cell_id} (${parent_cell.fitness:.2f})")
            print(f"   Strategy: {parent_cell.dsl_genome}")

            # LLM proposes intelligent mutation
            print(f"\n🧠 LLM analyzing cell and proposing mutation...")

            proposal = await propose_intelligent_mutation(
                cell=parent_cell,
                patterns=patterns,
                repo=repo,
                use_json=True,
            )

            # Track LLM cost (rough estimate: ~0.0001 per proposal)
            iteration_cost = 0.0001
            total_cost += iteration_cost

            if not proposal:
                print(f"  ❌ LLM failed to propose valid mutation")
                mutations_failed += 1
                history.append({
                    'iteration': iteration + 1,
                    'parent_cell_id': parent_cell.cell_id,
                    'proposed_strategy': None,
                    'fitness': None,
                    'survived': False,
                    'llm_cost': iteration_cost,
                    'status': 'proposal_failed',
                })
                continue

            proposed_strategy = proposal['proposed_strategy']
            rationale = proposal.get('rationale', 'N/A')
            confidence = proposal.get('confidence', 'unknown')

            print(f"\n✅ Mutation proposed:")
            print(f"   Strategy: {proposed_strategy}")
            print(f"   Rationale: {rationale[:100]}...")
            print(f"   Confidence: {confidence}")

            # Publish mutation proposal event
            if event_bus:
                await event_bus.publish(
                    Event(
                        type=EventType.MUTATION_PROPOSED,
                        content=f"Cell #{parent_cell.cell_id}: {proposed_strategy}\nRationale: {rationale[:150]}...\nConfidence: {confidence}"
                    ),
                    "trading-learn"
                )

            # Parse and validate
            program = interpreter.parse(proposed_strategy)
            if not program:
                print(f"  ❌ Proposed strategy failed to parse")
                mutations_failed += 1
                history.append({
                    'iteration': iteration + 1,
                    'parent_cell_id': parent_cell.cell_id,
                    'proposed_strategy': proposed_strategy,
                    'fitness': None,
                    'survived': False,
                    'llm_cost': iteration_cost,
                    'status': 'parse_failed',
                })
                continue

            # Test on multi-timeframe backtest
            print(f"\n⚙️  Running multi-timeframe backtest...")

            fitness, phenotypes, backtest_log = benchmark.run_multi_timeframe_backtest(
                program, multi_tf_data, 'PURR'
            )

            print(f"{backtest_log}")

            # Publish backtest result event
            if event_bus:
                await event_bus.publish(
                    Event(
                        type=EventType.MUTATION_TESTED,
                        content=f"Fitness: ${fitness:.2f} | Parent: ${parent_cell.fitness:.2f}\n{backtest_log}"
                    ),
                    "trading-learn"
                )

            # Check if strategy survived
            survived = any(p.total_profit > -100.0 for p in phenotypes.values())

            # Determine if we should birth the cell
            is_improvement = fitness > parent_cell.fitness

            if is_improvement:
                print(f"\n✓ IMPROVEMENT! ${parent_cell.fitness:.2f} → ${fitness:.2f}")

                # Birth child cell
                child_cell_id = repo.birth_cell(
                    generation=parent_cell.generation + 1,
                    parent_cell_id=parent_cell.cell_id,
                    dsl_genome=proposed_strategy,
                    fitness=fitness,
                    status='online',
                )
                cells_birthed += 1

                # Store phenotypes
                for tf, phenotype in phenotypes.items():
                    phenotype.cell_id = child_cell_id
                    repo.store_phenotype(phenotype)

                # Store mutation proposal in database
                repo.store_mutation_proposal(
                    cell_id=child_cell_id,
                    proposed_genome=proposed_strategy,
                    rationale=rationale,
                    confidence=confidence,
                    expected_improvement=proposal.get('expected_improvement', ''),
                    actual_fitness_change=fitness - parent_cell.fitness,
                )

                print(f"🧬 Cell #{child_cell_id} birthed (Gen {parent_cell.generation + 1}, LLM-guided)")

                # Publish cell birth event
                if event_bus:
                    await event_bus.publish(
                        Event(
                            type=EventType.CELL_BIRTHED,
                            content=f"Cell #{child_cell_id} birthed (Gen {parent_cell.generation + 1}, +${fitness - parent_cell.fitness:.2f})"
                        ),
                        "trading-learn"
                    )

                # Track best
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_strategy = proposed_strategy
                    best_cell_id = child_cell_id
                    print(f"🏆 NEW BEST! Fitness: ${best_fitness:.2f}")

                history.append({
                    'iteration': iteration + 1,
                    'parent_cell_id': parent_cell.cell_id,
                    'child_cell_id': child_cell_id,
                    'proposed_strategy': proposed_strategy,
                    'fitness': fitness,
                    'improvement': fitness - parent_cell.fitness,
                    'survived': True,
                    'llm_cost': iteration_cost,
                    'status': 'success',
                })

            else:
                print(f"\n→ No improvement. ${fitness:.2f} <= ${parent_cell.fitness:.2f}")

                # Record failed mutation
                failure_reason = 'lower_than_parent' if survived else 'negative_fitness'
                repo.record_failed_mutation(
                    parent_cell_id=parent_cell.cell_id,
                    attempted_genome=proposed_strategy,
                    generation=parent_cell.generation + 1,
                    failure_reason=failure_reason,
                    fitness_achieved=fitness,
                )
                mutations_failed += 1

                print(f"💀 Mutation did not improve - recorded statistics")

                history.append({
                    'iteration': iteration + 1,
                    'parent_cell_id': parent_cell.cell_id,
                    'proposed_strategy': proposed_strategy,
                    'fitness': fitness,
                    'survived': False,
                    'llm_cost': iteration_cost,
                    'status': 'no_improvement',
                })

            print(f"\n💰 Cost so far: ${total_cost:.4f}" + (f" / ${cost_threshold:.4f}" if cost_threshold else ""))

        # Complete evolution run in database
        if best_cell_id:
            repo.complete_evolution_run(
                run_id=run_id,
                best_cell_id=best_cell_id,
                total_cells_birthed=cells_birthed,
                total_mutations_failed=mutations_failed,
                final_best_fitness=best_fitness,
                termination_reason="iterations_complete",
                generations_without_improvement=0,
            )
            print(f"\n✅ LLM-guided run #{run_id} completed and saved to database")

        # Final summary
        print(f"\n{'='*70}")
        print(f"LLM-GUIDED EVOLUTION COMPLETE")
        print(f"{'='*70}")

        print(f"\n📊 STATISTICS:")
        print(f"   Total Iterations: {len(history)}")
        print(f"   Cells Birthed: {cells_birthed}")
        print(f"   Mutations Failed: {mutations_failed}")
        print(f"   Success Rate: {cells_birthed}/{cells_birthed + mutations_failed} ({100*cells_birthed/(cells_birthed + mutations_failed) if (cells_birthed + mutations_failed) > 0 else 0:.1f}%)")
        print(f"   Total LLM Cost: ${total_cost:.4f}")
        print(f"   Average Cost per Iteration: ${total_cost/len(history) if history else 0:.4f}")

        if best_strategy:
            print(f"\n🏆 BEST STRATEGY DISCOVERED:")
            print(f"   Cell ID: #{best_cell_id}")
            print(f"   Fitness: ${best_fitness:.2f}")
            print(f"   Strategy: {best_strategy}")

            # Display lineage
            if best_cell_id:
                print(f"\n🧬 LINEAGE OF BEST CELL:")
                lineage = repo.get_lineage(best_cell_id)
                for i, cell in enumerate(lineage):
                    indent = "  " * i
                    arrow = "└─" if i > 0 else "──"
                    print(f"{indent}{arrow} Cell #{cell.cell_id} (Gen {cell.generation}): ${cell.fitness:.2f}")
                    if i < len(lineage) - 1:
                        print(f"{indent}   ↓")
        else:
            print(f"\n❌ No improvements found")

        # Show progression
        print(f"\n📈 ITERATION PROGRESSION:")
        for h in history[-10:]:  # Show last 10
            status = "✓" if h['survived'] else "✗"
            improvement = f" (+${h.get('improvement', 0):.2f})" if h.get('improvement') else ""
            cell_info = f" → Cell #{h.get('child_cell_id')}" if h.get('child_cell_id') else ""
            fitness_val = h.get('fitness')
            fitness_str = f"${fitness_val:8.2f}" if fitness_val is not None else "     N/A"
            print(f"   Iter {h['iteration']:2d}: {status} {fitness_str}{improvement}{cell_info}")

        print(f"\n✓ Results saved to database: {db_path}")
        print(f"\n💡 Pattern taxonomy and mutation proposals stored for future analysis")

        # Mark callgraph node as complete
        if callgraph:
            await callgraph.complete_agent(
                node_id="trading-learn",
                result=f"Completed {len(history)} iterations, birthed {cells_birthed} cells",
                token_count=0,
                num_cached_tokens=0,
                cost=total_cost,
                success=True,
            )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        # Mark callgraph node as failed
        if callgraph:
            await callgraph.fail_agent("trading-learn", str(e))

    finally:
        # Keep web server running if it was started
        if web_server_task:
            print(f"\n🌐 Web server still running at http://localhost:8080")
            print(f"📊 View results in the 'Evolution Cells' tab")
            print(f"⌨️  Press Ctrl+C to stop the server and exit")

            try:
                # Wait indefinitely until user presses Ctrl+C
                await web_server_task
            except (KeyboardInterrupt, asyncio.CancelledError):
                print(f"\n🌐 Shutting down web visualization server...")
                web_server_task.cancel()
                try:
                    await web_server_task
                except asyncio.CancelledError:
                    pass
                print(f"✓ Web server shutdown complete")

async def run_trading_evolve(
    generations: int,
    workdir: Path,
    initial_strategy: str | None = None,
    fitness_goal: float = 200.0,  # Target fitness to achieve before early termination
    lenient_cell_count: int = 100,  # Birth any survivor for first N cells (genetic diversity)
    server_enabled: bool = False,  # Whether to run the web server
) -> None:
    """
    Evolve trading strategies through pure DSL mutation with cell-based storage.

    No LLM usage after generation 0 - completely FREE!
    Uses fitness-based selection and mutation to evolve strategies.
    Stores successful strategies as persistent "cells" (cell lines) with lineage tracking.

    Cell Birth Philosophy:
    - Offline evolution builds a genetic library (cell lines) for future LLM analysis
    - Lenient phase (first lenient_cell_count cells): Birth any survivor for diversity
    - Strict phase (after threshold): Only birth improvements (fitness > parent)
    - Even "losing" cells have value - LLM can analyze failure patterns

    Termination conditions:
    - Reaches fitness_goal (early success)
    - Completes all generations
    - No improvement for 100 consecutive generations (stagnation)
    """
    print("\n" + "="*70)
    print("TRADING STRATEGY EVOLUTION MODE (Cell-Based)")
    print("="*70)
    print(f"\nEvolving strategies for up to {generations} generations using pure mutation.")
    print(f"🎯 Goal: Achieve fitness of ${fitness_goal:.2f}")
    print("💰 FREE after Gen 0! No LLM costs, just natural selection.\n")

    # Start web server if enabled
    web_server_task = None
    if server_enabled:
        print(f"🌐 Starting web visualization server on http://localhost:8080")
        web_server_task = asyncio.create_task(run_server())

    try:
        from .src.benchmarks.trading_benchmarks.trading_benchmark import TradingBenchmark
        from .src.dsl.interpreter import DslInterpreter
        from .src.dsl.mutator import DslMutator
        from .src.storage.cell_repository import CellRepository
        from .src.data.hyperliquid_fetcher import HyperliquidDataFetcher
        import random

        # Initialize EventBus for web UI updates
        event_bus = await EventBus.get_instance() if server_enabled else None

        # Create root callgraph node for evolution run (enables web UI event display)
        callgraph = None
        if server_enabled:
            callgraph = await CallGraphManager.get_instance()
            await callgraph.start_agent(
                agent_name="trading-evolve",
                node_id="trading-evolve",
                args={"generations": generations, "fitness_goal": fitness_goal},
            )

        benchmark = TradingBenchmark()
        interpreter = DslInterpreter()
        mutator = DslMutator()
        problem_id = "trend_following_1"
        base_problem = benchmark.get_problem(problem_id)

        # Create results directory
        results_dir = workdir / "evolution"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cell repository
        db_path = results_dir / "cells.db"
        repo = CellRepository(db_path)
        print(f"📊 Cell database initialized: {db_path}")

        # Start evolution run tracking
        run_id = repo.start_evolution_run(
            run_type='evolution',
            max_generations=generations,
            fitness_goal=fitness_goal,
            symbol='PURR',
            timeframe='1h',
            initial_capital=100.0,
            transaction_fee_rate=0.00045,
        )
        print(f"🧬 Evolution run #{run_id} started")

        # Fetch multi-timeframe data for testing
        print(f"\n📈 Fetching multi-timeframe market data...")
        fetcher = HyperliquidDataFetcher()
        multi_tf_data = fetcher.fetch_multi_timeframe('PURR', lookback_days=30)

        # Track all generations (for logging only, not persistence)
        population_history = []

        # Track cell statistics
        cells_birthed = 0
        mutations_failed = 0

        # Generation 0: Initial strategy
        print(f"{'='*70}")
        print(f"GENERATION 0: Initial Strategy")
        print(f"{'='*70}")

        if initial_strategy:
            current_strategy = initial_strategy
            print(f"Using provided strategy: {current_strategy}")
        else:
            # Generate random initial strategy
            from .src.dsl.grammar import Indicator, Operator, Action as DslAction

            ind1 = random.choice(list(Indicator))
            ind2 = random.choice(list(Indicator))
            param1 = random.choice([0, 5, 10, 14, 20, 30, 50, 100, 200])
            param2 = random.choice([0, 5, 10, 14, 20, 30, 50, 100, 200])
            op = random.choice(list(Operator))
            action1 = random.choice(list(DslAction))
            action2 = random.choice([a for a in DslAction if a != action1])

            param1_str = "" if param1 == 0 else str(param1)
            param2_str = "" if param2 == 0 else str(param2)

            current_strategy = f"IF {ind1.name}({param1_str}) {op.value} {ind2.name}({param2_str}) THEN {action1.name} ELSE {action2.name}"
            print(f"Generated random strategy: {current_strategy}")

        # Setup and test Gen 0
        gen0_dir = results_dir / "gen_0"
        gen0_dir.mkdir(parents=True, exist_ok=True)

        await benchmark.setup_problem(base_problem, gen0_dir, "evolve_container")

        # Write strategy
        answer_file = gen0_dir / "answer.txt"
        answer_file.write_text(current_strategy)

        print(f"\n⚙️  Testing generation 0 on multi-timeframe data...")

        # Parse the strategy for multi-timeframe testing
        program = interpreter.parse(current_strategy)
        if not program:
            print(f"❌ Failed to parse initial strategy: {current_strategy}")
            return

        # Test on all timeframes
        fitness, phenotypes, backtest_log = benchmark.run_multi_timeframe_backtest(
            program, multi_tf_data, 'PURR'
        )

        current_fitness = fitness
        # Check if strategy survived trading on ANY timeframe (portfolio didn't go to zero)
        # Even negative fitness cells are valuable for genetic diversity
        any_timeframe_completed = any(p.total_profit > -100.0 for p in phenotypes.values())  # Didn't blow up entire $100 capital
        survived = any_timeframe_completed
        status = "✓ SURVIVED" if survived else "✗ DIED"
        print(f"\n{status} - Fitness: ${current_fitness:.2f}")
        print(f"{backtest_log}")

        # Birth Gen 0 cell if it survived trading (didn't blow up portfolio)
        # Note: We're building cell lines for LLM analysis, not live trading
        # Even negative fitness is valuable data!
        current_cell_id = None
        if survived:
            current_cell_id = repo.birth_cell(
                generation=0,
                parent_cell_id=None,
                dsl_genome=current_strategy,
                fitness=current_fitness,
                status='online',
            )
            cells_birthed += 1

            # Store phenotypes for each timeframe
            for tf, phenotype in phenotypes.items():
                phenotype.cell_id = current_cell_id
                repo.store_phenotype(phenotype)

            print(f"🧬 Cell #{current_cell_id} birthed (Gen 0, fitness: ${current_fitness:.2f})")
        else:
            print(f"💀 Gen 0 catastrophically failed in trading (portfolio went to zero)")
            print(f"   Cannot continue evolution without a starting cell")
            mutations_failed += 1

        population_history.append({
            'generation': 0,
            'strategy': current_strategy,
            'fitness': current_fitness,
            'survived': survived,
            'parent': None,
            'cell_id': current_cell_id,
        })

        best_fitness = current_fitness
        best_strategy = current_strategy
        best_generation = 0
        best_cell_id = current_cell_id
        generations_without_improvement = 0

        # Check if Gen 0 already met the goal
        if current_fitness >= fitness_goal:
            print(f"\n🎯 GOAL ACHIEVED IN GEN 0! Fitness: ${current_fitness:.2f} >= ${fitness_goal:.2f}")
            print(f"Terminating early - no need to evolve further!")
            # Save and exit early
            best_strategy_file = results_dir / "best_strategy.txt"
            best_strategy_file.write_text(current_strategy)
            print(f"✓ Best strategy saved to: {best_strategy_file}")
            return

        # Evolve!
        for gen in range(1, generations + 1):
            print(f"\n{'='*70}")
            print(f"GENERATION {gen}: Mutation & Selection")
            print(f"{'='*70}")

            # Publish generation progress event
            if event_bus:
                await event_bus.publish(
                    Event(
                        type=EventType.PROGRESS,
                        content=f"Generation {gen}/{generations}: Testing mutation..."
                    ),
                    "trading-evolve"
                )

            # Get best cell as parent (use database query)
            best_cells = repo.get_top_cells(limit=1, status='online')
            if not best_cells:
                print(f"❌ No surviving cells found - cannot continue evolution")
                break

            parent_cell = best_cells[0]
            program = interpreter.parse(parent_cell.dsl_genome)
            if not program:
                print(f"❌ Failed to parse parent strategy")
                continue

            # Mutate
            print(f"Parent (Cell #{parent_cell.cell_id}): {parent_cell.dsl_genome}")
            mutated_program = mutator.mutate(program)
            mutated_strategy = interpreter.to_string(mutated_program)
            print(f"Child:  {mutated_strategy}")

            # Setup and test mutant (save to gen_dir for logging)
            gen_dir = results_dir / f"gen_{gen}"
            gen_dir.mkdir(parents=True, exist_ok=True)
            answer_file = gen_dir / "answer.txt"
            answer_file.write_text(mutated_strategy)

            print(f"\n⚙️  Testing generation {gen} on multi-timeframe data...")

            # Test on all timeframes
            mutated_fitness, phenotypes, backtest_log = benchmark.run_multi_timeframe_backtest(
                mutated_program, multi_tf_data, 'PURR'
            )
            # Check if strategy survived trading (didn't blow up entire portfolio)
            mutated_survived = any(p.total_profit > -100.0 for p in phenotypes.values())

            print(f"{backtest_log}")

            # Determine if we're in lenient mode (building genetic diversity)
            lenient_mode = cells_birthed < lenient_cell_count

            # Selection and cell birth
            is_improvement = mutated_fitness > parent_cell.fitness

            if is_improvement:
                print(f"\n✓ IMPROVEMENT! ${parent_cell.fitness:.2f} → ${mutated_fitness:.2f}")
                should_birth = True
                birth_reason = "improvement"
            elif lenient_mode and mutated_survived:
                print(f"\n→ No improvement (${mutated_fitness:.2f} <= ${parent_cell.fitness:.2f})")
                print(f"   But in LENIENT mode ({cells_birthed}/{lenient_cell_count} cells) - birthing for genetic diversity")
                should_birth = True
                birth_reason = "lenient_diversity"
            else:
                print(f"\n→ No improvement. ${mutated_fitness:.2f} <= ${parent_cell.fitness:.2f}")
                should_birth = False
                birth_reason = None

            if should_birth:
                # Birth child cell
                child_cell_id = repo.birth_cell(
                    generation=gen,
                    parent_cell_id=parent_cell.cell_id,
                    dsl_genome=mutated_strategy,
                    fitness=mutated_fitness,
                    status='online',
                )
                cells_birthed += 1

                # Store phenotypes for each timeframe
                for tf, phenotype in phenotypes.items():
                    phenotype.cell_id = child_cell_id
                    repo.store_phenotype(phenotype)

                phase = "lenient" if birth_reason == "lenient_diversity" else "strict"
                print(f"🧬 Cell #{child_cell_id} birthed (Gen {gen}, parent: #{parent_cell.cell_id}, {phase} mode)")

                # Publish cell birth event
                if event_bus:
                    await event_bus.publish(
                        Event(
                            type=EventType.CELL_BIRTHED,
                            content=f"Cell #{child_cell_id} birthed (Gen {gen}, ${mutated_fitness:.2f})"
                        ),
                        "trading-evolve"
                    )

                current_strategy = mutated_strategy
                current_fitness = mutated_fitness
                current_cell_id = child_cell_id
                selection = "CHILD WINS"
            else:
                # Record failed mutation
                failure_reason = 'negative_fitness' if mutated_fitness < 0 else 'lower_than_parent'
                repo.record_failed_mutation(
                    parent_cell_id=parent_cell.cell_id,
                    attempted_genome=mutated_strategy,
                    generation=gen,
                    failure_reason=failure_reason,
                    fitness_achieved=mutated_fitness,
                )
                mutations_failed += 1

                print(f"💀 Mutation failed - recorded statistics (reason: {failure_reason})")

                # Keep parent
                current_strategy = parent_cell.dsl_genome
                current_fitness = parent_cell.fitness
                current_cell_id = parent_cell.cell_id
                selection = "PARENT WINS"

            # Track best ever
            if mutated_fitness > best_fitness:
                best_fitness = mutated_fitness
                best_strategy = mutated_strategy
                best_generation = gen
                best_cell_id = current_cell_id
                generations_without_improvement = 0
                print(f"🏆 NEW BEST! Fitness: ${best_fitness:.2f}")

                # Auto-save best strategy
                best_strategy_file = results_dir / "best_strategy.txt"
                best_strategy_file.write_text(best_strategy)
                print(f"✓ Auto-saved to: {best_strategy_file}")

                # Check if we've reached the goal
                if best_fitness >= fitness_goal:
                    print(f"\n🎯 GOAL ACHIEVED! Fitness: ${best_fitness:.2f} >= ${fitness_goal:.2f}")
                    print(f"Terminating early at generation {gen}")
                    termination_reason = "fitness_goal_reached"
                    break
            else:
                generations_without_improvement += 1

            population_history.append({
                'generation': gen,
                'strategy': mutated_strategy,
                'fitness': mutated_fitness,
                'survived': mutated_survived,
                'parent': parent_cell.dsl_genome,
                'selection': selection,
                'cell_id': current_cell_id if selection == "CHILD WINS" else None,
            })

            # Check for stagnation (no improvement for 100 generations)
            if generations_without_improvement >= 100:
                print(f"\n⚠️  STAGNATION DETECTED: No improvement for 100 generations")
                print(f"Terminating early at generation {gen}")
                termination_reason = "stagnation"
                break

        # Set termination reason if not already set
        if 'termination_reason' not in locals():
            termination_reason = "max_generations_reached"

        # Complete evolution run in database
        if best_cell_id:
            repo.complete_evolution_run(
                run_id=run_id,
                best_cell_id=best_cell_id,
                total_cells_birthed=cells_birthed,
                total_mutations_failed=mutations_failed,
                final_best_fitness=best_fitness,
                termination_reason=termination_reason,
                generations_without_improvement=generations_without_improvement,
            )
            print(f"\n✅ Evolution run #{run_id} completed and saved to database")

        # Final summary
        print(f"\n{'='*70}")
        print(f"EVOLUTION COMPLETE")
        print(f"{'='*70}")

        survivors = [h for h in population_history if h['survived']]
        print(f"\n📊 STATISTICS:")
        print(f"   Total Generations: {len(population_history)}")
        print(f"   Cells Birthed: {cells_birthed}")
        print(f"   Mutations Failed: {mutations_failed}")
        print(f"   Success Rate: {cells_birthed}/{cells_birthed + mutations_failed} ({100*cells_birthed/(cells_birthed + mutations_failed) if (cells_birthed + mutations_failed) > 0 else 0:.1f}%)")
        print(f"   Survival Rate: {len(survivors)}/{len(population_history)} ({100*len(survivors)/len(population_history):.1f}%)")
        print(f"   Lenient Threshold: {lenient_cell_count} cells")
        print(f"   Mode at End: {'LENIENT (genetic diversity)' if cells_birthed < lenient_cell_count else 'STRICT (improvements only)'}")
        print(f"   Termination Reason: {termination_reason}")

        print(f"\n🏆 BEST STRATEGY FOUND:")
        print(f"   Generation: {best_generation}")
        print(f"   Fitness: ${best_fitness:.2f}")
        print(f"   Strategy: {best_strategy}")
        if best_cell_id:
            print(f"   Cell ID: #{best_cell_id}")

        # Query and display lineage of best cell
        if best_cell_id:
            print(f"\n🧬 LINEAGE OF BEST CELL:")
            lineage = repo.get_lineage(best_cell_id)
            for i, cell in enumerate(lineage):
                indent = "  " * i
                arrow = "└─" if i > 0 else "──"
                print(f"{indent}{arrow} Cell #{cell.cell_id} (Gen {cell.generation}): ${cell.fitness:.2f}")
                if i < len(lineage) - 1:
                    print(f"{indent}   ↓")

        print(f"\n📈 FITNESS PROGRESSION:")
        for h in population_history[-10:]:  # Show last 10
            status = "✓" if h['survived'] else "✗"
            selection = f" [{h.get('selection', 'INITIAL')}]" if 'selection' in h else ""
            cell_info = f" (Cell #{h['cell_id']})" if h.get('cell_id') else ""
            print(f"   Gen {h['generation']:2d}: {status} ${h['fitness']:8.2f}{selection}{cell_info}")

        # Save summary
        summary_file = results_dir / "evolution_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Evolution Summary\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Generations: {len(population_history)}\n")
            f.write(f"Cells Birthed: {cells_birthed}\n")
            f.write(f"Mutations Failed: {mutations_failed}\n")
            f.write(f"Success Rate: {100*cells_birthed/(cells_birthed + mutations_failed) if (cells_birthed + mutations_failed) > 0 else 0:.1f}%\n")
            f.write(f"Survival Rate: {100*len(survivors)/len(population_history):.1f}%\n")
            f.write(f"Termination Reason: {termination_reason}\n\n")
            f.write(f"Best Strategy (Gen {best_generation}):\n")
            f.write(f"  Fitness: ${best_fitness:.2f}\n")
            f.write(f"  Strategy: {best_strategy}\n")
            if best_cell_id:
                f.write(f"  Cell ID: #{best_cell_id}\n")
            f.write(f"\nFull History:\n")
            for h in population_history:
                cell_info = f" (Cell #{h['cell_id']})" if h.get('cell_id') else ""
                f.write(f"  Gen {h['generation']}: ${h['fitness']:.2f} - {h['strategy']}{cell_info}\n")

        print(f"\n✓ Results saved to: {results_dir}")
        print(f"✓ Summary: {summary_file}")
        print(f"✓ Database: {db_path}")
        print(f"\n📊 Query cells: sqlite3 {db_path} 'SELECT cell_id, generation, fitness, status FROM cells ORDER BY fitness DESC LIMIT 10;'")
        print(f"📊 View lineage: sqlite3 {db_path} 'SELECT * FROM cells WHERE cell_id IN (SELECT cell_id FROM cells UNION SELECT parent_cell_id FROM cells WHERE cell_id={best_cell_id if best_cell_id else 1});'")

        # Mark callgraph node as complete
        if callgraph:
            await callgraph.complete_agent(
                node_id="trading-evolve",
                result=f"Completed {len(population_history)} generations, birthed {cells_birthed} cells, best fitness ${best_fitness:.2f}",
                token_count=0,
                num_cached_tokens=0,
                cost=0.0,
                success=True,
            )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        # Mark callgraph node as failed
        if callgraph:
            await callgraph.fail_agent("trading-evolve", str(e))

    finally:
        # Keep web server running if it was started
        if web_server_task:
            print(f"\n🌐 Web server still running at http://localhost:8080")
            print(f"📊 View results in the 'Evolution Cells' tab")
            print(f"⌨️  Press Ctrl+C to stop the server and exit")

            try:
                # Wait indefinitely until user presses Ctrl+C
                await web_server_task
            except (KeyboardInterrupt, asyncio.CancelledError):
                print(f"\n🌐 Shutting down web visualization server...")
                web_server_task.cancel()
                try:
                    await web_server_task
                except asyncio.CancelledError:
                    pass
                print(f"✓ Web server shutdown complete")


async def main():
    """Command-line entry point"""
    load_dotenv()
    import argparse

    parser = argparse.ArgumentParser()
    
    # Add subcommands for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Standard agent mode
    agent_parser = subparsers.add_parser("agent", help="Run the agent on a problem")
    agent_parser.add_argument(
        "--workdir",
        type=str,
        default=Path.home() / "workdir",
        help="Working directory for the agent (root of the default file tree)",
    )
    agent_parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Path where the agent output and logs for this run should be saved. Defaults to workdir / agent_outputs",
    )
    agent_parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="Write a python script for a bouncing yellow ball within a square, make sure to handle collision detection properly. Make the square slowly rotate. Implement it in python. Make sure the ball stays within the square.",
        help="The core prompt or problem statement you want the agent to work on.",
    )
    agent_parser.add_argument(
        "--server",
        "-s",
        action="store_true",
        help="Whether to run the visualisation server.",
    )
    agent_parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to output more verbose logs than usual.",
    )
    agent_parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=None,
        help="Timeout to apply to agent in seconds",
    )
    agent_parser.add_argument(
        "--cost-threshold",
        "-c",
        type=float,
        default=None,
        help="Maximum cost in USD before the agent execution is cancelled",
    )
    
    # Trading demo mode
    demo_parser = subparsers.add_parser("trading-demo", help="Run the trading evolution demo")
    
    # Trading test mode - run a single strategy through backtest
    test_parser = subparsers.add_parser("trading-test", help="Test a DSL strategy on the trading benchmark")
    test_parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="DSL strategy to test (e.g., 'IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL')",
    )
    test_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: workdir/trading_test)",
    )
    
    # Trading learn mode - agent generates and learns from strategies
    learn_parser = subparsers.add_parser("trading-learn", help="Agent learns to generate profitable strategies")
    learn_parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=5,
        help="Number of strategy generation attempts (default: 5)",
    )
    learn_parser.add_argument(
        "--workdir",
        type=str,
        default=Path.home() / "workdir",
        help="Working directory for the agent",
    )
    learn_parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Log directory for agent outputs",
    )
    learn_parser.add_argument(
        "--server",
        "-s",
        action="store_true",
        help="Enable the web visualization server on http://localhost:8080",
    )
    learn_parser.add_argument(
        "--cost-threshold",
        "-c",
        type=float,
        default=None,
        help="Maximum cost in USD before execution is cancelled",
    )

    # Trading evolve mode - pure DSL mutation evolution (no LLM after Gen 0)
    evolve_parser = subparsers.add_parser("trading-evolve", help="Evolve strategies through pure DSL mutation (FREE)")
    evolve_parser.add_argument(
        "--generations",
        "-g",
        type=int,
        default=10,
        help="Number of generations to evolve (default: 10)",
    )
    evolve_parser.add_argument(
        "--workdir",
        type=str,
        default=Path.home() / "workdir",
        help="Working directory for evolution results",
    )
    evolve_parser.add_argument(
        "--initial-strategy",
        type=str,
        default=None,
        help="Starting strategy (if None, generates random)",
    )
    evolve_parser.add_argument(
        "--fitness-goal",
        "-f",
        type=float,
        default=200.0,
        help="Target fitness to achieve for early termination (default: 200.0)",
    )
    evolve_parser.add_argument(
        "--server",
        "-s",
        action="store_true",
        help="Enable the web visualization server on http://localhost:8080",
    )

    # For backward compatibility, also accept args without subcommand
    parser.add_argument(
        "--workdir",
        type=str,
        default=Path.home() / "workdir",
        help="Working directory for the agent (root of the default file tree)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Path where the agent output and logs for this run should be saved.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="The core prompt or problem statement you want the agent to work on.",
    )
    parser.add_argument(
        "--server",
        "-s",
        action="store_true",
        help="Whether to run the visualisation server.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to output more verbose logs.",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=None,
        help="Timeout to apply to agent in seconds",
    )
    parser.add_argument(
        "--cost-threshold",
        "-c",
        type=float,
        default=None,
        help="Maximum cost in USD before execution is cancelled",
    )

    args = parser.parse_args()

    try:
        # Handle different modes
        if args.mode == "trading-demo":
            # Run the trading evolution demo
            await run_trading_demo()
            return
            
        elif args.mode == "trading-test":
            # Test a specific DSL strategy
            await run_trading_test(args.strategy, args.output)
            return
            
        elif args.mode == "trading-learn":
            # Agent learns to generate strategies
            await run_trading_learn(
                iterations=args.iterations,
                workdir=Path(args.workdir),
                logdir=Path(args.logdir) if args.logdir else None,
                server_enabled=args.server,
                cost_threshold=args.cost_threshold,
            )
            return

        elif args.mode == "trading-evolve":
            # Pure DSL mutation evolution (FREE after Gen 0)
            await run_trading_evolve(
                generations=args.generations,
                workdir=Path(args.workdir),
                initial_strategy=args.initial_strategy,
                fitness_goal=args.fitness_goal,
                server_enabled=args.server,
            )
            return
            
        # Standard agent mode (either explicit "agent" or backward compatibility)
        workdir = Path(args.workdir)
        logdir = Path(args.logdir) if args.logdir else None
        
        # For backward compatibility, if no prompt and no mode, show help
        if args.prompt is None and args.mode is None:
            parser.print_help()
            print("\nExample usage:")
            print('  python -m agent_code.agent -s -p "Your task here"')
            print('  python -m agent_code.agent trading-demo')
            print('  python -m agent_code.agent trading-test --strategy "IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL"')
            return

        # Initialise and run agent
        agent = Agent(
            workdir=workdir,
            logdir=logdir,
            server_enabled=args.server,
            debug_mode=args.debug,
        )
        
        prompt = args.prompt if args.prompt else "Write a python script for a bouncing yellow ball within a square, make sure to handle collision detection properly. Make the square slowly rotate. Implement it in python. Make sure the ball stays within the square."
        
        tokens, cached, cost, duration = await agent.exec(
            problem=prompt,
            timeout=args.timeout,
            cost_threshold=args.cost_threshold,
        )

        await agent.create_report()

        # Print the execution trace for visual inspection
        cg = await CallGraphManager.get_instance()

        print("\n\n" + await generate_execution_tree(cg, include_all_events=True))

        percent_cached = cached / tokens * 100 if tokens > 0 else 0

        # Log summary
        logger.info(f"\n{'='*80}")
        logger.info(f"Execution complete:")
        logger.info(f"  Tokens used: {tokens:,} (cached: {percent_cached:.2f}%)")
        logger.info(f"  Cost: ${cost:.4f}")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Output directory: {agent.logdir}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
