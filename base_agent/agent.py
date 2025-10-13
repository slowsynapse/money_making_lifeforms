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
# Web server moved to "not in use" - internal Docker visualization only
# Main APIs are at root level: trading_api.py (port 8081), sica_api.py (port 8082)
# from .src.web_server import run_server
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

# Import trading functions from dedicated module
from .src.trading import (
    run_trading_demo,
    run_trading_test,
    run_trading_evolve,
    run_trading_learn,
)


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
        # NOTE: Internal web server disabled - moved to "not in use"
        # Use root-level APIs instead: trading_api.py (8081), sica_api.py (8082)
        # if self.server_enabled:
        #     self.web_server_task = asyncio.create_task(run_server())

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
            # NOTE: Internal web server disabled - moved to "not in use"
            # if self.server_enabled and self.web_server_task:
            #     self.web_server_task.cancel()
            #     try:
            #         await self.web_server_task
            #     except asyncio.CancelledError:
            #         pass
            #     self.web_server_task = None
            pass  # Placeholder for removed web server shutdown

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




async def main():
    """
    Command-line entry point.

    DEPRECATED: This argparse setup is being phased out. Please use the
    top-level `trading_cli.py` script for a cleaner and more up-to-date
    interface. This will be removed in a future version.
    """
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
