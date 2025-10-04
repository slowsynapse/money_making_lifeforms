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
            print(f"  {i+1}. {mutator.to_string(mutated)}")
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
    Agent learns to generate profitable DSL trading strategies.
    
    The agent is given feedback on each attempt and learns to improve.
    This combines LLM reasoning with evolutionary pressure.
    """
    print("\n" + "="*70)
    print("TRADING STRATEGY LEARNING MODE")
    print("="*70)
    print(f"\nThe agent will attempt to generate {iterations} profitable strategies.")
    print("After each attempt, it will see its fitness score and learn from the result.\n")
    
    try:
        from .src.benchmarks.trading_benchmarks.trading_benchmark import TradingBenchmark
        
        benchmark = TradingBenchmark()
        problem_id = "trend_following_1"
        base_problem = benchmark.get_problem(problem_id)
        
        # Display initial state
        state = benchmark.get_state()
        print(f"\n💰 INITIAL STATE:")
        print(f"   Agent Balance: ${state['balance']:.2f}")
        print(f"   Starting Capital per Trade: ${state['initial_capital']:.2f}")
        print(f"   Transaction Cost: ${state['transaction_cost']:.2f}")
        
        # Track history
        history = []
        best_fitness = float('-inf')
        best_strategy = None
        
        for iteration in range(iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{iterations}")
            print(f"{'='*70}")
            
            # Get problem with history context
            if history:
                problem = benchmark.get_problem_with_history(problem_id, history)
                print(f"\nAgent has {len(history)} previous attempts to learn from...")
            else:
                problem = base_problem
                print("\nFirst attempt - no history yet...")
            
            # Setup workdir for this iteration
            iter_workdir = workdir / f"iteration_{iteration}"
            iter_logdir = (logdir / f"iteration_{iteration}") if logdir else (iter_workdir / "logs")
            iter_workdir.mkdir(parents=True, exist_ok=True)
            iter_logdir.mkdir(parents=True, exist_ok=True)
            
            # Setup problem data
            await benchmark.setup_problem(base_problem, iter_workdir, "learn_container")
            
            # Run the agent to generate a strategy
            agent = Agent(
                workdir=iter_workdir,
                logdir=iter_logdir,
                server_enabled=server_enabled,
                debug_mode=False,
            )
            
            print(f"\n🤖 Agent is thinking and generating a strategy...")
            
            try:
                tokens, cached, cost, duration = await agent.exec(
                    problem=problem.statement,
                    timeout=120,  # 2 minute timeout per iteration
                    cost_threshold=cost_threshold,
                )
                
                await agent.create_report()
                
                # Read the generated strategy (agent writes to logdir)
                answer_file = iter_logdir / "answer.txt"
                if answer_file.exists():
                    strategy = answer_file.read_text().strip()
                    print(f"\n📝 Generated Strategy: {strategy}")
                    
                    # Test the strategy
                    print(f"\n⚙️  Running backtest...")
                    score, error, discussion = await benchmark.score_problem(
                        base_problem,
                        str(iter_workdir),
                        str(iter_logdir),  # answer.txt is in logdir
                        "learn_container"
                    )
                    
                    # Update benchmark state with actual results
                    # Extract trading profit from the score (score already includes LLM cost subtraction)
                    trading_profit = score + cost  # Reverse the fitness calculation to get raw profit
                    actual_fitness = benchmark.update_state(trading_profit, cost, score > 0)
                    
                    # Record result
                    survived = score > 0
                    result = {
                        'iteration': iteration + 1,
                        'strategy': strategy,
                        'fitness': actual_fitness,
                        'survived': survived,
                        'llm_cost': cost,
                        'tokens': tokens,
                        'duration': duration,
                    }
                    
                    if error:
                        result['death_reason'] = error
                    elif not survived:
                        result['death_reason'] = "Fitness ≤ 0"
                    
                    history.append(result)
                    
                    # Display results
                    print(f"\n{'='*70}")
                    print(f"RESULTS FOR ITERATION {iteration + 1}")
                    print(f"{'='*70}")
                    print(f"Strategy: {strategy}")
                    print(f"Fitness: ${score:.2f}")
                    print(f"Status: {'✓ SURVIVED' if survived else '✗ DIED'}")
                    print(f"LLM Cost: ${cost:.4f}")
                    print(f"Tokens: {tokens:,} (cached: {cached})")
                    print(f"Duration: {duration:.2f}s")
                    print(f"\nDetails: {discussion}")
                    
                    # Update best
                    if score > best_fitness:
                        best_fitness = score
                        best_strategy = strategy
                        print(f"\n🏆 NEW BEST STRATEGY! Fitness: ${best_fitness:.2f}")
                    
                    # Show current state
                    current_state = benchmark.get_state()
                    print(f"\n💰 AGENT STATE AFTER ITERATION {iteration + 1}:")
                    print(f"   Current Balance: ${current_state['balance']:.2f}")
                    print(f"   Net Result: ${current_state['net_result']:.2f}")
                    print(f"   Total Attempts: {current_state['attempts']}")
                    
                    if current_state['balance'] <= 0:
                        print(f"\n⚠️  WARNING: Agent balance depleted! Agent would be bankrupt.")
                    
                else:
                    print(f"\n❌ Agent failed to create answer.txt")
                    history.append({
                        'iteration': iteration + 1,
                        'strategy': 'None',
                        'fitness': -1000,
                        'survived': False,
                        'death_reason': 'No answer file created',
                        'llm_cost': cost,
                    })
                    
            except Exception as e:
                print(f"\n❌ Error during iteration {iteration + 1}: {e}")
                history.append({
                    'iteration': iteration + 1,
                    'strategy': 'Error',
                    'fitness': -1000,
                    'survived': False,
                    'death_reason': str(e),
                    'llm_cost': 0,
                })
        
        # Final summary
        print(f"\n\n{'='*70}")
        print("LEARNING SESSION COMPLETE")
        print(f"{'='*70}")
        
        # Get final state
        final_state = benchmark.get_state()
        
        print(f"\n💰 FINAL FINANCIAL STATE:")
        print(f"   Starting Balance: ${final_state['initial_capital']:.2f}")
        print(f"   Final Balance: ${final_state['balance']:.2f}")
        print(f"   Net Result: ${final_state['net_result']:.2f}")
        print(f"   Total Trading Profit: ${final_state['total_trading_profit']:.2f}")
        print(f"   Total LLM Costs: ${final_state['total_llm_costs']:.4f}")
        print(f"   Total Attempts: {final_state['attempts']}")
        
        if final_state['balance'] > final_state['initial_capital']:
            print(f"\n✅ AGENT IS PROFITABLE! Grew balance by ${final_state['net_result']:.2f}")
        elif final_state['balance'] > 0:
            print(f"\n⚠️  AGENT SURVIVED but lost ${-final_state['net_result']:.2f}")
        else:
            print(f"\n❌ AGENT WENT BANKRUPT! Lost all capital.")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        survivors = [h for h in history if h['survived']]
        print(f"   Success Rate: {len(survivors)}/{len(history)} ({100*len(survivors)/len(history) if history else 0:.1f}%)")
        
        if survivors:
            avg_survivor_fitness = sum(h['fitness'] for h in survivors) / len(survivors)
            print(f"   Average Fitness (survivors): ${avg_survivor_fitness:.2f}")
        
        if best_strategy:
            print(f"\n🏆 BEST STRATEGY:")
            print(f"   {best_strategy}")
            print(f"   Fitness: ${best_fitness:.2f}")
        else:
            print(f"\n❌ No surviving strategies found")
        
        # Show progression
        print(f"\n📈 FITNESS PROGRESSION:")
        for h in history:
            status = "✓" if h['survived'] else "✗"
            print(f"   Iteration {h['iteration']}: {status} ${h['fitness']:.2f} - {h['strategy'][:50]}")
        
        print(f"\n✓ All logs saved to: {logdir if logdir else workdir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


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
