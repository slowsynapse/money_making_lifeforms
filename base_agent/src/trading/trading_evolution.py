"""
Trading Evolution System - Core Logic

Extracted from base_agent/agent.py for better code organization.
Contains all trading mode implementations:
- run_trading_demo: Demo of DSL parsing and mutation
- run_trading_test: Test a single strategy
- run_trading_evolve: Pure genetic evolution (no LLM costs)
- run_trading_learn: LLM-guided intelligent mutations
"""

import asyncio
import tempfile
import random
from pathlib import Path

from ..events import EventBus
from ..callgraph.manager import CallGraphManager
from ..types.event_types import EventType, Event

async def run_trading_demo():
    """Run the trading evolution demo"""
    print("\n" + "="*60)
    print("TRADING EVOLUTION SYSTEM - DEMO")
    print("="*60)
    print("\nLoading trading components...")
    
    try:
        from ..benchmarks.trading_benchmarks.trading_benchmark import TradingBenchmark
        from ..dsl.interpreter import DslInterpreter
        from ..dsl.mutator import DslMutator
        
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
        from ..benchmarks.trading_benchmarks.trading_benchmark import TradingBenchmark
        from ..benchmarks.base import Problem
        
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

    # Note: Web server (trading_api.py on port 8081) is managed externally
    # EventBus and CallGraphManager work independently of web server lifecycle

    try:
        from ..benchmarks.trading_benchmarks.trading_benchmark import TradingBenchmark
        from ..dsl.interpreter import DslInterpreter
        from ..storage.cell_repository import CellRepository
        from ..data.hyperliquid_fetcher import HyperliquidDataFetcher
        from ..analysis import (
            analyze_cells_in_batches,
            merge_pattern_discoveries,
            propose_intelligent_mutation,
        )

        # Initialize EventBus for web UI updates (always get instance if server enabled)
        event_bus = await EventBus.get_instance() if server_enabled else None

        # Create root callgraph node for learn run (enables web UI event display)
        # Always create callgraph if server_enabled, even if server already running
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
        # IMPORTANT: Filter out zero-trade strategies (min_trades=1) to avoid learning from inactivity
        top_cells = repo.get_top_cells(limit=100, status='online', min_trades=1)
        if not top_cells:
            print(f"❌ No online cells with trades found - cannot analyze patterns")
            print(f"   (Zero-trade strategies are excluded from LLM analysis)")
            return

        print(f"📊 Analyzing top {len(top_cells)} active cells for pattern discovery...")
        print(f"   (Filtering out zero-trade strategies)")
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

async def run_trading_evolve(
    generations: int,
    workdir: Path,
    initial_strategy: str | None = None,
    fitness_goal: float = 200.0,  # Target fitness to achieve before early termination
    lenient_cell_count: int = 100,  # Birth any survivor for first N cells (genetic diversity)
    server_enabled: bool = False,  # Whether to run the web server
    dish_name: str | None = None,  # Petri dish experiment name (e.g., 'baseline_purr')
    start_generation: int = 0,  # Starting generation (for resuming dishes)
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

    # Note: Web server (trading_api.py on port 8081) is managed externally
    # EventBus and CallGraphManager work independently of web server lifecycle

    try:
        from ..benchmarks.trading_benchmarks.trading_benchmark import TradingBenchmark
        from ..dsl.interpreter import DslInterpreter
        from ..dsl.mutator import DslMutator
        from ..storage.cell_repository import CellRepository
        from ..data.hyperliquid_fetcher import HyperliquidDataFetcher
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

        # Check if we're resuming a dish or starting fresh
        resuming_dish = start_generation > 0

        # Initialize tracking variables
        best_fitness = float('-inf')
        best_strategy = None
        best_generation = start_generation
        best_cell_id = None
        generations_without_improvement = 0

        # Generation 0: Initial strategy (only if not resuming)
        if not resuming_dish:
            print(f"{'='*70}")
            print(f"GENERATION {start_generation}: Initial Strategy")
            print(f"{'='*70}")

            if initial_strategy:
                current_strategy = initial_strategy
                print(f"Using provided strategy: {current_strategy}")
            else:
                # Generate random initial strategy
                from ..dsl.grammar import Indicator, Operator, Action as DslAction

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
            gen0_dir = results_dir / f"gen_{start_generation}"
            gen0_dir.mkdir(parents=True, exist_ok=True)

            await benchmark.setup_problem(base_problem, gen0_dir, "evolve_container")

            # Write strategy
            answer_file = gen0_dir / "answer.txt"
            answer_file.write_text(current_strategy)

            print(f"\n⚙️  Testing generation {start_generation} on multi-timeframe data...")

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
                    generation=start_generation,
                    parent_cell_id=None,
                    dsl_genome=current_strategy,
                    fitness=current_fitness,
                    status='online',
                    dish_name=dish_name,
                )
                cells_birthed += 1

                # Store phenotypes for each timeframe
                for tf, phenotype in phenotypes.items():
                    phenotype.cell_id = current_cell_id
                    repo.store_phenotype(phenotype)

                print(f"🧬 Cell #{current_cell_id} birthed (Gen {start_generation}, fitness: ${current_fitness:.2f})")
            else:
                print(f"💀 Gen {start_generation} catastrophically failed in trading (portfolio went to zero)")
                print(f"   Cannot continue evolution without a starting cell")
                mutations_failed += 1

            population_history.append({
                'generation': start_generation,
                'strategy': current_strategy,
                'fitness': current_fitness,
                'survived': survived,
                'parent': None,
                'cell_id': current_cell_id,
            })

            best_fitness = current_fitness
            best_strategy = current_strategy
            best_generation = start_generation
            best_cell_id = current_cell_id
            generations_without_improvement = 0

            # Check if Gen 0 already met the goal
            if current_fitness >= fitness_goal:
                print(f"\n🎯 GOAL ACHIEVED IN GEN {start_generation}! Fitness: ${current_fitness:.2f} >= ${fitness_goal:.2f}")
                print(f"Terminating early - no need to evolve further!")
                # Save and exit early
                best_strategy_file = results_dir / "best_strategy.txt"
                best_strategy_file.write_text(current_strategy)
                print(f"✓ Best strategy saved to: {best_strategy_file}")
                return
        else:
            # Resuming dish - load best cell from database
            print(f"{'='*70}")
            print(f"RESUMING DISH: {dish_name}")
            print(f"{'='*70}")
            print(f"\n📊 Loading existing cells from generation 0 to {start_generation - 1}...")

            # Get best cell as starting point
            best_cells = repo.get_top_cells(limit=1, status='online', dish_name=dish_name)
            if not best_cells:
                print(f"❌ No surviving cells found in dish '{dish_name}' - cannot resume evolution")
                return

            best_cell = best_cells[0]
            best_fitness = best_cell.fitness
            best_strategy = best_cell.dsl_genome
            best_generation = best_cell.generation
            best_cell_id = best_cell.cell_id

            print(f"✓ Resuming from best cell: #{best_cell_id} (Gen {best_generation})")
            print(f"   Fitness: ${best_fitness:.2f}")
            print(f"   Strategy: {best_strategy}")
            print(f"\n🚀 Starting evolution from generation {start_generation}...")

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
                    generation=start_generation + gen,
                    parent_cell_id=parent_cell.cell_id,
                    dsl_genome=mutated_strategy,
                    fitness=mutated_fitness,
                    status='online',
                    dish_name=dish_name,
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

            # Track best ever and update adaptive temperature
            if mutated_fitness > best_fitness:
                best_fitness = mutated_fitness
                best_strategy = mutated_strategy
                best_generation = gen
                best_cell_id = current_cell_id
                generations_without_improvement = 0
                print(f"🏆 NEW BEST! Fitness: ${best_fitness:.2f}")

                # Update mutator temperature (cooling down - improvements found)
                new_temp = mutator.update_temperature(best_fitness)
                print(f"🌡️  Mutation temperature: {new_temp:.3f} (COOLING - strategy improving)")

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

                # Update mutator temperature (may heat up after 100 gens plateau)
                new_temp = mutator.update_temperature(best_fitness)
                if mutator.gens_without_improvement == 0:  # Just reset after heating
                    print(f"🌡️  Mutation temperature: {new_temp:.3f} (HEATED UP after 100-gen plateau - trying complexity)")

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
