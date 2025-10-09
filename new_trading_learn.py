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
            run_type='llm_guided',
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
            print(f"   Iter {h['iteration']:2d}: {status} ${h.get('fitness', 0):8.2f}{improvement}{cell_info}")

        print(f"\n✓ Results saved to database: {db_path}")
        print(f"\n💡 Pattern taxonomy and mutation proposals stored for future analysis")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
