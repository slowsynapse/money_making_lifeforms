#!/usr/bin/env python3
"""
Trading Evolution System CLI
Clean command-line interface for the cell-based trading evolution system.
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import Optional

# Add base_agent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_agent.agent import (
    run_trading_evolve,
    run_trading_learn,
    run_trading_test,
    run_trading_demo
)


def create_parser():
    """Create the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Trading Evolution System - Evolve profitable trading strategies using genetic algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s evolve --generations 100 --fitness-goal 50
  %(prog)s learn --iterations 10 --use-local-llm
  %(prog)s test --strategy "IF DELTA(0) > DELTA(20) THEN BUY ELSE SELL"
  %(prog)s query top-cells --limit 10
  %(prog)s web --port 8081
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Trading Evolution System v1.0.0"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True
    )

    # EVOLVE command
    evolve_parser = subparsers.add_parser(
        "evolve",
        help="Run genetic evolution to discover trading strategies"
    )
    evolve_parser.add_argument(
        "-g", "--generations",
        type=int,
        default=100,
        help="Number of generations to evolve (default: 100)"
    )
    evolve_parser.add_argument(
        "-f", "--fitness-goal",
        type=float,
        default=50.0,
        help="Target fitness to achieve (default: $50.0)"
    )
    evolve_parser.add_argument(
        "-s", "--symbol",
        type=str,
        default="PURR",
        help="Trading symbol (default: PURR)"
    )
    evolve_parser.add_argument(
        "-c", "--initial-capital",
        type=float,
        default=1000.0,
        help="Starting capital (default: $1000)"
    )
    evolve_parser.add_argument(
        "--stagnation-limit",
        type=int,
        default=20,
        help="Generations without improvement before stopping (default: 20)"
    )
    evolve_parser.add_argument(
        "--lenient-cells",
        type=int,
        default=100,
        help="Number of cells to birth leniently for diversity (default: 100)"
    )
    evolve_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: auto-generated)"
    )
    evolve_parser.add_argument(
        "--dish",
        type=str,
        default=None,
        help="Named experiment dish (e.g., 'baseline_purr'). Creates experiments/dish_name/ structure."
    )
    evolve_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume evolution in existing dish (requires --dish)"
    )

    # LEARN command
    learn_parser = subparsers.add_parser(
        "learn",
        help="Run LLM-guided learning to improve strategies"
    )
    learn_parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=10,
        help="Number of learning iterations (default: 10)"
    )
    learn_parser.add_argument(
        "-c", "--cost-limit",
        type=float,
        default=1.0,
        help="Maximum LLM cost in USD (default: $1.00)"
    )
    learn_parser.add_argument(
        "--use-local-llm",
        action="store_true",
        help="Use local Ollama instead of cloud LLM"
    )
    learn_parser.add_argument(
        "--min-cells",
        type=int,
        default=50,
        help="Minimum cells required in database (default: 50)"
    )
    learn_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: auto-generated)"
    )
    learn_parser.add_argument(
        "--dish",
        type=str,
        default=None,
        help="Dish to learn from (default: most recent)"
    )

    # TEST command
    test_parser = subparsers.add_parser(
        "test",
        help="Test a specific strategy"
    )
    test_parser.add_argument(
        "strategy",
        type=str,
        help='DSL strategy string (e.g., "IF DELTA(0) > DELTA(20) THEN BUY ELSE SELL")'
    )
    test_parser.add_argument(
        "-s", "--symbol",
        type=str,
        default="PURR",
        help="Trading symbol (default: PURR)"
    )
    test_parser.add_argument(
        "-c", "--initial-capital",
        type=float,
        default=1000.0,
        help="Starting capital (default: $1000)"
    )
    test_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: auto-generated)"
    )

    # DEMO command
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run a demonstration with predefined strategies"
    )
    demo_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: auto-generated)"
    )

    # QUERY command
    query_parser = subparsers.add_parser(
        "query",
        help="Query the cell database"
    )
    query_parser.add_argument(
        "query_type",
        choices=["top-cells", "lineage", "patterns", "runs", "summary"],
        help="Type of query to perform"
    )
    query_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit number of results (default: 10)"
    )
    query_parser.add_argument(
        "--cell-id",
        type=int,
        help="Cell ID for lineage query"
    )
    query_parser.add_argument(
        "--min-trades",
        type=int,
        default=0,
        help="Minimum number of trades (default: 0)"
    )
    query_parser.add_argument(
        "--dish",
        type=str,
        default=None,
        help="Filter by dish name"
    )

    # LIST-DISHES command
    list_dishes_parser = subparsers.add_parser(
        "list-dishes",
        help="List all experiment dishes"
    )

    # WEB command (placeholder for now)
    web_parser = subparsers.add_parser(
        "web",
        help="Start the web interface"
    )
    web_parser.add_argument(
        "-p", "--port",
        type=int,
        default=8081,
        help="Port to run web server (default: 8081)"
    )
    web_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)"
    )

    return parser


async def run_evolve(args):
    """Run evolution mode with dish support."""
    print(f"🧬 Starting Evolution")
    print(f"  Generations: {args.generations}")
    print(f"  Fitness Goal: ${args.fitness_goal}")
    print(f"  Symbol: {args.symbol}")
    print(f"  Initial Capital: ${args.initial_capital}")
    print(f"  Stagnation Limit: {args.stagnation_limit}")
    print(f"  Lenient Cells: {args.lenient_cells}")

    if args.dish:
        print(f"  Dish: {args.dish} {'(resuming)' if args.resume else '(new)'}")
    print()

    # Dish-based architecture
    if args.dish:
        from base_agent.src.dish_manager import DishManager
        from base_agent.src.storage.cell_repository import CellRepository

        dm = DishManager(Path("experiments"))

        if args.resume:
            # Load existing dish
            dish_path, config = dm.load_dish(args.dish)
            db_path = dish_path / "evolution" / "cells.db"
            repo = CellRepository(db_path)
            start_gen = repo.get_max_generation(args.dish) + 1
            print(f"📊 Resuming from generation {start_gen}")
        else:
            # Check if dish already exists
            try:
                dm.load_dish(args.dish)
                print(f"❌ Error: Dish '{args.dish}' already exists. Use --resume to continue it.")
                return
            except FileNotFoundError:
                pass

            # Create new dish
            dish_path = dm.create_dish(
                dish_name=args.dish,
                symbol=args.symbol,
                initial_capital=args.initial_capital,
                description=f"Evolution run: {args.generations} generations, goal ${args.fitness_goal}"
            )
            start_gen = 0

        db_path = dish_path / "evolution" / "cells.db"

        # Run evolution
        await run_trading_evolve(
            generations=args.generations,
            workdir=dish_path,
            fitness_goal=args.fitness_goal,
            lenient_cell_count=args.lenient_cells,
            server_enabled=False,
            dish_name=args.dish,
            start_generation=start_gen
        )

        # Update dish config after run
        repo = CellRepository(db_path)
        top_cells = repo.get_top_cells(limit=1, dish_name=args.dish)

        if top_cells:
            best_cell = top_cells[0]
            dm.update_dish_config(
                dish_name=args.dish,
                total_generations=repo.get_max_generation(args.dish) + 1,
                total_cells=repo.get_cell_count(),
                best_fitness=best_cell.fitness,
                best_cell_name=best_cell.cell_name
            )

    else:
        # Legacy timestamp-based behavior
        if args.output_dir:
            output_dir = args.output_dir
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/evolve_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
        print(f"💡 Tip: Use --dish <name> for named experiments!\n")

        # Call evolution function (backward compatible)
        await run_trading_evolve(
            generations=args.generations,
            workdir=Path(output_dir),
            fitness_goal=args.fitness_goal,
            lenient_cell_count=args.lenient_cells,
            server_enabled=False
        )


async def run_learn(args):
    """Run LLM-guided learning mode."""
    print(f"🤖 Starting LLM-Guided Learning")
    print(f"  Iterations: {args.iterations}")
    print(f"  Cost Limit: ${args.cost_limit}")
    print(f"  Using: {'Local LLM (Ollama)' if args.use_local_llm else 'Cloud LLM'}")
    print(f"  Minimum Cells: {args.min_cells}")

    # Set environment variable for local LLM if requested
    if args.use_local_llm:
        os.environ['USE_LOCAL_LLM'] = 'true'

    # Determine workdir based on dish or output_dir
    if args.dish:
        from base_agent.src.dish_manager import DishManager
        dm = DishManager(Path("experiments"))
        try:
            dish_path, config = dm.load_dish(args.dish)
            workdir = dish_path
            print(f"  Using dish: {args.dish}")
            print(f"  Dish path: {dish_path}\n")
        except FileNotFoundError:
            print(f"\n❌ Dish '{args.dish}' not found. Available dishes:")
            for dish in dm.list_dishes():
                print(f"  - {dish['dish_name']}")
            return
    else:
        # Legacy behavior: create timestamp-based output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/learn_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)
        workdir = Path(output_dir)
        print(f"  Output directory: {output_dir}")
        print(f"  💡 Tip: Use --dish <name> to run on a specific experiment!\n")

    # Call the existing learning function
    await run_trading_learn(
        iterations=args.iterations,
        workdir=workdir,
        logdir=None,
        server_enabled=False,
        cost_threshold=args.cost_limit
    )


async def run_test(args):
    """Test a specific strategy."""
    print(f"🧪 Testing Strategy")
    print(f"  Strategy: {args.strategy}")
    print(f"  Symbol: {args.symbol}")
    print(f"  Initial Capital: ${args.initial_capital}")
    print()

    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/test_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")

    # Call the existing test function
    await run_trading_test(
        strategy_dsl=args.strategy,
        output_dir=output_dir
    )


async def run_demo(args):
    """Run demonstration mode."""
    print(f"🎮 Running Trading Demo")
    print()

    # Call the existing demo function
    await run_trading_demo()


async def run_query(args):
    """Query the cell database."""
    print(f"🔍 Query Mode: {args.query_type}")

    # Import here to avoid circular dependencies
    from base_agent.src.storage.cell_repository import CellRepository
    from base_agent.src.dish_manager import DishManager
    from pathlib import Path

    # If dish specified, use dish database
    if args.dish:
        dm = DishManager(Path("experiments"))
        try:
            dish_path, config = dm.load_dish(args.dish)
            db_path = dish_path / "evolution" / "cells.db"
            print(f"📊 Using dish: {args.dish}\n")
        except FileNotFoundError:
            print(f"❌ Dish '{args.dish}' not found.")
            print("Available dishes:")
            for dish in dm.list_dishes():
                print(f"  - {dish['dish_name']}")
            return
    else:
        # Find the most recent cells.db
        db_paths = []

        # First check experiments directory
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

        # Find first existing database
        db_path = None
        for path in db_paths:
            if path.exists():
                db_path = path
                break

        if not db_path:
            print("❌ No cell database found. Run 'evolve' first to create cells.")
            return

        print(f"📊 Using database: {db_path}")
        print(f"💡 Tip: Use --dish <name> to query a specific experiment\n")

    repo = CellRepository(db_path)

    if args.query_type == "summary":
        # Show dish summary
        print(f"📊 Database Summary\n")
        total_cells = repo.get_cell_count()
        max_gen = repo.get_max_generation(args.dish)

        print(f"  Total Cells: {total_cells}")
        print(f"  Max Generation: {max_gen}")

        if total_cells > 0:
            top_cells = repo.get_top_cells(limit=1, dish_name=args.dish)
            if top_cells:
                best_cell = top_cells[0]
                print(f"  Best Fitness: ${best_cell.fitness:.2f}")
                if best_cell.cell_name:
                    print(f"  Best Cell: {best_cell.cell_name}")
                else:
                    print(f"  Best Cell: #{best_cell.cell_id} (Gen {best_cell.generation})")
                print(f"  Strategy: {best_cell.dsl_genome}")

    elif args.query_type == "top-cells":
        cells = repo.get_top_cells(
            limit=args.limit,
            min_trades=args.min_trades,
            dish_name=args.dish
        )
        print(f"Top {len(cells)} Cells by Fitness:\n")

        # Adjust headers based on whether we have cell names
        if cells and cells[0].cell_name:
            print(f"{'Cell Name':<25} {'Gen':<5} {'Fitness':<10} {'Trades':<8} {'Strategy'}")
            print("-" * 90)
        else:
            print(f"{'ID':<6} {'Gen':<5} {'Fitness':<10} {'Trades':<8} {'Status':<10} {'Strategy'}")
            print("-" * 80)

        for cell in cells:
            # Get phenotype for trade count
            phenotypes = repo.get_phenotypes(cell.cell_id)
            total_trades = sum(p.total_trades for p in phenotypes) if phenotypes else 0

            # Truncate strategy for display
            strategy = cell.dsl_genome[:40] + "..." if len(cell.dsl_genome) > 40 else cell.dsl_genome

            if cell.cell_name:
                print(f"{cell.cell_name:<25} {cell.generation:<5} ${cell.fitness:<9.2f} {total_trades:<8} {strategy}")
            else:
                print(f"{cell.cell_id:<6} {cell.generation:<5} ${cell.fitness:<9.2f} {total_trades:<8} {cell.status:<10} {strategy}")

    elif args.query_type == "lineage":
        if not args.cell_id:
            print("❌ --cell-id required for lineage query")
            return

        lineage = repo.get_lineage(args.cell_id)
        print(f"Lineage for Cell #{args.cell_id}:\n")

        for i, ancestor in enumerate(lineage):
            indent = "  " * i
            cell_display = ancestor.cell_name or f"#{ancestor.cell_id}"
            print(f"{indent}└─ {cell_display} (Gen {ancestor.generation}): ${ancestor.fitness:.2f}")

    elif args.query_type == "patterns":
        # This will be implemented when we have pattern discovery
        print("Pattern discovery coming soon...")

    elif args.query_type == "runs":
        print("Evolution runs query coming soon...")


async def run_list_dishes(args):
    """List all experiment dishes."""
    from base_agent.src.dish_manager import DishManager
    from pathlib import Path

    dm = DishManager(Path("experiments"))
    dishes = dm.list_dishes()

    if not dishes:
        print("🧫 No dishes found.")
        print("\nCreate your first dish with:")
        print("  ./trade evolve --dish <name> --generations 100")
        return

    print(f"🧫 Experiment Dishes ({len(dishes)} total):\n")
    print(f"{'Dish Name':<25} {'Cells':<8} {'Gens':<6} {'Best Fitness':<15} {'Created':<12} {'Description'}")
    print("-" * 100)

    for dish in dishes:
        best_fit = f"${dish['best_fitness']:.2f}" if dish['best_fitness'] else "N/A"
        created = dish['created_at'][:10]  # Just the date
        desc = dish['description'][:30] + "..." if len(dish['description']) > 30 else dish['description']
        print(f"{dish['dish_name']:<25} {dish['total_cells']:<8} {dish['total_generations']:<6} {best_fit:<15} {created:<12} {desc}")

    print(f"\nQuery a dish with:")
    print(f"  ./trade query summary --dish <name>")
    print(f"  ./trade query top-cells --dish <name>")


async def run_web(args):
    """Start the web interface."""
    print(f"🌐 Starting Web Interface")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Press Ctrl+C to stop")
    print()

    # This will be implemented in Week 3
    print("Web interface coming in Week 3 of the OPUS_PLAN...")
    print("For now, use the existing web interface:")
    print("  python -m base_agent.src.web_server.server")


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        if args.command == "evolve":
            await run_evolve(args)
        elif args.command == "learn":
            await run_learn(args)
        elif args.command == "test":
            await run_test(args)
        elif args.command == "demo":
            await run_demo(args)
        elif args.command == "query":
            await run_query(args)
        elif args.command == "list-dishes":
            await run_list_dishes(args)
        elif args.command == "web":
            await run_web(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())