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

    # QUERY command (placeholder for now)
    query_parser = subparsers.add_parser(
        "query",
        help="Query the cell database"
    )
    query_parser.add_argument(
        "query_type",
        choices=["top-cells", "lineage", "patterns", "runs"],
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
    """Run evolution mode with clean parameters."""
    print(f"🧬 Starting Evolution")
    print(f"  Generations: {args.generations}")
    print(f"  Fitness Goal: ${args.fitness_goal}")
    print(f"  Symbol: {args.symbol}")
    print(f"  Initial Capital: ${args.initial_capital}")
    print(f"  Stagnation Limit: {args.stagnation_limit}")
    print(f"  Lenient Cells: {args.lenient_cells}")
    print()

    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/evolve_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")

    # Call the existing evolution function
    # Note: We'll refactor this to use the new extracted module in Step 2
    await run_trading_evolve(
        max_generations=args.generations,
        fitness_goal=args.fitness_goal,
        stagnation_limit=args.stagnation_limit,
        lenient_cell_count=args.lenient_cells,
        output_dir=output_dir
    )


async def run_learn(args):
    """Run LLM-guided learning mode."""
    print(f"🤖 Starting LLM-Guided Learning")
    print(f"  Iterations: {args.iterations}")
    print(f"  Cost Limit: ${args.cost_limit}")
    print(f"  Using: {'Local LLM (Ollama)' if args.use_local_llm else 'Cloud LLM'}")
    print(f"  Minimum Cells: {args.min_cells}")
    print()

    # Set environment variable for local LLM if requested
    if args.use_local_llm:
        os.environ['USE_LOCAL_LLM'] = 'true'

    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/learn_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")

    # Call the existing learning function
    await run_trading_learn(
        num_iterations=args.iterations,
        cost_limit=args.cost_limit,
        output_dir=output_dir
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
    from pathlib import Path

    # Find the most recent cells.db
    db_paths = []

    # Search for all cells.db files in results directory
    results_dir = Path("results")
    if results_dir.exists():
        # Use glob to find all cells.db files
        all_dbs = list(results_dir.glob("**/cells.db"))
        # Sort by modification time (most recent first)
        all_dbs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        db_paths.extend(all_dbs)

    # Also check root directory
    if Path("cells.db").exists():
        db_paths.append(Path("cells.db"))

    # Find first existing database
    db_path = None
    for path in db_paths:
        if path.exists():
            db_path = path
            break

    if not db_path:
        print("❌ No cell database found. Run 'evolve' first to create cells.")
        return

    print(f"📊 Using database: {db_path}\n")
    repo = CellRepository(db_path)

    if args.query_type == "top-cells":
        cells = repo.get_top_cells(limit=args.limit, min_trades=args.min_trades)
        print(f"Top {len(cells)} Cells by Fitness:\n")
        print(f"{'ID':<6} {'Gen':<5} {'Fitness':<10} {'Trades':<8} {'Status':<10} {'Strategy'}")
        print("-" * 80)

        for cell in cells:
            # Get phenotype for trade count
            phenotypes = repo.get_phenotypes(cell.cell_id)
            total_trades = sum(p.total_trades for p in phenotypes) if phenotypes else 0

            # Truncate strategy for display
            strategy = cell.dsl_genome[:40] + "..." if len(cell.dsl_genome) > 40 else cell.dsl_genome

            print(f"{cell.cell_id:<6} {cell.generation:<5} ${cell.fitness:<9.2f} {total_trades:<8} {cell.status:<10} {strategy}")

    elif args.query_type == "lineage":
        if not args.cell_id:
            print("❌ --cell-id required for lineage query")
            return

        lineage = repo.get_lineage(args.cell_id)
        print(f"Lineage for Cell #{args.cell_id}:\n")

        for i, ancestor in enumerate(lineage):
            indent = "  " * i
            print(f"{indent}└─ Cell #{ancestor.cell_id} (Gen {ancestor.generation}): ${ancestor.fitness:.2f}")

    elif args.query_type == "patterns":
        # This will be implemented when we have pattern discovery
        print("Pattern discovery coming soon...")

    elif args.query_type == "runs":
        print("Evolution runs query coming soon...")


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