#!/usr/bin/env python3
"""
Test a trading strategy on real PURR data.

This script tests the interpreter with lookback on real market data.
"""

import sys
import asyncio
from pathlib import Path

# Add base_agent to path
sys.path.insert(0, str(Path(__file__).parent / "base_agent"))

from base_agent.src.benchmarks.trading_benchmarks.trading_benchmark import TradingBenchmark
from base_agent.src.benchmarks.base import Problem


async def test_strategy(strategy_dsl: str):
    """Test a DSL strategy on PURR data."""
    print("=" * 70)
    print("TESTING STRATEGY ON REAL PURR DATA")
    print("=" * 70)
    print(f"\nStrategy: {strategy_dsl}")
    print()

    # Create benchmark
    benchmark = TradingBenchmark()
    problem = benchmark.problems[0]

    # Create temporary test directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        workdir = tmppath / "workdir"
        answer_dir = tmppath / "answer"
        workdir.mkdir()
        answer_dir.mkdir()

        # Write the strategy
        answer_file = answer_dir / "answer.txt"
        answer_file.write_text(strategy_dsl + "\n")

        # Setup problem (copies PURR data)
        await benchmark.setup_problem(problem, workdir, "test_container")

        # Run backtest
        print("Running backtest on PURR data...")
        print()
        score, error, discussion = await benchmark.score_problem(
            problem, str(workdir), str(answer_dir), "test_container"
        )

        # Display results
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        if error:
            print(f"❌ Error: {error}")
        else:
            print(f"Fitness Score: ${score:.2f}")
            print()
            print("Details:")
            print(discussion)
            print()

            if score > 0:
                print("✅ Strategy SURVIVED (fitness > 0)")
            else:
                print("❌ Strategy DIED (fitness ≤ 0)")
        print("=" * 70)


async def main():
    print("\nTest 1: Simple momentum strategy")
    print("Strategy: IF DELTA(0) > DELTA(10) THEN BUY ELSE SELL")
    print("Logic: If current close > close 10 candles ago, buy. Otherwise sell.")
    await test_strategy("IF DELTA(0) > DELTA(10) THEN BUY ELSE SELL")

    print("\n" * 2)
    print("Test 2: Mean reversion strategy")
    print("Strategy: IF DELTA(0) < ALPHA(20) THEN BUY ELSE SELL")
    print("Logic: If current close < open 20 candles ago, buy (mean reversion). Otherwise sell.")
    await test_strategy("IF DELTA(0) < ALPHA(20) THEN BUY ELSE SELL")

    print("\n" * 2)
    print("Test 3: Volume-based strategy")
    print("Strategy: IF EPSILON(0) > EPSILON(5) THEN BUY ELSE HOLD")
    print("Logic: If current volume > volume 5 candles ago, buy. Otherwise hold.")
    await test_strategy("IF EPSILON(0) > EPSILON(5) THEN BUY ELSE HOLD")

    print("\n" * 2)
    print("Test 4: High vs Low strategy")
    print("Strategy: IF BETA(1) > GAMMA(1) THEN BUY ELSE SELL")
    print("Logic: If previous high > previous low (always true), buy. Otherwise sell.")
    await test_strategy("IF BETA(1) > GAMMA(1) THEN BUY ELSE SELL")


if __name__ == "__main__":
    asyncio.run(main())
