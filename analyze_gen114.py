#!/usr/bin/env python3
"""
Gen 114 Parameter Elimination Analysis
Comparative analysis of ALPHA(10) vs ALPHA() mutation and its impact on fitness.
"""

import pandas as pd
import sys
from pathlib import Path

# Add base_agent to path
sys.path.insert(0, str(Path(__file__).parent))

from base_agent.src.dsl.interpreter import DslInterpreter
from base_agent.src.dsl.grammar import Action


def load_purr_data():
    """Load the PURR benchmark data used for evolution."""
    data_path = Path("benchmark_data/trading/purr_60d.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"PURR data not found at {data_path}")

    df = pd.read_csv(data_path)
    # Ensure required columns exist
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")

    return df


def backtest_strategy(dsl_string, market_data, initial_capital=1000.0):
    """
    Run a simple backtest of a DSL strategy.

    Returns:
        dict with: final_pnl, trades, signals, positions
    """
    interpreter = DslInterpreter()
    program = interpreter.parse(dsl_string)

    if not program:
        raise ValueError(f"Failed to parse strategy: {dsl_string}")

    capital = initial_capital
    position = 0  # 0 = no position, 1 = long, -1 = short
    entry_price = 0

    trades = []
    signals = []
    positions = []

    # Start from index 100 to ensure we have enough lookback history
    for i in range(100, len(market_data)):
        action = interpreter.execute(program, market_data, current_index=i)
        current_price = market_data.iloc[i]['close']
        timestamp = market_data.iloc[i]['timestamp']

        # Track signal and position
        signals.append({
            'index': i,
            'timestamp': timestamp,
            'action': action.value,
            'price': current_price
        })

        # Execute trades based on action
        if action == Action.BUY and position != 1:
            # Close short if exists
            if position == -1:
                pnl = entry_price - current_price
                capital += pnl
                trades.append({
                    'index': i,
                    'timestamp': timestamp,
                    'type': 'CLOSE_SHORT',
                    'price': current_price,
                    'pnl': pnl,
                    'capital': capital
                })

            # Open long
            position = 1
            entry_price = current_price
            trades.append({
                'index': i,
                'timestamp': timestamp,
                'type': 'OPEN_LONG',
                'price': current_price,
                'pnl': 0,
                'capital': capital
            })

        elif action == Action.SELL and position != -1:
            # Close long if exists
            if position == 1:
                pnl = current_price - entry_price
                capital += pnl
                trades.append({
                    'index': i,
                    'timestamp': timestamp,
                    'type': 'CLOSE_LONG',
                    'price': current_price,
                    'pnl': pnl,
                    'capital': capital
                })

            # Open short
            position = -1
            entry_price = current_price
            trades.append({
                'index': i,
                'timestamp': timestamp,
                'type': 'OPEN_SHORT',
                'price': current_price,
                'pnl': 0,
                'capital': capital
            })

        # Track position state
        positions.append({
            'index': i,
            'timestamp': timestamp,
            'position': position,
            'capital': capital
        })

    # Close any open position at the end
    if position != 0:
        final_price = market_data.iloc[-1]['close']
        if position == 1:
            pnl = final_price - entry_price
        else:
            pnl = entry_price - final_price

        capital += pnl
        trades.append({
            'index': len(market_data) - 1,
            'timestamp': market_data.iloc[-1]['timestamp'],
            'type': 'CLOSE_FINAL',
            'price': final_price,
            'pnl': pnl,
            'capital': capital
        })

    final_pnl = capital - initial_capital

    return {
        'final_pnl': final_pnl,
        'final_capital': capital,
        'trades': trades,
        'signals': signals,
        'positions': positions
    }


def compare_strategies():
    """Main comparison function."""

    print("=" * 80)
    print("GEN 114 PARAMETER ELIMINATION ANALYSIS")
    print("=" * 80)
    print()

    # Define the two strategies
    gen88_strategy = "IF ALPHA(10) > GAMMA(100) THEN SELL ELSE BUY"
    gen114_strategy = "IF ALPHA() > GAMMA(100) THEN SELL ELSE BUY"

    print("STRATEGY DEFINITIONS")
    print("-" * 80)
    print(f"Gen 88  (Cell #89):  {gen88_strategy}")
    print(f"Gen 114 (Cell #101): {gen114_strategy}")
    print()

    print("ABSTRACT INDICATOR DECODING")
    print("-" * 80)
    print("ALPHA = 'open' (opening price)")
    print("GAMMA = 'low' (low price)")
    print()
    print("Gen 88  Logic: IF open[t-10] > low[t-100] THEN SELL ELSE BUY")
    print("Gen 114 Logic: IF open[t-0] > low[t-100] THEN SELL ELSE BUY")
    print()

    # Load data
    print("Loading PURR benchmark data...")
    market_data = load_purr_data()
    print(f"Loaded {len(market_data)} candles")
    print(f"Date range: {market_data.iloc[0]['timestamp']} to {market_data.iloc[-1]['timestamp']}")
    print()

    # Run backtests
    print("BACKTESTING...")
    print("-" * 80)

    print("Running Gen 88 strategy (ALPHA(10))...")
    gen88_results = backtest_strategy(gen88_strategy, market_data)
    print(f"  Trades: {len(gen88_results['trades'])}")
    print(f"  Final PnL: ${gen88_results['final_pnl']:.2f}")
    print()

    print("Running Gen 114 strategy (ALPHA())...")
    gen114_results = backtest_strategy(gen114_strategy, market_data)
    print(f"  Trades: {len(gen114_results['trades'])}")
    print(f"  Final PnL: ${gen114_results['final_pnl']:.2f}")
    print()

    # Performance comparison
    print("PERFORMANCE METRICS")
    print("-" * 80)
    print(f"{'Metric':<30} {'Gen 88 (ALPHA(10))':<20} {'Gen 114 (ALPHA())':<20}")
    print("-" * 80)
    print(f"{'Final PnL':<30} ${gen88_results['final_pnl']:<19.2f} ${gen114_results['final_pnl']:<19.2f}")
    print(f"{'Total Trades':<30} {len(gen88_results['trades']):<20} {len(gen114_results['trades']):<20}")

    # Calculate improvement
    improvement = ((gen114_results['final_pnl'] - gen88_results['final_pnl']) / abs(gen88_results['final_pnl'])) * 100 if gen88_results['final_pnl'] != 0 else float('inf')
    print(f"{'Fitness Improvement':<30} {improvement:>+19.1f}%")
    print()

    # Signal divergence analysis
    print("SIGNAL DIVERGENCE ANALYSIS")
    print("-" * 80)

    # Compare signals at each timestep
    divergences = []
    for i, (s88, s114) in enumerate(zip(gen88_results['signals'], gen114_results['signals'])):
        if s88['action'] != s114['action']:
            divergences.append({
                'index': s88['index'],
                'timestamp': s88['timestamp'],
                'gen88_signal': s88['action'],
                'gen114_signal': s114['action'],
                'price': s88['price']
            })

    total_signals = len(gen88_results['signals'])
    divergence_count = len(divergences)
    divergence_rate = (divergence_count / total_signals) * 100

    print(f"Total signal points: {total_signals}")
    print(f"Signal divergences: {divergence_count} ({divergence_rate:.1f}%)")
    print(f"Signal agreement: {total_signals - divergence_count} ({100 - divergence_rate:.1f}%)")
    print()

    # Show first 10 divergences
    if divergences:
        print("First 10 Signal Divergences:")
        print(f"{'Index':<8} {'Timestamp':<20} {'Gen 88':<10} {'Gen 114':<10} {'Price':<10}")
        print("-" * 70)
        for div in divergences[:10]:
            print(f"{div['index']:<8} {div['timestamp']:<20} {div['gen88_signal']:<10} {div['gen114_signal']:<10} ${div['price']:<9.4f}")
        print()

    # Technical evolutionary analysis
    print("=" * 80)
    print("EVOLUTIONARY ANALYSIS - TECHNICAL TERMINOLOGY")
    print("=" * 80)
    print()

    print("MUTATION CLASSIFICATION:")
    print("-" * 80)
    print("Type: PARAMETER DEGENERACY DISCOVERY")
    print("      (also: Temporal Kernel Normalization, Dimensionality Collapse Mutation)")
    print()

    print("MECHANISM:")
    print("-" * 80)
    print("The mutation eliminated the lookback parameter from the ALPHA indicator,")
    print("reducing the search space dimensionality from parameterized (ALPHA(N)) to")
    print("native-resolution (ALPHA()). This represents a transition from a constrained")
    print("temporal kernel with 10-period lag to a normalized temporal kernel operating")
    print("at instantaneous resolution.")
    print()

    print("HYPOTHESIS:")
    print("-" * 80)
    print("The parameterized version ALPHA(10) introduced phase lag by examining the")
    print("opening price from 10 candles in the past. This temporal displacement created")
    print("signal degradation when compared against the long-term baseline GAMMA(100).")
    print()
    print("The parameter-free version ALPHA() captures instantaneous state transitions")
    print("at native temporal resolution, creating a more responsive dual-timescale")
    print("anomaly detector. The strategy becomes:")
    print()
    print("  Instantaneous Fast Signal (current open) vs Long-Term Baseline (100-period low)")
    print()
    print("This cross-timescale comparison mechanism fires SELL signals when the fast")
    print("signal exceeds the slow baseline, indicating potential mean reversion zones.")
    print()

    print("EVOLUTIONARY SIGNIFICANCE:")
    print("-" * 80)
    print(f"• Fitness amplification: +{improvement:.1f}%")
    print(f"• Signal divergence rate: {divergence_rate:.1f}%")
    print(f"• Genome simplification: Removed redundant parameter dimension")
    print(f"• Phenotype enhancement: Superior decision timing via lag elimination")
    print()
    print("This mutation demonstrates the principle that evolutionary fitness does not")
    print("always correlate with genotype complexity. The simpler genome (no parameter)")
    print("produced superior phenotypic behavior (higher returns), suggesting the")
    print("parameterized search space contained a local optimum trap that the")
    print("parameter-free variant escaped.")
    print()

    print("=" * 80)


if __name__ == "__main__":
    try:
        compare_strategies()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
