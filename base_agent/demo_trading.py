#!/usr/bin/env python3
"""
Quick demo of the trading evolution system.
Run this inside the Docker container with: python test_trading_demo.py
"""

import sys
from pathlib import Path
import tempfile

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.benchmarks.trading_benchmarks.trading_benchmark import TradingBenchmark
from src.dsl.interpreter import DslInterpreter
from src.dsl.mutator import DslMutator

def demo_dsl_parsing():
    """Demonstrate DSL parsing."""
    print("=" * 60)
    print("1. DSL PARSING DEMO")
    print("=" * 60)
    
    interpreter = DslInterpreter()
    
    strategies = [
        "IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL",
        "IF GAMMA(14) < 30 THEN BUY ELSE HOLD",
        "IF OMEGA() >= PSI() THEN HOLD ELSE SELL",
    ]
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        program = interpreter.parse(strategy)
        if program:
            print(f"  ✓ Parsed successfully!")
            rule = program[0]
            print(f"  - Condition: {rule.condition.indicator1.name}({rule.condition.param1}) "
                  f"{rule.condition.operator.value} {rule.condition.indicator2.name}({rule.condition.param2})")
            print(f"  - Actions: {rule.true_action.name} / {rule.false_action.name}")
        else:
            print(f"  ✗ Failed to parse")
    print()

def demo_dsl_mutation():
    """Demonstrate DSL mutation."""
    print("=" * 60)
    print("2. DSL MUTATION DEMO")
    print("=" * 60)
    
    interpreter = DslInterpreter()
    mutator = DslMutator()
    
    base_strategy = "IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL"
    print(f"\nBase Strategy: {base_strategy}")
    
    program = interpreter.parse(base_strategy)
    
    print("\nGenerating 5 mutations:")
    for i in range(5):
        mutated = mutator.mutate(program)
        mutated_str = mutator.to_string(mutated)
        print(f"  Mutation {i+1}: {mutated_str}")
        # Parse the original again for next mutation (since mutate modifies in place)
        program = interpreter.parse(base_strategy)
    print()

def demo_trading_benchmark():
    """Demonstrate the trading benchmark."""
    print("=" * 60)
    print("3. TRADING BENCHMARK DEMO")
    print("=" * 60)
    
    benchmark = TradingBenchmark()
    
    print(f"\nBenchmark name: {benchmark.name}")
    print(f"Number of problems: {len(benchmark.problems)}")
    
    # Show the problem statement (truncated)
    problem = benchmark.problems[0]
    print(f"\nProblem ID: {problem.problem_id}")
    print(f"Problem statement (first 200 chars):")
    print(problem.statement[:200] + "...")
    
    print("\n✓ Trading benchmark loaded successfully!")
    print()

def demo_strategy_mutation():
    """Demonstrate generating a mutated strategy file."""
    print("=" * 60)
    print("4. STRATEGY FILE MUTATION DEMO")
    print("=" * 60)
    
    benchmark = TradingBenchmark()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create base strategy
        base_file = tmppath / "base.txt"
        base_file.write_text("IF GAMMA(20) <= DELTA(100) THEN SELL ELSE BUY\n")
        
        print(f"\nBase strategy: {base_file.read_text().strip()}")
        
        # Mutate it 3 times
        print("\nGenerating mutations:")
        for i in range(3):
            output_file = tmppath / f"mutated_{i}.txt"
            success = benchmark.generate_mutated_strategy(base_file, output_file)
            if success:
                print(f"  Mutation {i+1}: {output_file.read_text().strip()}")
            else:
                print(f"  Mutation {i+1}: Failed")
            
            # Use the mutated version as the base for the next mutation
            base_file = output_file
    
    print("\n✓ Strategy mutation working correctly!")
    print()

def demo_fitness_concept():
    """Explain the fitness concept."""
    print("=" * 60)
    print("5. FITNESS & SURVIVAL CONCEPT")
    print("=" * 60)
    
    print("""
The evolutionary loop works as follows:

1. GENERATION 0: Random initial strategy
   Example: IF PSI() < EPSILON(50) THEN BUY ELSE HOLD
   
2. BACKTEST: Run strategy on historical OHLCV data
   - Start with $10,000 capital
   - Execute BUY/SELL/HOLD actions based on DSL
   - Track portfolio value at every tick
   
3. FITNESS CALCULATION:
   Fitness = Trading Profit - Transaction Costs - LLM API Costs
   
   Example outcomes:
   • Strategy makes $200, costs $0.02 → Fitness = $199.98 ✓ SURVIVED
   • Strategy makes $50, costs $0.02 → Fitness = $49.98 ✓ SURVIVED  
   • Strategy loses $500, costs $0.02 → Fitness = -$500.02 ✗ DIED
   • Balance hits $0 mid-backtest → Fitness = -$10,000 ✗ DIED
   
4. SELECTION: Only strategies with Fitness > 0 survive

5. MUTATION: Best survivor is mutated to create next generation

6. REPEAT: New strategy is tested, and the cycle continues

Over 100+ generations, profitable patterns emerge through pure
natural selection. No human tells the system what "should" work.
The market is the teacher.
""")

def main():
    print("\n" + "=" * 60)
    print("TRADING EVOLUTION SYSTEM - INTERACTIVE DEMO")
    print("=" * 60)
    print()
    
    try:
        demo_dsl_parsing()
        demo_dsl_mutation()
        demo_trading_benchmark()
        demo_strategy_mutation()
        demo_fitness_concept()
        
        print("=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print("""
The system is ready for evolution. To run the full loop:
1. Exit this container
2. On your host machine, run:
   python3 runner.py --evolution-mode --iterations 10

This will evolve trading strategies through natural selection!
""")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
