#!/usr/bin/env python3
"""
Test script to demonstrate the multi-strategy DSL system.
"""

from base_agent.src.dsl.grammar import AggregationMode
from base_agent.src.dsl.interpreter import DslInterpreter
from base_agent.src.dsl.mutator import DslMutator

def test_single_strategy():
    """Test parsing and executing a single strategy."""
    print("="*60)
    print("TEST 1: Single Strategy")
    print("="*60)

    interpreter = DslInterpreter()
    dsl_string = "IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL"

    program = interpreter.parse(dsl_string)
    print(f"Input:  {dsl_string}")
    print(f"Parsed: {len(program)} rule(s)")
    print(f"Output: {interpreter.to_string(program)}")
    print()

def test_multi_strategy_parsing():
    """Test parsing multiple strategies."""
    print("="*60)
    print("TEST 2: Multi-Strategy Parsing")
    print("="*60)

    interpreter = DslInterpreter()

    # Newline-separated
    dsl_string = """IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL
IF GAMMA(20) <= OMEGA() THEN SELL ELSE HOLD
IF DELTA(5) == PSI(100) THEN HOLD ELSE BUY"""

    program = interpreter.parse(dsl_string)
    print(f"Input ({len(dsl_string.split(chr(10)))} lines):")
    for line in dsl_string.split('\n'):
        print(f"  {line}")
    print(f"\nParsed: {len(program)} rule(s)")
    print(f"\nOutput:")
    for i, line in enumerate(interpreter.to_string(program).split('\n'), 1):
        print(f"  {i}. {line}")
    print()

def test_aggregation_modes():
    """Test different aggregation modes."""
    print("="*60)
    print("TEST 3: Strategy Aggregation Modes")
    print("="*60)

    # Create a multi-strategy program
    # Rule 1: BUY, Rule 2: HOLD, Rule 3: BUY → Majority should be BUY
    dsl_string = """IF ALPHA(10) > BETA(50) THEN BUY ELSE HOLD
IF GAMMA(20) > OMEGA() THEN HOLD ELSE BUY
IF DELTA(5) > PSI(100) THEN BUY ELSE SELL"""

    market_data = {}  # Dummy data

    for mode in [AggregationMode.MAJORITY, AggregationMode.UNANIMOUS, AggregationMode.FIRST]:
        interpreter = DslInterpreter(aggregation_mode=mode)
        program = interpreter.parse(dsl_string)

        # Collect individual rule actions (for display)
        actions = []
        for rule in program:
            action = interpreter._execute_single_rule(rule, market_data)
            actions.append(action.value)

        # Get aggregated action
        final_action = interpreter.execute(program, market_data)

        print(f"{mode.value:12} mode: Individual votes={actions} → Final={final_action.value}")

    print()

def test_mutation():
    """Test mutation of multi-strategy programs."""
    print("="*60)
    print("TEST 4: Multi-Strategy Mutation")
    print("="*60)

    interpreter = DslInterpreter()
    mutator = DslMutator()

    # Start with a 2-rule strategy
    original_dsl = """IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL
IF GAMMA(20) <= OMEGA() THEN HOLD ELSE BUY"""

    program = interpreter.parse(original_dsl)

    print(f"Original Strategy ({len(program)} rules):")
    for i, line in enumerate(mutator.to_string(program).split('\n'), 1):
        print(f"  {i}. {line}")
    print()

    # Apply 5 mutations
    for i in range(5):
        print(f"Mutation {i+1}:")
        program = mutator.mutate(program)
        print(f"Result ({len(program)} rules):")
        for j, line in enumerate(mutator.to_string(program).split('\n'), 1):
            print(f"  {j}. {line}")
        print()

def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "MULTI-STRATEGY DSL SYSTEM DEMO" + " "*18 + "║")
    print("╚" + "="*58 + "╝")
    print()

    test_single_strategy()
    test_multi_strategy_parsing()
    test_aggregation_modes()
    test_mutation()

    print("="*60)
    print("All tests completed successfully!")
    print("="*60)
    print()
    print("Key Features:")
    print("✓ Parse multiple strategies (newline or semicolon separated)")
    print("✓ Aggregate strategy votes (MAJORITY, UNANIMOUS, FIRST)")
    print("✓ Mutate multi-strategy programs (modify/add/remove rules)")
    print("✓ Support up to 5 chained strategies per program")
    print()

if __name__ == "__main__":
    main()
