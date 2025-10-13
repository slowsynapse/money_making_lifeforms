#!/usr/bin/env python3
"""Test the new recursive descent parser for compound conditions."""

from base_agent.src.dsl.interpreter import DslInterpreter
from base_agent.src.dsl.grammar import CompoundCondition, Condition, LogicalOp

def test_parser():
    interpreter = DslInterpreter()

    # Test 1: V1 Simple condition (backward compatibility)
    print("Test 1: V1 Simple condition")
    program = interpreter.parse("IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL")
    if program:
        print("✓ Parsed successfully")
        print(f"  String: {interpreter.to_string(program)}")
        assert isinstance(program[0].condition, Condition)
    else:
        print("✗ Failed to parse")
    print()

    # Test 2: Compound AND condition
    print("Test 2: Compound AND condition")
    program = interpreter.parse("IF ALPHA() > GAMMA(100) AND BETA(0) > DELTA(50) THEN SELL ELSE BUY")
    if program:
        print("✓ Parsed successfully")
        print(f"  String: {interpreter.to_string(program)}")
        assert isinstance(program[0].condition, CompoundCondition)
        assert program[0].condition.op == LogicalOp.AND
    else:
        print("✗ Failed to parse")
    print()

    # Test 3: Compound OR condition
    print("Test 3: Compound OR condition")
    program = interpreter.parse("IF ALPHA(5) > BETA(10) OR GAMMA(20) < DELTA(30) THEN BUY ELSE HOLD")
    if program:
        print("✓ Parsed successfully")
        print(f"  String: {interpreter.to_string(program)}")
        assert isinstance(program[0].condition, CompoundCondition)
        assert program[0].condition.op == LogicalOp.OR
    else:
        print("✗ Failed to parse")
    print()

    # Test 4: NOT condition
    print("Test 4: NOT condition")
    program = interpreter.parse("IF NOT OMEGA() < 0 THEN BUY ELSE SELL")
    if program:
        print("✓ Parsed successfully")
        print(f"  String: {interpreter.to_string(program)}")
        assert isinstance(program[0].condition, CompoundCondition)
        assert program[0].condition.op == LogicalOp.NOT
    else:
        print("✗ Failed to parse")
    print()

    # Test 5: Nested conditions with parentheses
    print("Test 5: Nested conditions with parentheses")
    program = interpreter.parse("IF (ALPHA() > GAMMA(100) OR BETA() > DELTA(50)) AND EPSILON() > 1000 THEN BUY ELSE HOLD")
    if program:
        print("✓ Parsed successfully")
        print(f"  String: {interpreter.to_string(program)}")
        assert isinstance(program[0].condition, CompoundCondition)
        assert program[0].condition.op == LogicalOp.AND
    else:
        print("✗ Failed to parse")
    print()

    # Test 6: Triple AND (left-associative)
    print("Test 6: Triple AND (left-associative)")
    program = interpreter.parse("IF ALPHA() > BETA() AND GAMMA() < DELTA() AND EPSILON() > 100 THEN BUY ELSE SELL")
    if program:
        print("✓ Parsed successfully")
        print(f"  String: {interpreter.to_string(program)}")
    else:
        print("✗ Failed to parse")
    print()

    # Test 7: Mixed AND/OR (proper precedence: AND binds tighter)
    print("Test 7: Mixed AND/OR (AND has higher precedence)")
    program = interpreter.parse("IF ALPHA() > BETA() OR GAMMA() < DELTA() AND EPSILON() > 100 THEN BUY ELSE SELL")
    if program:
        print("✓ Parsed successfully")
        print(f"  String: {interpreter.to_string(program)}")
        # Should parse as: ALPHA() > BETA() OR (GAMMA() < DELTA() AND EPSILON() > 100)
    else:
        print("✗ Failed to parse")
    print()

    # Test 8: Empty params (OMEGA style)
    print("Test 8: Empty params (OMEGA style)")
    program = interpreter.parse("IF OMEGA() > 0 THEN BUY ELSE SELL")
    if program:
        print("✓ Parsed successfully")
        print(f"  String: {interpreter.to_string(program)}")
    else:
        print("✗ Failed to parse")
    print()

    print("=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    test_parser()
