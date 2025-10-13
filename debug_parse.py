#!/usr/bin/env python3
"""Debug parser step by step."""

from base_agent.src.dsl.interpreter import DslInterpreter

def debug_parse(text):
    interpreter = DslInterpreter()

    # First tokenize
    tokens = interpreter._tokenize(text)
    print(f"Input: {text}")
    print(f"Tokens: {tokens}")

    # Try to parse
    try:
        program = interpreter.parse(text)
        if program:
            print(f"✓ SUCCESS")
            print(f"  Output: {interpreter.to_string(program)}")
        else:
            print(f"✗ FAILED: parse() returned None")
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    print()

# Test the failing cases one by one
debug_parse("IF NOT OMEGA() < 0 THEN BUY ELSE SELL")
debug_parse("IF OMEGA() > 0 THEN BUY ELSE SELL")
debug_parse("IF ALPHA() > BETA() AND GAMMA() < DELTA() AND EPSILON() > 100 THEN BUY ELSE SELL")
