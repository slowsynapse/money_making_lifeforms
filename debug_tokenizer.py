#!/usr/bin/env python3
"""Debug tokenizer to see what tokens are produced."""

from base_agent.src.dsl.interpreter import DslInterpreter

def debug_tokenize(text):
    interpreter = DslInterpreter()
    tokens = interpreter._tokenize(text)
    print(f"Input: {text}")
    print(f"Tokens: {tokens}")
    print()

# Test cases that failed
debug_tokenize("IF NOT OMEGA() < 0 THEN BUY ELSE SELL")
debug_tokenize("IF (ALPHA() > GAMMA(100) OR BETA() > DELTA(50)) AND EPSILON() > 1000 THEN BUY ELSE HOLD")
debug_tokenize("IF OMEGA() > 0 THEN BUY ELSE SELL")
