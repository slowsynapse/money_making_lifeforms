#!/bin/bash
# Test trading strategies inside Docker container with real PURR data

echo "========================================================================"
echo "TESTING STRATEGIES ON REAL PURR DATA (Inside Docker)"
echo "========================================================================"
echo ""

# Strategy 1: Simple momentum
echo "Test 1: Momentum Strategy"
echo "Strategy: IF DELTA(0) > DELTA(10) THEN BUY ELSE SELL"
echo "Logic: If current close > close 10 candles ago, BUY"
echo ""

docker run --rm \
  -v ${PWD}/base_agent:/home/agent/agent_code:ro \
  -v ${PWD}/benchmark_data:/home/agent/benchmark_data:ro \
  -v ${PWD}/results/test_output:/home/agent/workdir:rw \
  sica_sandbox \
  python -m agent_code.agent trading-test --strategy "IF DELTA(0) > DELTA(10) THEN BUY ELSE SELL"

echo ""
echo "========================================================================"
echo ""

# Strategy 2: Mean reversion
echo "Test 2: Mean Reversion Strategy"
echo "Strategy: IF DELTA(0) < ALPHA(20) THEN BUY ELSE SELL"
echo "Logic: If current close < open 20 candles ago, BUY (mean reversion)"
echo ""

docker run --rm \
  -v ${PWD}/base_agent:/home/agent/agent_code:ro \
  -v ${PWD}/benchmark_data:/home/agent/benchmark_data:ro \
  -v ${PWD}/results/test_output:/home/agent/workdir:rw \
  sica_sandbox \
  python -m agent_code.agent trading-test --strategy "IF DELTA(0) < ALPHA(20) THEN BUY ELSE SELL"

echo ""
echo "========================================================================"
echo ""

# Strategy 3: Volume breakout
echo "Test 3: Volume Breakout Strategy"
echo "Strategy: IF EPSILON(0) > EPSILON(5) THEN BUY ELSE HOLD"
echo "Logic: If current volume > volume 5 candles ago, BUY"
echo ""

docker run --rm \
  -v ${PWD}/base_agent:/home/agent/agent_code:ro \
  -v ${PWD}/benchmark_data:/home/agent/benchmark_data:ro \
  -v ${PWD}/results/test_output:/home/agent/workdir:rw \
  sica_sandbox \
  python -m agent_code.agent trading-test --strategy "IF EPSILON(0) > EPSILON(5) THEN BUY ELSE HOLD"

echo ""
echo "========================================================================"
echo "ALL TESTS COMPLETE"
echo "========================================================================"
