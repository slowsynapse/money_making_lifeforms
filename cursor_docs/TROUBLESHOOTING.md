# Troubleshooting Guide

## Where is `answer.txt` located?

### Directory Structure

When running `trading-learn`, the files are organized like this:

```
/home/agent/  (inside Docker container)
└── workdir/
    ├── iteration_0/
    │   ├── ohlcv.csv              # Market data (copied from benchmark_data)
    │   └── logs/
    │       ├── answer.txt         # ← AGENT'S GENERATED STRATEGY
    │       ├── agent_report.md    # Execution report
    │       └── events/            # Event logs
    ├── iteration_1/
    │   ├── ohlcv.csv
    │   └── logs/
    │       └── answer.txt         # ← Strategy for iteration 1
    └── iteration_2/
        └── ...
```

**Key Point**: The agent writes `answer.txt` to the **logdir**, not the workdir!

### On Your Host Machine

If you run:
```bash
docker run --rm \
  -v "$(pwd)/base_agent":/home/agent/agent_code \
  -v "$(pwd)/benchmark_data":/home/agent/benchmark_data \
  sica_sandbox \
  python -m agent_code.agent trading-learn --iterations 3
```

Then `answer.txt` will be at:
```
/home/agent/workdir/iteration_0/logs/answer.txt  (inside container)
```

**But this is ephemeral!** Since we don't mount `/home/agent/workdir`, it disappears when the container exits.

### Solution: Mount a Persistent Workdir

```bash
# Create a local results directory
mkdir -p results/trading_learn

# Mount it as the workdir
docker run --rm \
  -v "$(pwd)/base_agent":/home/agent/agent_code \
  -v "$(pwd)/benchmark_data":/home/agent/benchmark_data \
  -v "$(pwd)/results/trading_learn":/home/agent/workdir \
  --env-file .env \
  sica_sandbox \
  python -m agent_code.agent trading-learn --iterations 3 --server
```

Now you can find it on your host at:
```
./results/trading_learn/iteration_0/logs/answer.txt
./results/trading_learn/iteration_1/logs/answer.txt
./results/trading_learn/iteration_2/logs/answer.txt
```

## Common Issues

### Issue 1: "Agent failed to create answer.txt"

**Symptom:**
```
❌ Agent failed to create answer.txt
```

**Causes:**
1. **Agent timed out** before generating a strategy
2. **LLM didn't call the submission tool** (prompt engineering issue)
3. **Path mismatch** (logdir vs workdir)

**Solutions:**
- Increase timeout: `--timeout 300` (5 minutes)
- Check the event log in `logs/events/` to see what the agent did
- Ensure the problem statement clearly instructs to use `overwrite_file` tool

### Issue 2: "No surviving strategies"

**Symptom:**
```
❌ No surviving strategies found
Final Balance: $78.34
```

**Cause:** All generated strategies had negative fitness (lost money or didn't cover LLM costs).

**What to check:**
1. **View the strategies**: Open each `iteration_N/logs/answer.txt`
2. **Check if they're valid DSL**: Should be `IF ALPHA(N) > BETA(M) THEN BUY ELSE HOLD`
3. **Look at the backtest results** in terminal output

**Example of a bad strategy:**
```
IF ALPHA(10) < BETA(50) THEN SELL ELSE SELL
```
This never buys, so profit = $0, fitness = -$0.02 (LLM cost).

### Issue 3: Agent goes bankrupt too quickly

**Symptom:**
```
💰 AGENT STATE AFTER ITERATION 5:
   Current Balance: $0.00
   
⚠️ WARNING: Agent balance depleted! Agent would be bankrupt.
```

**Cause:** Starting capital ($100) is too low relative to costs.

**Solutions:**

**Option A: Increase starting capital**
```python
# In base_agent/src/benchmarks/trading_benchmarks/trading_benchmark.py
INITIAL_CAPITAL: ClassVar[float] = 500.0  # More runway
```

**Option B: Use a cheaper LLM**
```python
# In your .env file
LLM_PROVIDER=deepseek  # Much cheaper than Anthropic
```

**Option C: Reduce transaction costs**
```python
TRANSACTION_COST: ClassVar[float] = 0.01  # Lower fees
```

### Issue 4: Web interface shows nothing

**Symptom:** Browser loads but execution tree is empty.

**Cause:** Agent execution is very fast, or hasn't started yet.

**Solutions:**
1. **Refresh the page** during agent execution
2. Check terminal - is the agent actually running?
3. Check browser console for errors (F12)

### Issue 5: "Port 8080 already in use"

**Symptom:**
```
OSError: [Errno 98] Address already in use
```

**Cause:** Another process is using port 8080.

**Solution:**
```bash
# Find and kill the process
lsof -ti:8080 | xargs kill -9

# Or use a different port (requires code change)
```

### Issue 6: Can't see strategies in terminal

**Symptom:** Want to see what the agent generated, but terminal scrolled too far.

**Solution:**
```bash
# Mount a persistent workdir
docker run --rm \
  -v "$(pwd)/results/trading_learn":/home/agent/workdir \
  ...

# Then read the files directly
cat results/trading_learn/iteration_0/logs/answer.txt
cat results/trading_learn/iteration_1/logs/answer.txt
```

## Debugging Checklist

When things go wrong:

- [ ] **Check if Docker container is running**: `docker ps`
- [ ] **Verify environment variables**: `.env` file exists and has API keys
- [ ] **Check logs directory**: Is it being created?
- [ ] **Read answer.txt files**: Are strategies being generated?
- [ ] **Check terminal output**: What does the fitness progression show?
- [ ] **Inspect agent_report.md**: Full execution details
- [ ] **Monitor balance**: Is the agent going bankrupt?
- [ ] **Web interface**: Is it accessible at localhost:8080?

## File Locations Reference

| File | Purpose | Location (in container) |
|------|---------|-------------------------|
| `answer.txt` | Agent's generated strategy | `/home/agent/workdir/iteration_N/logs/answer.txt` |
| `agent_report.md` | Execution summary | `/home/agent/workdir/iteration_N/logs/agent_report.md` |
| `ohlcv.csv` | Market data | `/home/agent/workdir/iteration_N/ohlcv.csv` |
| Events | Tool calls, LLM interactions | `/home/agent/workdir/iteration_N/logs/events/` |

## Getting More Detail

### View Full Execution Report

```bash
# Inside the mounted workdir
cat results/trading_learn/iteration_0/logs/agent_report.md
```

### View All Generated Strategies

```bash
for i in {0..4}; do
  echo "=== Iteration $i ==="
  cat results/trading_learn/iteration_$i/logs/answer.txt
done
```

### Check What Tools Were Called

```bash
ls results/trading_learn/iteration_0/logs/events/
cat results/trading_learn/iteration_0/logs/events/*.json
```

This shows every tool call, LLM message, and system event during execution.
