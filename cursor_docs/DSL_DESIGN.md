# Abstract Symbolic DSL: Design Rationale

## The Problem with Technical Indicators

Traditional algorithmic trading systems are built on **human-designed concepts**:
- SMA (Simple Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Fibonacci retracements

These concepts embody **decades of human bias** about how markets work. When we tell an agent to use "RSI," we're injecting the assumption that this particular mathematical transformation of price data is meaningful.

**But what if it's not?**

What if the most profitable patterns have nothing to do with moving averages or oscillators? What if they emerge from combinations that humans have never thought to name?

## The Abstract Symbolic Approach

Our DSL uses **meaningless symbols**:

```
ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA, OMEGA, PSI
```

These are just **labels**. They have no predefined meaning. The agent doesn't "know" that ALPHA(10) means "10-period moving average" or that GAMMA(14) means "14-period RSI."

### Why This Matters

1. **No Human Bias**: The system can't fall back on human trading folklore. It must discover what works through pure trial and error.

2. **Emergent Meaning**: If `IF ALPHA(10) > BETA(50) THEN BUY` consistently generates profit, the evolutionary process will favor it—not because of some theory about "golden crosses," but because it empirically works in the data.

3. **Freedom to Discover**: The symbols could end up representing:
   - Traditional indicators (if they're actually profitable)
   - Novel combinations of price/volume/order book data
   - Contextual factors we never thought to encode
   - Or nothing at all—random noise that happens to work in this specific market regime

4. **Tabula Rasa**: The agent starts with a blank slate. Every DSL strategy is equally plausible at generation 0. Only through market selection do certain patterns become dominant.

## How It Works

### Phase 1: Random Exploration
Early generations produce mostly random strategies:
```
IF ZETA(3) < OMEGA() THEN HOLD ELSE BUY
IF PSI() >= EPSILON(200) THEN SELL ELSE SELL
IF DELTA(7) == GAMMA(14) THEN BUY ELSE HOLD
```

Most of these will **die** (fitness < 0). A few might survive by luck.

### Phase 2: Mutation and Selection
Surviving strategies are mutated:
- Change operators: `>` → `<=`
- Change symbols: `ALPHA` → `BETA`
- Change parameters: `(10)` → `(20)`
- Change actions: `BUY` → `HOLD`

If a mutation improves fitness, it propagates. If it degrades fitness, it dies.

### Phase 3: Emergence (Long-term)
Over hundreds or thousands of generations, **patterns emerge**:
- Certain symbol combinations become dominant
- Certain parameter ranges become favored
- Certain operator/action pairings become common

**These patterns are discovered, not designed.**

## Contrast with Traditional Approaches

### Traditional Approach:
```python
# Human designs the strategy
if sma_10 > sma_50:
    action = "BUY"
else:
    action = "SELL"
```
→ Encodes human theory about trend-following.

### Our Approach:
```
IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL
```
→ Neutral symbolic representation. Fitness determines survival.

### The Key Difference:
In the traditional approach, the human **pre-selected** which indicators to use. In our approach, **the market selects** which symbol combinations survive.

## Future Extensions

### Multi-Level Rules
Currently: Single IF-THEN-ELSE
Future: Nested conditions, AND/OR logic

```
IF (ALPHA(10) > BETA(50) AND GAMMA(14) < 30) THEN BUY ELSE HOLD
```

### Dynamic Symbol Mapping
The symbols could eventually map to **actual implementations** that get discovered:
- ALPHA(N) → rolling mean of close prices over N periods
- BETA(N) → exponential moving average
- GAMMA(N) → volatility measure
- etc.

But initially, they're all just **placeholders** with dummy implementations. The interpreter can evolve alongside the strategies.

### Hybrid Search
Combine evolutionary search (DSL mutation) with:
- Gradient-free optimization (Covariance Matrix Adaptation)
- Genetic algorithms (crossover between strategies)
- Reinforcement learning (learn which mutations are promising)

## Philosophical Note

This system embodies a kind of **epistemic humility**:

> "We don't know what patterns actually work in markets. Let's not pretend we do by hardcoding RSI and Bollinger Bands. Instead, let's create a language for expressing strategies, and let natural selection figure out which ones are fit."

Traditional quant finance says: "Here are the tools (indicators). Go optimize their parameters."

Our approach says: "Here is a grammar. Go discover what the tools should even be."

## Testing the Concept

Run the system and monitor which symbol combinations dominate after N generations:

```bash
# After 100 generations
Top 5 Surviving Strategies:
1. IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL (fitness: $312.45)
2. IF GAMMA(14) < 30 THEN BUY ELSE HOLD (fitness: $287.33)
3. IF OMEGA() >= PSI() THEN HOLD ELSE SELL (fitness: $198.76)
4. IF DELTA(20) <= EPSILON(5) THEN SELL ELSE BUY (fitness: $173.21)
5. IF ZETA(7) > BETA(30) THEN BUY ELSE HOLD (fitness: $156.89)
```

**Question**: Did `ALPHA(10) > BETA(50)` succeed because it happens to behave like a moving average crossover? Or because it discovered something else?

**Answer**: We don't know, and that's the point. The market told us it works. The "why" is for humans to reverse-engineer after the fact.

## Summary

- **Abstract symbols** (ALPHA, BETA, GAMMA...) have no predefined meaning
- **Market selection** determines which combinations survive
- **No human bias** about indicators or strategies
- **Emergent discovery** of profitable patterns
- **Pure evolutionary computation** driven by real economic constraints

This is not "AI trading with technical indicators." This is **evolutionary search through strategy space with no priors**.
