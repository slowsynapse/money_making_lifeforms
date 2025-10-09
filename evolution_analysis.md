# Evolution Run #1: Analysis & DSL Limitations

**Date**: 2025-10-09
**Generations**: 20 (stopped by stagnation detection)
**Best Fitness**: $6.17
**Best Strategy**: `IF EPSILON(20) <= DELTA(50) THEN HOLD ELSE BUY`

---

## Critical Problem Identified: **DSL is Too Primitive**

You're absolutely right - the DSL is missing the ability to create **true AI-made technical indicators**. Here's why:

###  1. **No Mathematical Operations**

**Current Reality:**
```
IF EPSILON(20) <= DELTA(50) THEN HOLD ELSE BUY
```
This compares: `volume[t-20] <= close[t-50]`

**Problem**: Comparing volume to price is nonsensical! They're different units/magnitudes.

**What We Need**: Math operations to create meaningful indicators
```
IF (DELTA(0) - DELTA(10)) / DELTA(10) > 0.05 THEN BUY ELSE HOLD
```
This creates a **momentum indicator**: 10-period percent change

---

### 2. **No Derived Indicators**

**Current**: Can only access raw data directly
- ALPHA = open
- DELTA = close
- EPSILON = volume

**Missing**: Ability to compute transformations
- Moving averages: `(DELTA(0) + DELTA(1) + ... + DELTA(N)) / N`
- Ratios: `DELTA(0) / ALPHA(0)` (close/open ratio for candle body size)
- Differences: `BETA(0) - GAMMA(0)` (high - low = candle range)
- Momentum: `DELTA(0) - DELTA(N)` (price change over N periods)

---

### 3. **Stagnation at Local Maximum**

**Observation**: Gen 0 found `$6.17` and never improved for 20 generations.

**Why?**
- Random mutations mostly make things worse
- The fitness landscape is **sparse** - most random combinations are garbage
- Example: `EPSILON(20) <= DELTA(50)` (volume vs price) happens to work by luck

**The Real Issue**: The DSL can't express intermediate concepts like:
- "Is price trending up?"
- "Is volatility increasing?"
- "Is volume confirming the price move?"

---

## What Evolution Tried (and Why It Failed)

### Attempted Mutations:

1. **Operator flips**: `<=` → `>=` (Gen 4, 15, 19)
   - Result: Failed (-$0.02)
   - Why: Inverts logic without understanding context

2. **Symbol substitutions**: EPSILON → ZETA, BETA (Gen 1, 9)
   - Result: No change ($6.17)
   - Why: Substituting one raw column for another doesn't create new patterns

3. **Adding second rules** (Gen 7, 11, 14, 16, 18, 20)
   - Result: All failed (-$0.02)
   - Why: Second rule votes with first, but random rules mostly HOLD or cause ties

4. **Action changes**: HOLD → BUY, SELL (Gen 2, 3, 5, 6, 13)
   - Result: No change ($6.17)
   - Why: `THEN HOLD ELSE BUY` vs `THEN BUY ELSE BUY` both result in mostly buying

---

## The Core Missing Pieces

### **Missing Feature #1: Arithmetic Expressions**

Need to support expressions like:
```python
# Momentum: price change as percentage
(DELTA(0) - DELTA(10)) / DELTA(10)

# Candle body ratio
(DELTA(0) - ALPHA(0)) / (BETA(0) - GAMMA(0))

# Volume-weighted price
EPSILON(0) * DELTA(0)

# Moving average (simplified)
(DELTA(0) + DELTA(1) + DELTA(2)) / 3
```

**DSL Extension Needed**:
```
IF (DELTA(0) - DELTA(10)) / DELTA(10) > 0.05 THEN BUY ELSE HOLD
```

---

### **Missing Feature #2: Aggregation Functions**

Currently can only look at individual candles. Need:
```python
# Sum over window
SUM(DELTA, 0, 10)  # Sum of close prices from t-0 to t-10

# Average
AVG(DELTA, 0, 10)  # 10-period moving average

# Max/Min
MAX(BETA, 0, 20)   # Highest high in last 20 candles
MIN(GAMMA, 0, 20)  # Lowest low in last 20 candles

# Standard deviation (volatility)
STD(DELTA, 0, 14)  # 14-period volatility
```

**DSL Extension Needed**:
```
IF DELTA(0) > AVG(DELTA, 0, 20) THEN BUY ELSE HOLD
```
This creates "buy when price > 20-MA" strategy

---

### **Missing Feature #3: Derived Symbols**

Instead of just mapping to raw columns, symbols could represent **computed indicators**:

```python
# Define what each symbol computes
ALPHA = lambda t, n: (close[t] - close[t-n]) / close[t-n]  # Momentum
BETA = lambda t, n: (high[t] - low[t]) / close[t]          # Volatility proxy
GAMMA = lambda t, n: sum(volume[t-n:t]) / n                 # Avg volume
DELTA = lambda t, n: (close[t] - open[t]) / open[t]         # Body ratio
```

Then strategies like:
```
IF ALPHA(0, 10) > 0.05 THEN BUY ELSE HOLD
```
Would mean: "If 10-period momentum > 5%, buy"

---

## Proposed DSL V2 Architecture

### **Option A: Function-Based DSL (Recommended)**

```ebnf
<expression> ::= <comparison> | <arithmetic>
<comparison> ::= <arithmetic> <op> <arithmetic>
<arithmetic> ::= <term> | <term> "+" <term> | <term> "-" <term>
<term> ::= <factor> | <factor> "*" <factor> | <factor> "/" <factor>
<factor> ::= <number> | <symbol> | <function> | "(" <expression> ")"

<function> ::= <func_name> "(" <symbol> "," <start> "," <end> ")"
<func_name> ::= "AVG" | "SUM" | "MAX" | "MIN" | "STD"
<symbol> ::= "ALPHA" | "BETA" | "DELTA" | ...
```

**Example Strategies**:
```
# Mean reversion
IF DELTA(0) < AVG(DELTA, 0, 20) * 0.95 THEN BUY ELSE HOLD

# Breakout
IF DELTA(0) > MAX(DELTA, 1, 50) THEN BUY ELSE HOLD

# Volume confirmation
IF DELTA(0) > DELTA(1) AND EPSILON(0) > AVG(EPSILON, 0, 10) THEN BUY ELSE HOLD

# Volatility filter
IF STD(DELTA, 0, 14) < 0.02 THEN HOLD ELSE <sub-strategy>
```

---

### **Option B: Template-Based Indicators**

Pre-define indicator templates that evolution can compose:

```python
INDICATOR_TEMPLATES = {
    "MOMENTUM": lambda symbol, period: symbol(0) - symbol(period),
    "MA": lambda symbol, period: avg(symbol, 0, period),
    "RSI_LIKE": lambda symbol, period: <simplified RSI calc>,
    "VOLATILITY": lambda symbol, period: std(symbol, 0, period),
    "RATIO": lambda sym1, sym2: sym1(0) / sym2(0),
}
```

**DSL**:
```
IF MOMENTUM(DELTA, 10) > 0 THEN BUY ELSE HOLD
IF RATIO(DELTA, MA(DELTA, 20)) > 1.05 THEN BUY ELSE HOLD
```

---

### **Option C: Intermediate Representation**

Keep the simple DSL but add a **compilation step** that maps symbols to computed features:

```python
# Evolution still uses simple syntax
strategy = "IF ALPHA(10) > BETA(20) THEN BUY ELSE HOLD"

# But interpreter computes rich features
FEATURE_MAP = {
    "ALPHA": lambda t, n: (close[t] - close[t-n]) / close[t-n],  # Momentum
    "BETA": lambda t, n: avg(close, t-n, t),                      # MA
    "GAMMA": lambda t, n: std(close, t-n, t),                     # Volatility
    "DELTA": lambda t, n: close[t],                               # Raw close
    ...
}
```

This way:
- Evolution mutates simple rules
- But symbols represent actual **learnable indicators**
- Can even mutate what ALPHA/BETA compute!

---

## Immediate Recommendations

### **Phase 1: Add Arithmetic (Quick Win)**

Extend grammar to support:
```
IF (DELTA(0) - DELTA(10)) / DELTA(10) > 0.05 THEN BUY ELSE HOLD
```

**Impact**: Enables momentum, ratios, differences
**Complexity**: Medium (need expression parser)
**Mutation**: Mutate constants, operators, term order

---

### **Phase 2: Add Aggregation Functions**

Add: `AVG()`, `SUM()`, `MAX()`, `MIN()`, `STD()`

```
IF DELTA(0) > AVG(DELTA, 0, 20) THEN BUY ELSE HOLD
```

**Impact**: Enables moving averages, volatility, support/resistance
**Complexity**: Medium (need windowed computations)
**Mutation**: Mutate window sizes, function types

---

### **Phase 3: Multi-Step Composition (Advanced)**

Allow chaining:
```
DEF ma20 = AVG(DELTA, 0, 20)
DEF ma50 = AVG(DELTA, 0, 50)
IF ma20 > ma50 THEN BUY ELSE HOLD
```

**Impact**: Enables complex multi-indicator strategies
**Complexity**: High (need variable binding)
**Mutation**: Add/remove definitions, mutate references

---

## Why Current Run Stagnated

Looking at the best strategy:
```
IF EPSILON(20) <= DELTA(50) THEN HOLD ELSE BUY
```

**What it's actually doing**:
- Comparing `volume[t-20]` to `close[t-50]`
- This is **meaningless** mathematically
- But happens to work because: when volume 20hrs ago is "small" relative to price 50hrs ago, it buys
- This is pure overfitting to noise, not a real pattern

**Why it stuck**:
- Any small mutation breaks the lucky coincidence
- Can't discover better patterns because DSL can't express them
- Needs arithmetic to normalize units: `EPSILON(20) / AVG(EPSILON, 0, 100)` (volume vs its average)

---

## Next Steps

###  1. **Design Decision Required**

Choose DSL V2 architecture:
- **Option A**: Full arithmetic expressions (most flexible)
- **Option B**: Template indicators (easier to mutate)
- **Option C**: Hidden feature mapping (simple syntax, rich semantics)

**My Recommendation**: Start with **Option C** (fastest), then migrate to **Option A** (most powerful).

###  2. **Implement Mathematical Operations**

Quick prototype:
```python
# Add to grammar
<arithmetic_expr> ::= <expr> <op> <expr>
<op> ::= "+" | "-" | "*" | "/"

# Update interpreter
def evaluate_expression(expr, market_data, current_index):
    if isinstance(expr, BinaryOp):
        left = evaluate_expression(expr.left, ...)
        right = evaluate_expression(expr.right, ...)
        return apply_op(expr.op, left, right)
    elif isinstance(expr, IndicatorAccess):
        return get_indicator_value(...)
```

### 3. **Rerun Evolution**

Once arithmetic is added, strategies can discover:
- Momentum: `(DELTA(0) - DELTA(N)) / DELTA(N)`
- Normalized volume: `EPSILON(0) / AVG(EPSILON, 0, 100)`
- Candle patterns: `(BETA(0) - GAMMA(0)) / DELTA(0)`

---

## Summary: What's Missing

| Feature | Current | Needed | Impact |
|---------|---------|--------|--------|
| **Raw Access** | ✅ DELTA(10) = close[t-10] | - | Low (implemented) |
| **Arithmetic** | ❌ Can't do math | `+`, `-`, `*`, `/` | **HIGH** |
| **Aggregations** | ❌ No functions | `AVG`, `SUM`, `MAX`, `MIN`, `STD` | **HIGH** |
| **Normalization** | ❌ Comparing different units | Ratios, percent changes | **CRITICAL** |
| **Composition** | ❌ Single rule only | Multi-step indicators | Medium |
| **Conditionals** | ✅ IF/THEN/ELSE | AND, OR, NOT | Medium |

---

**Bottom Line**: The current DSL is a **direct column accessor**, not an **indicator creator**. It needs mathematical operations to discover patterns evolution can actually improve.

The stagnation at $6.17 proves the system works (selection pressure exists), but it can't escape local maxima because the mutation space doesn't contain better solutions expressible in the current grammar.

**Recommendation**: Implement arithmetic expressions (Option A or C) as the minimum viable upgrade.
