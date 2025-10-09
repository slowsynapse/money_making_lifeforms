# DSL V2 Specification: Extended Grammar

## Current Problem

The current DSL (V1) is too primitive to express meaningful technical indicators. From `evolution_analysis.md`:

**Critical limitation**: Cannot create ratios, percentages, or normalized values.

**Example failure**:
```
Best strategy: IF EPSILON(20) <= DELTA(50) THEN HOLD ELSE BUY
```

This compares `volume[t-20]` to `close[t-50]` - nonsensical because they have different units and magnitudes. It only worked by random overfitting, not genuine pattern discovery.

## What's Missing

### 1. Mathematical Operations

**Need**: `+`, `-`, `*`, `/` for creating indicators

**Why**: Enable ratios, momentum, normalization

**Example**:
```
# Current V1 (broken)
IF EPSILON(20) <= DELTA(50) THEN BUY ELSE HOLD
→ Compares volume to price (nonsense)

# With V2 (meaningful)
IF (EPSILON(0) / EPSILON(20)) > 1.5 THEN BUY ELSE HOLD
→ Compares normalized volume (volume vs 20-period average)
```

### 2. Aggregation Functions

**Need**: `AVG()`, `SUM()`, `MAX()`, `MIN()`, `STD()`

**Why**: Compute moving averages, volatility, support/resistance

**Example**:
```
# Moving average crossover
IF DELTA(0) > AVG(DELTA, 0, 20) THEN BUY ELSE HOLD

# Breakout detection
IF DELTA(0) > MAX(DELTA, 1, 50) THEN BUY ELSE HOLD

# Volatility filter
IF STD(DELTA, 0, 14) < 0.02 THEN HOLD ELSE BUY
```

### 3. Logical Operators

**Need**: `AND`, `OR`, `NOT` for complex conditions

**Why**: Combine multiple signals

**Example**:
```
# Volume confirmation
IF (EPSILON(0) / AVG(EPSILON, 0, 10)) > 2.0 AND DELTA(0) < DELTA(10) THEN BUY ELSE HOLD

# Multiple exit conditions
IF DELTA(0) < DELTA(5) OR (EPSILON(0) / EPSILON(20)) < 0.5 THEN SELL ELSE HOLD
```

### 4. Multi-Timeframe Access

**Need**: Reference different timeframes in same strategy

**Why**: Align entries with larger trends

**Example**:
```
# Trade 1H dips in 4H uptrend
IF DELTA_4H(0) > DELTA_4H(10) AND DELTA_1H(0) < DELTA_1H(5) THEN BUY ELSE HOLD

# Daily trend filter
IF DELTA_1D(0) > AVG(DELTA_1D, 0, 20) THEN <1H_STRATEGY> ELSE HOLD
```

## DSL V2 Grammar (EBNF)

### Complete Syntax

```ebnf
(* Top-level program *)
<program> ::= <rule> | <rule> "\n" <program>

(* Rule structure *)
<rule> ::= "IF" <condition> "THEN" <action> "ELSE" <action>

(* Conditions *)
<condition> ::= <comparison>
              | <condition> "AND" <condition>
              | <condition> "OR" <condition>
              | "NOT" <condition>
              | "(" <condition> ")"

(* Comparisons *)
<comparison> ::= <expression> <operator> <expression>
<operator> ::= ">" | "<" | ">=" | "<=" | "==" | "!="

(* Expressions (arithmetic) *)
<expression> ::= <term>
               | <expression> "+" <term>
               | <expression> "-" <term>

<term> ::= <factor>
         | <term> "*" <factor>
         | <term> "/" <factor>

<factor> ::= <number>
           | <indicator>
           | <function>
           | "(" <expression> ")"

(* Indicators *)
<indicator> ::= <symbol> "(" <number>? ")"
              | <symbol> "_" <timeframe> "(" <number>? ")"

<symbol> ::= "ALPHA" | "BETA" | "GAMMA" | "DELTA" | "EPSILON"
           | "ZETA" | "OMEGA" | "PSI"

<timeframe> ::= "1H" | "4H" | "1D"

(* Functions *)
<function> ::= <func_name> "(" <symbol> "," <number> "," <number> ")"
             | <func_name> "(" <symbol> "_" <timeframe> "," <number> "," <number> ")"

<func_name> ::= "AVG" | "SUM" | "MAX" | "MIN" | "STD"

(* Actions *)
<action> ::= "BUY" | "SELL" | "HOLD"

(* Primitives *)
<number> ::= [0-9]+
```

## Example Strategies (V2)

### 1. Normalized Volume Spike

```
IF (EPSILON(0) / AVG(EPSILON, 0, 20)) > 1.5 THEN BUY ELSE HOLD
```

**Meaning**: Buy when current volume is 50%+ above 20-period average

**What changed**:
- V1 couldn't divide → Compared volume to price (nonsense)
- V2 can divide → Normalized volume ratio (meaningful)

### 2. Mean Reversion with Volatility Filter

```
IF DELTA(0) < AVG(DELTA, 0, 20) * 0.95 AND STD(DELTA, 0, 14) > 0.02 THEN BUY ELSE HOLD
```

**Meaning**: Buy when price is 5% below 20-MA AND volatility is high enough to make the trade worthwhile

### 3. Breakout Strategy

```
IF DELTA(0) > MAX(DELTA, 1, 50) THEN BUY ELSE HOLD
```

**Meaning**: Buy when price breaks above 50-period high

### 4. Volume-Confirmed Momentum

```
IF (DELTA(0) - DELTA(10)) / DELTA(10) > 0.05 AND (EPSILON(0) / AVG(EPSILON, 0, 10)) > 1.2 THEN BUY ELSE HOLD
```

**Meaning**: Buy when 10-period momentum > 5% AND volume is 20%+ above average

### 5. Multi-Timeframe Trend Alignment

```
IF DELTA_4H(0) > AVG(DELTA_4H, 0, 20) AND DELTA_1H(0) < DELTA_1H(5) THEN BUY ELSE HOLD
```

**Meaning**: Buy on 1H dips when 4H is in uptrend

## Backward Compatibility

### V1 Strategies Still Valid

```
# V1 syntax
IF EPSILON(20) <= DELTA(50) THEN HOLD ELSE BUY

# This is valid V2 (subset)
```

All V1 strategies parse correctly in V2. The grammar is **strictly additive**.

### Migration Path

1. **Existing cells keep V1 genomes** - No changes to database
2. **New mutations can use V2 features** - Mutator updated to include arithmetic/functions
3. **Interpreter supports both** - Detects which features are used
4. **LLM recognizes V2 patterns** - Updated analysis prompt with V2 syntax

## Implementation Phases

### Phase 1: Arithmetic Operations ✅ (Highest Priority)

**Add**: `+`, `-`, `*`, `/`

**Grammar changes**:
```python
# Add to interpreter.py
class BinaryOp:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op  # '+', '-', '*', '/'
        self.right = right

def evaluate_expression(expr, market_data, current_index):
    if isinstance(expr, BinaryOp):
        left_val = evaluate_expression(expr.left, market_data, current_index)
        right_val = evaluate_expression(expr.right, market_data, current_index)

        if expr.op == '+': return left_val + right_val
        elif expr.op == '-': return left_val - right_val
        elif expr.op == '*': return left_val * right_val
        elif expr.op == '/': return left_val / right_val if right_val != 0 else float('inf')
```

**Mutation support**:
```python
# Add to mutator.py
def mutate_arithmetic(expr):
    """Mutate arithmetic expressions."""
    mutations = [
        'change_operator',  # + → -, * → /
        'add_parentheses',  # a + b * c → (a + b) * c
        'swap_operands'     # a - b → b - a
    ]
```

**Impact**: Enables momentum, ratios, normalized indicators

### Phase 2: Aggregation Functions (Medium Priority)

**Add**: `AVG()`, `SUM()`, `MAX()`, `MIN()`, `STD()`

**Grammar changes**:
```python
# Add to interpreter.py
class FunctionCall:
    def __init__(self, func_name, symbol, start, end):
        self.func_name = func_name  # 'AVG', 'MAX', etc.
        self.symbol = symbol
        self.start = start  # Start of window
        self.end = end      # End of window

def evaluate_function(func, market_data, current_index):
    column = SYMBOL_TO_COLUMN[func.symbol]

    # Extract window of data
    window_start = max(0, current_index - func.end)
    window_end = current_index - func.start
    window = market_data.iloc[window_start:window_end + 1][column]

    if func.func_name == 'AVG': return window.mean()
    elif func.func_name == 'SUM': return window.sum()
    elif func.func_name == 'MAX': return window.max()
    elif func.func_name == 'MIN': return window.min()
    elif func.func_name == 'STD': return window.std()
```

**Impact**: Enables moving averages, support/resistance, volatility measures

### Phase 3: Logical Operators (Medium Priority)

**Add**: `AND`, `OR`, `NOT`

**Current**: Multi-rule voting achieves similar effect
**Future**: Single complex rule more expressive

**Grammar changes**:
```python
# Add to interpreter.py
class LogicalOp:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op  # 'AND', 'OR'
        self.right = right

class NotOp:
    def __init__(self, condition):
        self.condition = condition
```

**Impact**: Enables multi-signal strategies in single rule

### Phase 4: Multi-Timeframe (Advanced)

**Add**: `SYMBOL_TIMEFRAME(N)` syntax

**Example**: `DELTA_4H(10)` = close price 10 4H-candles ago

**Data requirements**:
- Fetch 1H, 4H, 1D data for each symbol
- 30 days: 720 (1H) + 180 (4H) + 30 (1D) = 930 candles
- Size: ~112 KB per symbol × 3 timeframes = ~336 KB per symbol

**Alignment challenge**:
- 1H and 4H candles don't align perfectly
- Need timestamp matching logic

**Grammar changes**:
```python
# Add to interpreter.py
SYMBOL_TO_COLUMN = {
    ('DELTA', '1H'): ('close', '1h_data'),
    ('DELTA', '4H'): ('close', '4h_data'),
    ('DELTA', '1D'): ('close', '1d_data'),
    # ...
}

def _get_indicator_value(self, indicator, timeframe, param, market_data_dict, current_timestamp):
    # market_data_dict = {'1H': df_1h, '4H': df_4h, '1D': df_1d}
    df = market_data_dict[timeframe]

    # Find closest candle to current_timestamp
    idx = find_closest_index(df, current_timestamp)
    lookback_idx = max(0, idx - param)

    column = SYMBOL_TO_COLUMN[(indicator, timeframe)][0]
    return float(df.iloc[lookback_idx][column])
```

**Impact**: Enables trend alignment, multi-timeframe filters

## Mutation Strategy for V2

### Random Mutations (Same as V1)

- Change operators: `+` → `-`, `*` → `/`
- Change numbers: constants, lookback periods
- Change functions: `AVG` → `MAX`
- Change symbols: `DELTA` → `EPSILON`

### Structure Mutations (New)

- **Wrap in function**: `DELTA(0)` → `AVG(DELTA, 0, 20)`
- **Add arithmetic**: `DELTA(0)` → `DELTA(0) / DELTA(20)`
- **Add logical**: `COND1` → `COND1 AND COND2`
- **Change timeframe**: `DELTA(0)` → `DELTA_4H(0)`

### Constraints

- **Prevent division by zero**: Mutator checks denominators
- **Prevent infinite loops**: Aggregation functions have max window size
- **Prevent nonsensical ops**: Don't compare different timeframes directly without normalization

## Testing V2

### Unit Tests

```python
def test_arithmetic_parsing():
    dsl = "IF (DELTA(0) - DELTA(10)) / DELTA(10) > 0.05 THEN BUY ELSE HOLD"
    program = interpreter.parse(dsl)
    assert isinstance(program[0].condition.left, BinaryOp)

def test_aggregation_function():
    dsl = "IF DELTA(0) > AVG(DELTA, 0, 20) THEN BUY ELSE HOLD"
    program = interpreter.parse(dsl)
    # Execute on test data
    action = interpreter.execute(program, test_market_data, current_index=50)
    assert action in [Action.BUY, Action.HOLD]

def test_multi_timeframe():
    dsl = "IF DELTA_4H(0) > DELTA_4H(10) AND DELTA_1H(0) < DELTA_1H(5) THEN BUY ELSE HOLD"
    program = interpreter.parse(dsl)
    # Requires multi-timeframe data
```

### Evolution Test

Run 50 generations with V2 grammar, check if:
1. Fitness improves beyond V1 best ($6.17)
2. LLM can name the discovered patterns
3. Patterns use arithmetic/aggregations meaningfully

## Multi-Timeframe Data Structure

### Storage Format

```python
market_data = {
    '1H': pd.DataFrame({
        'timestamp': [...],  # 720 rows (30 days)
        'open': [...],
        'high': [...],
        'low': [...],
        'close': [...],
        'volume': [...],
    }),
    '4H': pd.DataFrame({
        'timestamp': [...],  # 180 rows (30 days)
        # ... same columns
    }),
    '1D': pd.DataFrame({
        'timestamp': [...],  # 30 rows (30 days)
        # ... same columns
    })
}
```

### Interpreter Signature Change

```python
# Old (V1)
def execute(self, program: DslProgram, market_data: pd.DataFrame, current_index: int) -> Action

# New (V2)
def execute(self, program: DslProgram, market_data: dict[str, pd.DataFrame], current_timestamp: datetime) -> Action
```

### Timestamp Alignment

```python
def find_closest_candle(df: pd.DataFrame, target_timestamp: datetime) -> int:
    """Find index of candle closest to target timestamp."""
    timestamps = pd.to_datetime(df['timestamp'])
    deltas = abs(timestamps - target_timestamp)
    return deltas.idxmin()
```

## Summary

### V1 → V2 Improvements

| Feature | V1 | V2 | Impact |
|---------|----|----|--------|
| **Arithmetic** | ❌ | ✅ `+`, `-`, `*`, `/` | Enable ratios, momentum, normalization |
| **Aggregations** | ❌ | ✅ `AVG`, `MAX`, `STD`, etc. | Enable moving averages, volatility |
| **Logical** | Multi-rule voting | ✅ `AND`, `OR`, `NOT` | Complex conditions in single rule |
| **Multi-TF** | ❌ | ✅ `DELTA_4H(10)` | Cross-timeframe strategies |

### Migration

- V1 strategies remain valid (backward compatible)
- V2 mutations unlocked incrementally (phase by phase)
- Database unchanged (genome is still text)
- LLM learns to interpret V2 patterns

### Priority

1. **Phase 1** (Arithmetic) - CRITICAL - Fixes "comparing volume to price" problem
2. **Phase 2** (Aggregations) - HIGH - Enables classic indicator patterns
3. **Phase 3** (Logical) - MEDIUM - Nice to have, multi-rule voting works
4. **Phase 4** (Multi-TF) - ADVANCED - Future enhancement

**Next step**: Implement Phase 1 (arithmetic) and rerun evolution to see if it escapes the $6.17 local maximum.
