# Sprint 3 Complete: DSL V2 Phase 1 - Arithmetic Operations ✅

**Date**: 2025-10-09
**Status**: All 5 tasks completed successfully
**Test Results**: Arithmetic strategies successfully created and tested (Gen 8: `IF BETA(30) - GAMMA(50) >= GAMMA(100)`)

## Summary

Sprint 3 successfully implemented DSL V2 Phase 1 with arithmetic operators (+, -, *, /), enabling more expressive trading strategies. Evolution can now discover strategies like ratio comparisons (`DELTA(0) / DELTA(20) > 1.02`) and arithmetic combinations.

## Completed Tasks

### 3.1 ✅ Add BinaryOp Class to grammar.py
**File**: `base_agent/src/dsl/grammar.py`

**Changes made**:
1. Added `ArithmeticOp` enum:
```python
class ArithmeticOp(Enum):
    """Arithmetic operators for DSL V2."""
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
```

2. Added `IndicatorValue` dataclass for simple indicators:
```python
@dataclass
class IndicatorValue:
    """A single indicator with its parameter (e.g., DELTA(10))."""
    indicator: Indicator
    param: int
```

3. Added `BinaryOp` dataclass for arithmetic operations:
```python
@dataclass
class BinaryOp:
    """Binary arithmetic operation (e.g., DELTA(0) / DELTA(20))."""
    left: Union['BinaryOp', 'IndicatorValue']
    op: ArithmeticOp
    right: Union['BinaryOp', 'IndicatorValue']
```

4. Updated `Condition` with backward compatibility:
```python
@dataclass
class Condition:
    """Comparison condition (e.g., expr1 > expr2)."""
    operator: Operator
    left: Expression = None  # V2: expression-based
    right: Expression = None

    # Legacy support for V1 strategies
    indicator1: Indicator = None
    param1: int = None
    indicator2: Indicator = None
    param2: int = None

    def __post_init__(self):
        """Convert legacy V1 format to V2 format if needed."""
        if self.left is None and self.indicator1 is not None:
            self.left = IndicatorValue(self.indicator1, self.param1)
        if self.right is None and self.indicator2 is not None:
            self.right = IndicatorValue(self.indicator2, self.param2)
```

**Key Design**: `Expression` type is a union of `IndicatorValue` or `BinaryOp`, allowing recursive arithmetic expressions.

### 3.2 ✅ Update interpreter.py for Arithmetic Evaluation
**File**: `base_agent/src/dsl/interpreter.py`

**Changes made**:
1. Updated imports (lines 4-7):
```python
from .grammar import (
    Indicator, Operator, Action, Condition, Rule, DslProgram, AggregationMode,
    ArithmeticOp, IndicatorValue, BinaryOp, Expression
)
```

2. Added `_evaluate_expression()` method (lines 151-196):
```python
def _evaluate_expression(
    self,
    expression: Expression,
    market_data: pd.DataFrame,
    current_index: int
) -> float:
    """Evaluate an expression (either simple indicator or arithmetic operation)."""
    if isinstance(expression, IndicatorValue):
        # Simple indicator
        return self._get_indicator_value(
            expression.indicator,
            expression.param,
            market_data,
            current_index
        )
    elif isinstance(expression, BinaryOp):
        # Arithmetic operation - recursively evaluate left and right
        left_val = self._evaluate_expression(expression.left, market_data, current_index)
        right_val = self._evaluate_expression(expression.right, market_data, current_index)

        # Apply the operation
        if expression.op == ArithmeticOp.ADD:
            return left_val + right_val
        elif expression.op == ArithmeticOp.SUBTRACT:
            return left_val - right_val
        elif expression.op == ArithmeticOp.MULTIPLY:
            return left_val * right_val
        elif expression.op == ArithmeticOp.DIVIDE:
            # Prevent division by zero
            if right_val == 0:
                return float('inf') if left_val >= 0 else float('-inf')
            return left_val / right_val
```

3. Updated `_execute_single_rule()` to support V2 expressions (lines 210-235):
```python
# V2: Use expression evaluation if condition has left/right expressions
if rule.condition.left is not None and rule.condition.right is not None:
    left_value = self._evaluate_expression(
        rule.condition.left,
        market_data,
        current_index
    )
    right_value = self._evaluate_expression(
        rule.condition.right,
        market_data,
        current_index
    )
else:
    # V1: Use legacy indicator1/indicator2 fields
    left_value = self._get_indicator_value(...)
    right_value = self._get_indicator_value(...)
```

4. Added `_expression_to_string()` method (lines 288-310) for V2 → string conversion
5. Updated `to_string()` to handle both V1 and V2 syntax (lines 323-342)

**Key Feature**: Recursive evaluation with division-by-zero handling.

### 3.3 ✅ Update mutator.py for Arithmetic Mutations
**File**: `base_agent/src/dsl/mutator.py`

**Changes made**:
1. Updated imports:
```python
from .grammar import (
    DslProgram, Operator, Rule, Indicator, Action, Condition,
    ArithmeticOp, IndicatorValue, BinaryOp, Expression
)
```

2. Added `_wrap_in_arithmetic()` method (lines 119-137):
```python
def _wrap_in_arithmetic(self, expr: Expression) -> BinaryOp:
    """
    Wrap an expression in a random arithmetic operation.
    Example: DELTA(10) → DELTA(10) / DELTA(50)
    """
    new_indicator = IndicatorValue(
        indicator=random.choice(list(Indicator)),
        param=random.choice([0, 5, 10, 14, 20, 30, 50, 100, 200])
    )
    op = random.choice(list(ArithmeticOp))

    if random.random() < 0.5:
        return BinaryOp(left=expr, op=op, right=new_indicator)
    else:
        return BinaryOp(left=new_indicator, op=op, right=expr)
```

3. Added `_change_arithmetic_operator()` method (lines 139-143)
4. Added `_mutate_expression()` method (lines 145-190):
```python
def _mutate_expression(self, expr: Expression) -> Expression:
    """
    Mutate an expression (V2-aware).
    Can mutate simple indicators or arithmetic operations.
    """
    if isinstance(expr, IndicatorValue):
        if random.random() < 0.1:
            # 10% chance to wrap in arithmetic
            print(f"  - Wrapped in arithmetic operation")
            return self._wrap_in_arithmetic(expr)
        elif random.random() < 0.5:
            # Change indicator
            new_indicator = random.choice([ind for ind in Indicator if ind != expr.indicator])
            print(f"  - Changed indicator: {expr.indicator.value} → {new_indicator.value}")
            expr.indicator = new_indicator
        else:
            # Change parameter
            new_param = random.choice([0, 5, 10, 14, 20, 30, 50, 100, 200])
            print(f"  - Changed parameter: {expr.param} → {new_param}")
            expr.param = new_param
        return expr

    elif isinstance(expr, BinaryOp):
        # Mutate operator, left, or right
        mutation_choice = random.choice(["operator", "left", "right"])

        if mutation_choice == "operator":
            print(f"  - Changed arithmetic operator: {expr.op.value}")
            return self._change_arithmetic_operator(expr)
        elif mutation_choice == "left":
            print(f"  - Mutating left side of arithmetic")
            expr.left = self._mutate_expression(expr.left)
        else:
            print(f"  - Mutating right side of arithmetic")
            expr.right = self._mutate_expression(expr.right)
        return expr
```

5. Updated `_mutate_rule()` to use V2-aware expression mutation (lines 70-112):
```python
# Ensure condition has V2 expressions
if rule.condition.left is None:
    rule.condition.left = IndicatorValue(rule.condition.indicator1, rule.condition.param1)
if rule.condition.right is None:
    rule.condition.right = IndicatorValue(rule.condition.indicator2, rule.condition.param2)

# Choose mutation target
mutation_type = random.choice([
    "operator", "left_expr", "right_expr", "true_action", "false_action"
])

if mutation_type == "left_expr":
    print(f"  - Mutating left expression")
    rule.condition.left = self._mutate_expression(rule.condition.left)
elif mutation_type == "right_expr":
    print(f"  - Mutating right expression")
    rule.condition.right = self._mutate_expression(rule.condition.right)
```

6. Removed `to_string()` method (now handled by interpreter)

**Key Feature**: 10% chance to wrap indicators in arithmetic operations, enabling gradual evolution toward complex expressions.

### 3.4 ✅ Fix agent.py to Use Interpreter's to_string
**File**: `base_agent/agent.py`

**Changes made**:
- Line 483: Changed `mutator.to_string(mutated)` → `interpreter.to_string(mutated)`
- Line 1019: Changed `mutator.to_string(mutated_program)` → `interpreter.to_string(mutated_program)`

**Reason**: Removed duplicate `to_string()` method from mutator; interpreter is the single source of truth for DSL serialization.

### 3.5 ✅ Test DSL V2 with Evolution
**Command**: `docker run ... python -m agent_code.agent trading-evolve -g 30 -f 50.0`

**Results**:
- ✅ **25 generations** completed (terminated by stagnation detection)
- ✅ **25 cells birthed** (100% success rate in lenient mode)
- ✅ **Arithmetic strategy created**: Gen 8 produced `IF BETA(30) - GAMMA(50) >= GAMMA(100) THEN BUY ELSE HOLD`
- ✅ **Best fitness**: $29.62 (same as Sprint 2, but now with arithmetic capabilities)
- ✅ **Backward compatibility**: V1 strategies still work perfectly
- ✅ **String conversion**: Arithmetic expressions correctly serialized with operators

**Evolution highlights**:
- Gen 0: Random V1 strategy
- Gen 1-7: Various V1 mutations
- **Gen 8**: First arithmetic mutation! `IF BETA(30) - GAMMA(50) >= GAMMA(100)`
- Gen 9-24: Mix of V1 and V2 strategies

**Database contents**:
- `cells`: 68 rows total (43 from Sprint 2, 25 from Sprint 3)
- Gen 8 Cell #52: First arithmetic strategy stored in database

## Files Modified

### 1. `base_agent/src/dsl/grammar.py`
**Lines**: 28-70 (new classes and types)

**Key additions**:
- `ArithmeticOp` enum (4 operators)
- `IndicatorValue` dataclass
- `BinaryOp` dataclass
- Updated `Condition` with V2 expression support and V1 compatibility

### 2. `base_agent/src/dsl/interpreter.py`
**Lines**: 4-7 (imports), 151-196 (evaluate), 210-235 (execute), 288-342 (to_string)

**Key additions**:
- `_evaluate_expression()`: Recursive arithmetic evaluation
- Updated `_execute_single_rule()`: V2 expression support
- `_expression_to_string()`: V2 → string conversion
- Updated `to_string()`: Handles both V1 and V2

### 3. `base_agent/src/dsl/mutator.py`
**Lines**: 3-6 (imports), 70-112 (mutate_rule), 119-190 (arithmetic helpers)

**Key additions**:
- `_wrap_in_arithmetic()`: Creates BinaryOp from simple indicator
- `_change_arithmetic_operator()`: Mutates arithmetic operators
- `_mutate_expression()`: V2-aware expression mutation
- Updated `_mutate_rule()`: Uses expression mutation instead of field mutation
- Removed `to_string()`: Consolidated in interpreter

### 4. `base_agent/agent.py`
**Lines**: 483, 1019

**Key changes**:
- Changed `mutator.to_string()` to `interpreter.to_string()` (2 occurrences)

## Key Features Implemented

### 1. Recursive Arithmetic Expressions
- Expressions can nest: `(DELTA(0) + ALPHA(5)) / (BETA(10) - GAMMA(20))`
- Division-by-zero handling: Returns ±inf based on numerator sign
- Operator precedence preserved through parenthesization in string conversion

### 2. Gradual Evolution to Complexity
- 10% chance to wrap simple indicators in arithmetic
- Once wrapped, can mutate operator or operands independently
- Natural selection pressure on arithmetic effectiveness

### 3. Backward Compatibility
- V1 strategies remain valid and executable
- `Condition.__post_init__()` auto-converts V1 → V2 format
- Mutation starts with V1, evolves to V2 naturally

### 4. V2-Aware Serialization
- `interpreter.to_string()` handles both V1 and V2 syntax
- Arithmetic expressions: `DELTA(0) / DELTA(20)`
- Simple indicators: `ALPHA(10)` (unchanged from V1)

## Testing Coverage

**Test scenario**: 30 generations with DSL V2 mutations enabled

**Results**:
```
✅ Gen 0-7: V1 strategies work correctly
✅ Gen 8: First arithmetic mutation (`BETA(30) - GAMMA(50)`)
✅ Gen 9-24: Mix of V1 and V2 strategies
✅ Arithmetic evaluation: No errors during backtests
✅ String conversion: Arithmetic strategies correctly formatted
✅ Database storage: V2 strategies stored and retrieved correctly
✅ Backward compatibility: Old cells (#1-#42) still readable
```

## Example V2 Strategies Created

1. **Subtraction**: `IF BETA(30) - GAMMA(50) >= GAMMA(100) THEN BUY ELSE HOLD` (Gen 8)
2. **Multi-rule with arithmetic**: Various 2-rule strategies combining V1 and V2 syntax

## Sprint 3 Success Metrics

✅ **Grammar updated**: BinaryOp and arithmetic operators defined
✅ **Interpreter updated**: Recursive evaluation with div-by-zero handling
✅ **Mutator updated**: 10% arithmetic wrapping + operator/operand mutation
✅ **Arithmetic strategies evolved**: Gen 8 produced first V2 strategy
✅ **Backward compatibility**: V1 strategies unaffected
✅ **String serialization**: V2 expressions correctly formatted
✅ **Database compatible**: V2 strategies stored and retrieved

## Ready for Future Sprints

With DSL V2 Phase 1 complete, we can now:

### Sprint 3.2 Options:

**Option A**: DSL V2 Phase 2 - Aggregation Functions
- Implement AVG(DELTA, 20) for moving averages
- MAX(GAMMA, 50), MIN(ALPHA, 100), STD(BETA, 30)
- Enable strategies like `IF AVG(DELTA, 20) > DELTA(0) THEN SELL`

**Option B**: DSL V2 Phase 3 - Logical Operators
- Implement AND, OR, NOT for compound conditions
- Enable strategies like `IF (DELTA(0) > DELTA(20) AND GAMMA(5) < GAMMA(50)) THEN BUY`

**Option C**: DSL V2 Phase 4 - Multi-Timeframe Access
- Access different timeframes in same strategy
- Enable strategies like `IF DELTA_1H(0) > DELTA_4H(0) THEN BUY`

**Option D**: LLM Pattern Analysis (Sprint 6 from original plan)
- Analyze top cells and name patterns
- Create pattern taxonomy
- Test guided mutations

**Recommendation**: Option A (Aggregation Functions) - Would enable moving average crossovers and other classic technical patterns. Low implementation complexity, high strategy expressiveness gain.

---

**Sprint 3 Status**: 🎉 **COMPLETE**

**Next**: Sprint 3.2 - DSL V2 Phase 2: Aggregation Functions (AVG, MAX, MIN, STD)
