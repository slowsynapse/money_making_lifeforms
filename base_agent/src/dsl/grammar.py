from enum import Enum
from dataclasses import dataclass
from typing import Union

class Indicator(Enum):
    """Abstract symbols for the DSL - no technical analysis concepts exposed."""
    ALPHA = "ALPHA"
    BETA = "BETA"
    GAMMA = "GAMMA"
    DELTA = "DELTA"
    EPSILON = "EPSILON"
    ZETA = "ZETA"
    OMEGA = "OMEGA"
    PSI = "PSI"

class Operator(Enum):
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    EQUAL = "=="

class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class ArithmeticOp(Enum):
    """Arithmetic operators for DSL V2."""
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"

class AggregationFunc(Enum):
    """Aggregation functions for DSL V2 Phase 2."""
    AVG = "AVG"      # Average over a window
    SUM = "SUM"      # Sum over a window
    MAX = "MAX"      # Maximum over a window
    MIN = "MIN"      # Minimum over a window
    STD = "STD"      # Standard deviation over a window

class LogicalOp(Enum):
    """Logical operators for DSL V2 Phase 3."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

@dataclass
class IndicatorValue:
    """A single indicator with its parameter (e.g., DELTA(10))."""
    indicator: Indicator
    param: int

@dataclass
class BinaryOp:
    """Binary arithmetic operation (e.g., DELTA(0) / DELTA(20))."""
    left: Union['BinaryOp', 'IndicatorValue', 'FunctionCall']
    op: ArithmeticOp
    right: Union['BinaryOp', 'IndicatorValue', 'FunctionCall']

@dataclass
class FunctionCall:
    """Aggregation function call (e.g., AVG(DELTA, 20))."""
    func: AggregationFunc
    indicator: Indicator
    window: int  # Lookback period for aggregation

# Expression type can be an indicator, binary operation, or function call
Expression = Union[IndicatorValue, BinaryOp, FunctionCall]

@dataclass
class Condition:
    """Comparison condition (e.g., expr1 > expr2)."""
    operator: Operator
    left: Expression = None
    right: Expression = None

    # Legacy support for V1 strategies (will be deprecated)
    # These are only used if left/right are None
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

@dataclass
class CompoundCondition:
    """Compound condition with logical operators (e.g., cond1 AND cond2, NOT cond1)."""
    op: LogicalOp
    left: Union['CompoundCondition', Condition] = None
    right: Union['CompoundCondition', Condition] = None  # None for NOT operator

    def __post_init__(self):
        """Validate compound condition structure."""
        if self.op == LogicalOp.NOT:
            if self.left is None:
                raise ValueError("NOT operator requires a left operand")
            if self.right is not None:
                raise ValueError("NOT operator cannot have a right operand")
        else:  # AND, OR
            if self.left is None or self.right is None:
                raise ValueError(f"{self.op.value} operator requires both left and right operands")

# Condition type can be simple or compound
ConditionType = Union[Condition, CompoundCondition]

@dataclass
class Rule:
    condition: ConditionType  # Can be Condition or CompoundCondition
    true_action: Action
    false_action: Action

class AggregationMode(Enum):
    """How to combine multiple strategy signals."""
    MAJORITY = "MAJORITY"  # Take the most common action (BUY/SELL/HOLD)
    UNANIMOUS = "UNANIMOUS"  # All must agree, else HOLD
    FIRST = "FIRST"  # Only use the first rule (legacy mode)

# A program is a list of rules that can be chained together
DslProgram = list[Rule]
