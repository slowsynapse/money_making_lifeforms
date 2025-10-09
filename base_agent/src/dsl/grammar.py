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

@dataclass
class IndicatorValue:
    """A single indicator with its parameter (e.g., DELTA(10))."""
    indicator: Indicator
    param: int

@dataclass
class BinaryOp:
    """Binary arithmetic operation (e.g., DELTA(0) / DELTA(20))."""
    left: Union['BinaryOp', 'IndicatorValue']
    op: ArithmeticOp
    right: Union['BinaryOp', 'IndicatorValue']

# Expression type can be either a simple indicator or a binary operation
Expression = Union[IndicatorValue, BinaryOp]

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
class Rule:
    condition: Condition
    true_action: Action
    false_action: Action

class AggregationMode(Enum):
    """How to combine multiple strategy signals."""
    MAJORITY = "MAJORITY"  # Take the most common action (BUY/SELL/HOLD)
    UNANIMOUS = "UNANIMOUS"  # All must agree, else HOLD
    FIRST = "FIRST"  # Only use the first rule (legacy mode)

# A program is a list of rules that can be chained together
DslProgram = list[Rule]
