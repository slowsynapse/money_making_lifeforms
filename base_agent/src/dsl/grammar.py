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

@dataclass
class Condition:
    indicator1: Indicator
    param1: int
    operator: Operator
    indicator2: Indicator
    param2: int

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
