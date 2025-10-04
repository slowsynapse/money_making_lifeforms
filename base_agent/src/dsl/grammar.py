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

# A program is a list of rules, but for now we'll start with one.
DslProgram = list[Rule]
