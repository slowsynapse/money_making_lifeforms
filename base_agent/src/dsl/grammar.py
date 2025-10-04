from enum import Enum
from dataclasses import dataclass
from typing import Union

class Indicator(Enum):
    SMA = "SMA"
    EMA = "EMA"
    RSI = "RSI"
    PRICE = "PRICE"
    VOLUME = "VOLUME"

class Operator(Enum):
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="

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
