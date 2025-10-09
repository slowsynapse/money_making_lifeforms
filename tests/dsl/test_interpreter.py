import pytest
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, 'base_agent')

from src.dsl.grammar import Indicator, Operator, Action
from src.dsl.interpreter import DslInterpreter

class TestDslInterpreter:
    @pytest.fixture
    def interpreter(self):
        return DslInterpreter()

    def test_parse_valid_string(self, interpreter):
        dsl_string = "IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL"
        program = interpreter.parse(dsl_string)
        assert program is not None
        assert len(program) == 1
        rule = program[0]
        assert rule.condition.indicator1 == Indicator.ALPHA
        assert rule.condition.param1 == 10
        assert rule.condition.operator == Operator.GREATER_THAN
        assert rule.condition.indicator2 == Indicator.BETA
        assert rule.condition.param2 == 50
        assert rule.true_action == Action.BUY
        assert rule.false_action == Action.SELL

    @pytest.mark.parametrize("invalid_string", [
        "IF ALPHA(10) > BETA(50) THEN BUY",  # Missing ELSE
        "IF ALPHA(10) > BETA(50) SELL ELSE BUY", # Missing THEN
        "FOO ALPHA(10) > BETA(50) THEN BUY ELSE SELL", # Wrong keyword
        "IF FAKE(10) > ALPHA(50) THEN BUY ELSE SELL", # Invalid indicator
        "IF ALPHA(10) @ BETA(50) THEN BUY ELSE SELL", # Invalid operator
        "IF ALPHA(10) > BETA(50) THEN JUMP ELSE CROUCH", # Invalid action
    ])
    def test_parse_invalid_string(self, interpreter, invalid_string):
        program = interpreter.parse(invalid_string)
        assert program is None

    def test_execute_true_condition(self, interpreter):
        dsl_string = "IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL"
        program = interpreter.parse(dsl_string)
        # The interpreter's dummy logic has indicator1_value > indicator2_value
        action = interpreter.execute(program, {})
        assert action == Action.BUY

    def test_execute_false_condition(self, interpreter):
        dsl_string = "IF ALPHA(10) < BETA(50) THEN BUY ELSE SELL"
        program = interpreter.parse(dsl_string)
        # The interpreter's dummy logic has indicator1_value > indicator2_value
        action = interpreter.execute(program, {})
        assert action == Action.SELL
