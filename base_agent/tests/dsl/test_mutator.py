import pytest
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dsl.grammar import Operator
from src.dsl.interpreter import DslInterpreter
from src.dsl.mutator import DslMutator

class TestDslMutator:
    @pytest.fixture
    def mutator(self):
        return DslMutator()
        
    @pytest.fixture
    def simple_program(self):
        interpreter = DslInterpreter()
        return interpreter.parse("IF SMA(10) > SMA(50) THEN BUY ELSE SELL")

    def test_mutate_operator(self, mutator, simple_program):
        original_operator = simple_program[0].condition.operator
        
        # Run mutation multiple times to ensure it changes
        changed = False
        for _ in range(10): # High probability of changing
            mutated_program = mutator.mutate(simple_program)
            new_operator = mutated_program[0].condition.operator
            if new_operator != original_operator:
                changed = True
                break
        assert changed, "Operator was not mutated"
        assert new_operator in [op for op in Operator if op != original_operator]

    def test_to_string_conversion(self, mutator, simple_program):
        dsl_string = mutator.to_string(simple_program)
        expected_string = "IF SMA(10) > SMA(50) THEN BUY ELSE SELL"
        # The regex in the interpreter is a bit loose on whitespace, so let's normalize
        assert " ".join(dsl_string.split()) == " ".join(expected_string.split())

    def test_string_to_program_to_string(self, mutator):
        interpreter = DslInterpreter()
        original_string = "IF EMA(20) <= PRICE(0) THEN SELL ELSE HOLD"
        program = interpreter.parse(original_string)
        new_string = mutator.to_string(program)
        assert " ".join(original_string.split()) == " ".join(new_string.split())
