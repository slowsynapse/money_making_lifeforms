import random
from .grammar import DslProgram, Operator, Rule

class DslMutator:
    """
    Applies mutations to a DSL program to evolve it.
    """
    def mutate(self, program: DslProgram) -> DslProgram:
        """
        Applies a random mutation to the program.
        For now, it only mutates the operator.
        """
        # We're only handling single-rule programs for now
        mutated_rule = self._mutate_operator(program[0])
        return [mutated_rule]

    def _mutate_operator(self, rule: Rule) -> Rule:
        """
        Randomly changes the operator in a rule.
        """
        new_rule = rule
        current_operator = new_rule.condition.operator
        
        # Choose a different operator at random
        possible_operators = [op for op in Operator if op != current_operator]
        new_rule.condition.operator = random.choice(possible_operators)
        
        print(f"Mutated operator from {current_operator.value} to {new_rule.condition.operator.value}")
        return new_rule

    def to_string(self, program: DslProgram) -> str:
        """
        Converts a DslProgram back into its string representation.
        """
        # We're only handling single-rule programs for now
        rule = program[0]
        cond = rule.condition
        
        return (
            f"IF {cond.indicator1.name}({cond.param1}) {cond.operator.value} {cond.indicator2.name}({cond.param2}) "
            f"THEN {rule.true_action.name} "
            f"ELSE {rule.false_action.name}"
        )
