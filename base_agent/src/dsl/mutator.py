import random
import copy
from .grammar import DslProgram, Operator, Rule, Indicator, Action, Condition

class DslMutator:
    """
    Applies mutations to a DSL program to evolve it.
    Supports multi-rule programs with various mutation strategies.
    """
    def mutate(self, program: DslProgram) -> DslProgram:
        """
        Applies a random mutation to the program.

        Mutation types:
        - Modify existing rule (operator, indicator, parameter, action)
        - Add a new rule (if < 5 rules)
        - Remove a rule (if > 1 rule)
        """
        if not program:
            return program

        # Deep copy to avoid modifying the original
        mutated_program = copy.deepcopy(program)

        # Choose mutation type
        mutation_types = []
        mutation_types.append(("modify_rule", 0.6))  # 60% chance to modify existing rule

        if len(mutated_program) < 5:
            mutation_types.append(("add_rule", 0.25))  # 25% chance to add rule (if room)

        if len(mutated_program) > 1:
            mutation_types.append(("remove_rule", 0.15))  # 15% chance to remove rule (if > 1)

        # Normalize probabilities
        total_prob = sum(prob for _, prob in mutation_types)
        mutation_types = [(name, prob/total_prob) for name, prob in mutation_types]

        # Select mutation
        rand = random.random()
        cumulative = 0
        selected_mutation = mutation_types[0][0]
        for mutation_name, prob in mutation_types:
            cumulative += prob
            if rand < cumulative:
                selected_mutation = mutation_name
                break

        # Apply mutation
        if selected_mutation == "modify_rule":
            rule_idx = random.randint(0, len(mutated_program) - 1)
            mutated_program[rule_idx] = self._mutate_rule(mutated_program[rule_idx])
            print(f"Modified rule {rule_idx + 1}/{len(mutated_program)}")

        elif selected_mutation == "add_rule":
            new_rule = self._create_random_rule()
            mutated_program.append(new_rule)
            print(f"Added new rule (now {len(mutated_program)} rules)")

        elif selected_mutation == "remove_rule":
            removed_idx = random.randint(0, len(mutated_program) - 1)
            mutated_program.pop(removed_idx)
            print(f"Removed rule {removed_idx + 1} (now {len(mutated_program)} rules)")

        return mutated_program

    def _mutate_rule(self, rule: Rule) -> Rule:
        """Mutate a single rule by changing one component."""
        mutation_type = random.choice([
            "operator", "indicator1", "indicator2",
            "param1", "param2", "true_action", "false_action"
        ])

        if mutation_type == "operator":
            possible_operators = [op for op in Operator if op != rule.condition.operator]
            new_op = random.choice(possible_operators)
            print(f"  - Changed operator: {rule.condition.operator.value} → {new_op.value}")
            rule.condition.operator = new_op

        elif mutation_type == "indicator1":
            possible_indicators = [ind for ind in Indicator if ind != rule.condition.indicator1]
            new_ind = random.choice(possible_indicators)
            print(f"  - Changed indicator1: {rule.condition.indicator1.value} → {new_ind.value}")
            rule.condition.indicator1 = new_ind

        elif mutation_type == "indicator2":
            possible_indicators = [ind for ind in Indicator if ind != rule.condition.indicator2]
            new_ind = random.choice(possible_indicators)
            print(f"  - Changed indicator2: {rule.condition.indicator2.value} → {new_ind.value}")
            rule.condition.indicator2 = new_ind

        elif mutation_type == "param1":
            new_param = random.choice([0, 5, 10, 14, 20, 30, 50, 100, 200])
            print(f"  - Changed param1: {rule.condition.param1} → {new_param}")
            rule.condition.param1 = new_param

        elif mutation_type == "param2":
            new_param = random.choice([0, 5, 10, 14, 20, 30, 50, 100, 200])
            print(f"  - Changed param2: {rule.condition.param2} → {new_param}")
            rule.condition.param2 = new_param

        elif mutation_type == "true_action":
            possible_actions = [a for a in Action if a != rule.true_action]
            new_action = random.choice(possible_actions)
            print(f"  - Changed true_action: {rule.true_action.value} → {new_action.value}")
            rule.true_action = new_action

        elif mutation_type == "false_action":
            possible_actions = [a for a in Action if a != rule.false_action]
            new_action = random.choice(possible_actions)
            print(f"  - Changed false_action: {rule.false_action.value} → {new_action.value}")
            rule.false_action = new_action

        return rule

    def _create_random_rule(self) -> Rule:
        """Create a completely random rule."""
        condition = Condition(
            indicator1=random.choice(list(Indicator)),
            param1=random.choice([0, 5, 10, 14, 20, 30, 50, 100, 200]),
            operator=random.choice(list(Operator)),
            indicator2=random.choice(list(Indicator)),
            param2=random.choice([0, 5, 10, 14, 20, 30, 50, 100, 200])
        )
        return Rule(
            condition=condition,
            true_action=random.choice(list(Action)),
            false_action=random.choice(list(Action))
        )

    def to_string(self, program: DslProgram) -> str:
        """
        Converts a DslProgram back into its string representation.
        Multiple rules are separated by newlines.
        """
        if not program:
            return ""

        rule_strings = []
        for rule in program:
            cond = rule.condition

            # Format parameters: show empty parens () for param=0, otherwise show the number
            param1_str = "" if cond.param1 == 0 else str(cond.param1)
            param2_str = "" if cond.param2 == 0 else str(cond.param2)

            rule_str = (
                f"IF {cond.indicator1.name}({param1_str}) {cond.operator.value} {cond.indicator2.name}({param2_str}) "
                f"THEN {rule.true_action.name} "
                f"ELSE {rule.false_action.name}"
            )
            rule_strings.append(rule_str)

        return "\n".join(rule_strings)
