import random
import copy
from .grammar import (
    DslProgram, Operator, Rule, Indicator, Action, Condition,
    ArithmeticOp, IndicatorValue, BinaryOp, FunctionCall, AggregationFunc, Expression,
    CompoundCondition, LogicalOp, ConditionType, Timeframe
)

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
        """
        Mutate a single rule by changing one component.
        Supports both V1 (legacy), V2 (expression-based), and V3 (compound conditions) rules.
        """
        # Ensure condition has V2 expressions (Condition.__post_init__ should handle this)
        if isinstance(rule.condition, Condition):
            if rule.condition.left is None:
                rule.condition.left = IndicatorValue(rule.condition.indicator1, rule.condition.param1)
            if rule.condition.right is None:
                rule.condition.right = IndicatorValue(rule.condition.indicator2, rule.condition.param2)

        # Choose mutation target
        mutation_options = ["condition", "true_action", "false_action"]
        mutation_type = random.choice(mutation_options)

        if mutation_type == "condition":
            rule.condition = self._mutate_condition(rule.condition)
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

    def _wrap_in_arithmetic(self, expr: Expression) -> BinaryOp:
        """
        Wrap an expression in a random arithmetic operation.
        Example: DELTA(10) → DELTA(10) / DELTA(50)
        """
        # Create a new random indicator value
        new_indicator = IndicatorValue(
            indicator=random.choice(list(Indicator)),
            param=random.choice([0, 5, 10, 14, 20, 30, 50, 100, 200])
        )

        # Choose arithmetic operator
        op = random.choice(list(ArithmeticOp))

        # Randomly decide if expr is left or right operand
        if random.random() < 0.5:
            return BinaryOp(left=expr, op=op, right=new_indicator)
        else:
            return BinaryOp(left=new_indicator, op=op, right=expr)

    def _change_arithmetic_operator(self, binary_op: BinaryOp) -> BinaryOp:
        """Change the operator in a BinaryOp."""
        possible_ops = [op for op in ArithmeticOp if op != binary_op.op]
        binary_op.op = random.choice(possible_ops)
        return binary_op

    def _mutate_expression(self, expr: Expression) -> Expression:
        """
        Mutate an expression (V2-aware with aggregation functions and multi-timeframe).
        Can mutate simple indicators, arithmetic operations, or aggregation functions.
        """
        if isinstance(expr, IndicatorValue):
            # For simple indicators, randomly either:
            # 1. Change the indicator (40%)
            # 2. Change the parameter (40%)
            # 3. Change the timeframe (5% - DSL V2 Phase 4)
            # 4. Wrap in arithmetic operation (10% chance)
            # 5. Convert to aggregation function (5% chance - DSL V2 Phase 2)

            rand = random.random()
            if rand < 0.05:
                # Convert to aggregation function
                func = random.choice(list(AggregationFunc))
                window = random.choice([5, 10, 14, 20, 30, 50])
                print(f"  - Converted to aggregation: {func.value}({expr.indicator.value}, {window})")
                return FunctionCall(func=func, indicator=expr.indicator, window=window, timeframe=expr.timeframe)
            elif rand < 0.10:
                # Change timeframe (DSL V2 Phase 4)
                timeframes = [tf for tf in Timeframe if tf != expr.timeframe]
                new_timeframe = random.choice(timeframes)
                old_tf_str = expr.timeframe.value if expr.timeframe.value else "DEFAULT"
                new_tf_str = new_timeframe.value if new_timeframe.value else "DEFAULT"
                print(f"  - Changed timeframe: {old_tf_str} → {new_tf_str}")
                expr.timeframe = new_timeframe
                return expr
            elif rand < 0.20:
                # Wrap in arithmetic
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

        elif isinstance(expr, FunctionCall):
            # For aggregation functions, randomly:
            # 1. Change the function type (AVG → MAX, etc.)
            # 2. Change the indicator
            # 3. Change the window size
            # 4. Change the timeframe (5% - DSL V2 Phase 4)
            # 5. Convert back to simple indicator (5% chance)

            if random.random() < 0.05:
                # Convert back to simple indicator
                print(f"  - Simplified aggregation to indicator")
                return IndicatorValue(indicator=expr.indicator, param=0, timeframe=expr.timeframe)

            mutation_choice = random.choice(["function", "indicator", "window", "timeframe"])

            if mutation_choice == "function":
                new_func = random.choice([f for f in AggregationFunc if f != expr.func])
                print(f"  - Changed aggregation function: {expr.func.value} → {new_func.value}")
                expr.func = new_func
            elif mutation_choice == "indicator":
                new_indicator = random.choice([ind for ind in Indicator if ind != expr.indicator])
                print(f"  - Changed aggregation indicator: {expr.indicator.value} → {new_indicator.value}")
                expr.indicator = new_indicator
            elif mutation_choice == "window":
                new_window = random.choice([5, 10, 14, 20, 30, 50, 100])
                print(f"  - Changed aggregation window: {expr.window} → {new_window}")
                expr.window = new_window
            else:  # timeframe (DSL V2 Phase 4)
                timeframes = [tf for tf in Timeframe if tf != expr.timeframe]
                new_timeframe = random.choice(timeframes)
                old_tf_str = expr.timeframe.value if expr.timeframe.value else "DEFAULT"
                new_tf_str = new_timeframe.value if new_timeframe.value else "DEFAULT"
                print(f"  - Changed aggregation timeframe: {old_tf_str} → {new_tf_str}")
                expr.timeframe = new_timeframe
            return expr

        elif isinstance(expr, BinaryOp):
            # For arithmetic operations, randomly:
            # 1. Change the operator
            # 2. Mutate left expression
            # 3. Mutate right expression
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

        return expr

    def _mutate_condition(self, condition: ConditionType) -> ConditionType:
        """
        Mutate a condition (simple or compound).
        Can convert simple → compound, compound → simple, or mutate within type.
        """
        if isinstance(condition, Condition):
            # Simple condition - randomly either:
            # 1. Mutate the operator (40%)
            # 2. Mutate left expression (25%)
            # 3. Mutate right expression (25%)
            # 4. Convert to compound condition (10% - DSL V2 Phase 3)

            rand = random.random()
            if rand < 0.10:
                # Convert to compound condition (combine with another random condition)
                print(f"  - Converting simple to compound condition")
                logical_op = random.choice([LogicalOp.AND, LogicalOp.OR])

                # Create a new random simple condition
                new_condition = Condition(
                    left=IndicatorValue(
                        indicator=random.choice(list(Indicator)),
                        param=random.choice([0, 5, 10, 14, 20, 30, 50])
                    ),
                    right=IndicatorValue(
                        indicator=random.choice(list(Indicator)),
                        param=random.choice([0, 5, 10, 14, 20, 30, 50])
                    ),
                    operator=random.choice(list(Operator))
                )

                return CompoundCondition(op=logical_op, left=condition, right=new_condition)

            elif rand < 0.50:
                # Mutate operator
                possible_operators = [op for op in Operator if op != condition.operator]
                new_op = random.choice(possible_operators)
                print(f"  - Changed operator: {condition.operator.value} → {new_op.value}")
                condition.operator = new_op
                return condition

            elif rand < 0.75:
                # Mutate left expression
                print(f"  - Mutating left expression")
                condition.left = self._mutate_expression(condition.left)
                return condition

            else:
                # Mutate right expression
                print(f"  - Mutating right expression")
                condition.right = self._mutate_expression(condition.right)
                return condition

        elif isinstance(condition, CompoundCondition):
            # Compound condition - randomly either:
            # 1. Change logical operator (AND ↔ OR) (20%)
            # 2. Mutate left condition (30%)
            # 3. Mutate right condition (30%)
            # 4. Wrap in NOT (5%)
            # 5. Simplify to left condition only (10%)
            # 6. Simplify to right condition only (5%)

            rand = random.random()
            if rand < 0.10 and condition.op != LogicalOp.NOT:
                # Simplify to left condition
                print(f"  - Simplified compound condition to left side")
                return condition.left

            elif rand < 0.15 and condition.op != LogicalOp.NOT:
                # Simplify to right condition
                print(f"  - Simplified compound condition to right side")
                return condition.right

            elif rand < 0.20:
                # Wrap in NOT
                print(f"  - Wrapped condition in NOT")
                return CompoundCondition(op=LogicalOp.NOT, left=condition)

            elif rand < 0.40 and condition.op != LogicalOp.NOT:
                # Change logical operator (AND ↔ OR)
                new_op = LogicalOp.OR if condition.op == LogicalOp.AND else LogicalOp.AND
                print(f"  - Changed logical operator: {condition.op.value} → {new_op.value}")
                condition.op = new_op
                return condition

            elif rand < 0.70:
                # Mutate left condition
                print(f"  - Mutating left side of compound condition")
                condition.left = self._mutate_condition(condition.left)
                return condition

            else:
                # Mutate right condition (if exists)
                if condition.right is not None:
                    print(f"  - Mutating right side of compound condition")
                    condition.right = self._mutate_condition(condition.right)
                else:
                    # For NOT, unwrap it
                    print(f"  - Unwrapped NOT condition")
                    return condition.left
                return condition

        return condition

    def _create_random_rule(self) -> Rule:
        """
        Create a completely random rule.
        Creates V1 rules by default, but mutations can evolve them into V2.
        """
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
