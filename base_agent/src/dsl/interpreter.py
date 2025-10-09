import re
import pandas as pd
import numpy as np
from collections import Counter
from .grammar import (
    Indicator, Operator, Action, Condition, Rule, DslProgram, AggregationMode,
    ArithmeticOp, IndicatorValue, BinaryOp, FunctionCall, AggregationFunc, Expression
)

class DslInterpreter:
    """
    Parses and executes a DSL program with support for multiple chained strategies.
    """
    def __init__(self, aggregation_mode: AggregationMode = AggregationMode.MAJORITY):
        # Example DSL strings:
        # Single rule: IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL
        # Multiple rules (newline or semicolon separated):
        #   IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL
        #   IF GAMMA(20) <= OMEGA() THEN SELL ELSE HOLD
        #   IF DELTA(5) == PSI(100) THEN HOLD ELSE BUY
        # Supports both parameterized (N) and parameterless () symbols
        self.rule_pattern = re.compile(
            r"IF\s+"
            r"(\w+)\((\d*)\)\s*"          # Indicator 1 and optional param 1
            r"([><]=?|==)\s*"             # Operator (including ==)
            r"(\w+)\((\d*)\)\s*"          # Indicator 2 and optional param 2
            r"THEN\s+(\w+)\s*"            # True Action
            r"ELSE\s+(\w+)"               # False Action
        )
        self.aggregation_mode = aggregation_mode

    def parse(self, dsl_string: str) -> DslProgram | None:
        """
        Parses a DSL string into a structured program.
        Supports multiple rules separated by newlines or semicolons.
        Returns None if parsing fails.
        """
        # Split on newlines and semicolons to get individual rules
        rule_strings = [s.strip() for s in re.split(r'[;\n]+', dsl_string.strip()) if s.strip()]

        if not rule_strings:
            return None

        rules = []
        for rule_str in rule_strings:
            match = self.rule_pattern.match(rule_str)
            if not match:
                return None  # If any rule fails to parse, fail the whole program

            try:
                ind1_str, param1_str, op_str, ind2_str, param2_str, true_action_str, false_action_str = match.groups()

                # Handle empty params (for symbols like OMEGA())
                param1 = int(param1_str) if param1_str else 0
                param2 = int(param2_str) if param2_str else 0

                condition = Condition(
                    indicator1=Indicator[ind1_str],
                    param1=param1,
                    operator=self._string_to_operator(op_str),
                    indicator2=Indicator[ind2_str],
                    param2=param2
                )

                rule = Rule(
                    condition=condition,
                    true_action=Action[true_action_str],
                    false_action=Action[false_action_str]
                )
                rules.append(rule)
            except (KeyError, ValueError):
                # Invalid indicator, operator, or action name
                return None

        return rules if rules else None

    def _string_to_operator(self, op_str: str) -> Operator:
        for op in Operator:
            if op.value == op_str:
                return op
        raise ValueError(f"Invalid operator: {op_str}")

    def execute(self, program: DslProgram, market_data: pd.DataFrame, current_index: int = None) -> Action:
        """
        Executes a parsed DSL program against the current market data.
        If program contains multiple rules, aggregates their signals according to aggregation_mode.

        Args:
            program: Parsed DSL program (list of rules)
            market_data: Full DataFrame with OHLCV columns, or dict for backward compatibility
            current_index: Row index for current decision point (required if market_data is DataFrame)

        Returns:
            Action (BUY, SELL, or HOLD)
        """
        if not program:
            return Action.HOLD

        # Execute each rule and collect their actions
        actions = []
        for rule in program:
            action = self._execute_single_rule(rule, market_data, current_index)
            actions.append(action)

        # Aggregate the actions
        return self._aggregate_actions(actions)

    # Mapping of abstract symbols to OHLCV columns
    SYMBOL_TO_COLUMN = {
        Indicator.ALPHA: 'open',
        Indicator.BETA: 'high',
        Indicator.GAMMA: 'low',
        Indicator.DELTA: 'close',
        Indicator.EPSILON: 'volume',
        Indicator.ZETA: 'trades',
        Indicator.OMEGA: 'funding_rate',
        Indicator.PSI: 'open_interest',
    }

    def _get_indicator_value(
        self,
        indicator: Indicator,
        param: int,
        market_data: pd.DataFrame,
        current_index: int
    ) -> float:
        """
        Get indicator value with lookback.

        Args:
            indicator: The abstract symbol (ALPHA, BETA, etc.)
            param: Lookback period (0 = current, N = N candles ago)
            market_data: Full DataFrame with OHLCV data
            current_index: Current row index

        Returns:
            Float value from the appropriate column and lookback offset
        """
        # Calculate lookback index
        lookback_index = current_index - param

        # Safety: Don't look back before data starts
        if lookback_index < 0:
            lookback_index = 0

        # Get the column name for this symbol
        column = self.SYMBOL_TO_COLUMN[indicator]

        # Return the value
        return float(market_data.iloc[lookback_index][column])

    def _evaluate_aggregation(
        self,
        func: AggregationFunc,
        indicator: Indicator,
        window: int,
        market_data: pd.DataFrame,
        current_index: int
    ) -> float:
        """
        Evaluate an aggregation function over a window of indicator values.

        Args:
            func: Aggregation function (AVG, SUM, MAX, MIN, STD)
            indicator: The indicator to aggregate (ALPHA, BETA, etc.)
            window: Window size for aggregation
            market_data: Full DataFrame with OHLCV data
            current_index: Current row index

        Returns:
            Aggregated value

        Example:
            AVG(DELTA, 20) = average of last 20 close prices
            MAX(BETA, 10) = maximum of last 10 high prices
        """
        # Calculate window boundaries
        start_index = max(0, current_index - window + 1)
        end_index = current_index + 1  # +1 because slice is exclusive

        # Get the column name for this indicator
        column = self.SYMBOL_TO_COLUMN[indicator]

        # Extract the window of values
        window_data = market_data.iloc[start_index:end_index][column].values

        # Handle empty window
        if len(window_data) == 0:
            return 0.0

        # Apply the aggregation function
        if func == AggregationFunc.AVG:
            return float(np.mean(window_data))
        elif func == AggregationFunc.SUM:
            return float(np.sum(window_data))
        elif func == AggregationFunc.MAX:
            return float(np.max(window_data))
        elif func == AggregationFunc.MIN:
            return float(np.min(window_data))
        elif func == AggregationFunc.STD:
            return float(np.std(window_data))
        else:
            raise ValueError(f"Unknown aggregation function: {func}")

    def _evaluate_expression(
        self,
        expression: Expression,
        market_data: pd.DataFrame,
        current_index: int
    ) -> float:
        """
        Evaluate an expression (simple indicator, arithmetic operation, or aggregation function).

        Args:
            expression: IndicatorValue, BinaryOp, or FunctionCall
            market_data: Full DataFrame with OHLCV data
            current_index: Current row index

        Returns:
            Float value of the expression
        """
        if isinstance(expression, IndicatorValue):
            # Simple indicator - get its value directly
            return self._get_indicator_value(
                expression.indicator,
                expression.param,
                market_data,
                current_index
            )
        elif isinstance(expression, BinaryOp):
            # Arithmetic operation - recursively evaluate left and right
            left_val = self._evaluate_expression(expression.left, market_data, current_index)
            right_val = self._evaluate_expression(expression.right, market_data, current_index)

            # Apply the operation
            if expression.op == ArithmeticOp.ADD:
                return left_val + right_val
            elif expression.op == ArithmeticOp.SUBTRACT:
                return left_val - right_val
            elif expression.op == ArithmeticOp.MULTIPLY:
                return left_val * right_val
            elif expression.op == ArithmeticOp.DIVIDE:
                # Prevent division by zero
                if right_val == 0:
                    return float('inf') if left_val >= 0 else float('-inf')
                return left_val / right_val
            else:
                raise ValueError(f"Unknown arithmetic operator: {expression.op}")
        elif isinstance(expression, FunctionCall):
            # Aggregation function - calculate over window
            return self._evaluate_aggregation(
                expression.func,
                expression.indicator,
                expression.window,
                market_data,
                current_index
            )
        else:
            raise TypeError(f"Unknown expression type: {type(expression)}")

    def _execute_single_rule(self, rule: Rule, market_data: pd.DataFrame | dict, current_index: int = None) -> Action:
        """Execute a single rule and return its action. Supports both V1 and V2 conditions."""

        # Backward compatibility: if market_data is a dict, use dummy values
        if isinstance(market_data, dict):
            left_value = 100
            right_value = 98
        else:
            # New path: use real market data with lookback
            if current_index is None:
                raise ValueError("current_index must be provided when market_data is a DataFrame")

            # V2: Use expression evaluation if condition has left/right expressions
            if rule.condition.left is not None and rule.condition.right is not None:
                left_value = self._evaluate_expression(
                    rule.condition.left,
                    market_data,
                    current_index
                )
                right_value = self._evaluate_expression(
                    rule.condition.right,
                    market_data,
                    current_index
                )
            else:
                # V1: Use legacy indicator1/indicator2 fields
                left_value = self._get_indicator_value(
                    rule.condition.indicator1,
                    rule.condition.param1,
                    market_data,
                    current_index
                )
                right_value = self._get_indicator_value(
                    rule.condition.indicator2,
                    rule.condition.param2,
                    market_data,
                    current_index
                )

        # Evaluate the condition
        condition_met = False
        op = rule.condition.operator
        if op == Operator.GREATER_THAN:
            condition_met = left_value > right_value
        elif op == Operator.LESS_THAN:
            condition_met = left_value < right_value
        elif op == Operator.GREATER_THAN_OR_EQUAL:
            condition_met = left_value >= right_value
        elif op == Operator.LESS_THAN_OR_EQUAL:
            condition_met = left_value <= right_value
        elif op == Operator.EQUAL:
            condition_met = left_value == right_value

        if condition_met:
            return rule.true_action
        else:
            return rule.false_action

    def _aggregate_actions(self, actions: list[Action]) -> Action:
        """
        Aggregate multiple strategy signals into a single action.

        Modes:
        - MAJORITY: Most common action wins (BUY/SELL/HOLD). On tie, HOLD.
        - UNANIMOUS: All strategies must agree, otherwise HOLD.
        - FIRST: Only use the first strategy's signal (legacy mode).
        """
        if not actions:
            return Action.HOLD

        if self.aggregation_mode == AggregationMode.FIRST:
            return actions[0]

        if self.aggregation_mode == AggregationMode.UNANIMOUS:
            if all(a == actions[0] for a in actions):
                return actions[0]
            else:
                return Action.HOLD

        # MAJORITY mode (default)
        action_counts = Counter(actions)
        most_common = action_counts.most_common()

        # If there's a clear winner, return it
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            return most_common[0][0]

        # On tie, prefer HOLD
        return Action.HOLD

    def _expression_to_string(self, expr: Expression) -> str:
        """Convert an expression to its string representation."""
        if isinstance(expr, IndicatorValue):
            # Simple indicator
            ind = expr.indicator.value
            param = f"({expr.param})" if expr.param else "()"
            return f"{ind}{param}"
        elif isinstance(expr, FunctionCall):
            # Aggregation function
            func = expr.func.value
            ind = expr.indicator.value
            window = expr.window
            return f"{func}({ind}, {window})"
        elif isinstance(expr, BinaryOp):
            # Arithmetic operation - recursively convert left and right
            left_str = self._expression_to_string(expr.left)
            right_str = self._expression_to_string(expr.right)
            op_str = expr.op.value

            # Add parentheses to preserve order of operations
            # Simple heuristic: parenthesize if child is also a BinaryOp
            if isinstance(expr.left, BinaryOp):
                left_str = f"({left_str})"
            if isinstance(expr.right, BinaryOp):
                right_str = f"({right_str})"

            return f"{left_str} {op_str} {right_str}"
        else:
            return str(expr)

    def to_string(self, program: DslProgram) -> str:
        """
        Convert a DslProgram back to a string representation.
        Multiple rules are separated by newlines.
        Supports both V1 and V2 syntax.
        """
        if not program:
            return ""

        rule_strings = []
        for rule in program:
            # V2: Use expression-based formatting if available
            if rule.condition.left is not None and rule.condition.right is not None:
                left_str = self._expression_to_string(rule.condition.left)
                right_str = self._expression_to_string(rule.condition.right)
            else:
                # V1: Use legacy indicator1/indicator2 fields
                ind1 = rule.condition.indicator1.value
                param1 = f"({rule.condition.param1})" if rule.condition.param1 else "()"
                left_str = f"{ind1}{param1}"

                ind2 = rule.condition.indicator2.value
                param2 = f"({rule.condition.param2})" if rule.condition.param2 else "()"
                right_str = f"{ind2}{param2}"

            op = rule.condition.operator.value
            true_action = rule.true_action.value
            false_action = rule.false_action.value

            rule_str = f"IF {left_str} {op} {right_str} THEN {true_action} ELSE {false_action}"
            rule_strings.append(rule_str)

        return "\n".join(rule_strings)
