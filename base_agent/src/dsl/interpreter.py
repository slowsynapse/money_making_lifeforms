import re
import pandas as pd
from collections import Counter
from .grammar import Indicator, Operator, Action, Condition, Rule, DslProgram, AggregationMode

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

    def _execute_single_rule(self, rule: Rule, market_data: pd.DataFrame | dict, current_index: int = None) -> Action:
        """Execute a single rule and return its action."""

        # Backward compatibility: if market_data is a dict, use dummy values
        if isinstance(market_data, dict):
            indicator1_value = 100
            indicator2_value = 98
        else:
            # New path: use real market data with lookback
            if current_index is None:
                raise ValueError("current_index must be provided when market_data is a DataFrame")

            indicator1_value = self._get_indicator_value(
                rule.condition.indicator1,
                rule.condition.param1,
                market_data,
                current_index
            )

            indicator2_value = self._get_indicator_value(
                rule.condition.indicator2,
                rule.condition.param2,
                market_data,
                current_index
            )

        # Evaluate the condition
        condition_met = False
        op = rule.condition.operator
        if op == Operator.GREATER_THAN:
            condition_met = indicator1_value > indicator2_value
        elif op == Operator.LESS_THAN:
            condition_met = indicator1_value < indicator2_value
        elif op == Operator.GREATER_THAN_OR_EQUAL:
            condition_met = indicator1_value >= indicator2_value
        elif op == Operator.LESS_THAN_OR_EQUAL:
            condition_met = indicator1_value <= indicator2_value
        elif op == Operator.EQUAL:
            condition_met = indicator1_value == indicator2_value

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

    def to_string(self, program: DslProgram) -> str:
        """
        Convert a DslProgram back to a string representation.
        Multiple rules are separated by newlines.
        """
        if not program:
            return ""

        rule_strings = []
        for rule in program:
            ind1 = rule.condition.indicator1.value
            param1 = f"({rule.condition.param1})" if rule.condition.param1 else "()"
            op = rule.condition.operator.value
            ind2 = rule.condition.indicator2.value
            param2 = f"({rule.condition.param2})" if rule.condition.param2 else "()"
            true_action = rule.true_action.value
            false_action = rule.false_action.value

            rule_str = f"IF {ind1}{param1} {op} {ind2}{param2} THEN {true_action} ELSE {false_action}"
            rule_strings.append(rule_str)

        return "\n".join(rule_strings)
