import re
from .grammar import Indicator, Operator, Action, Condition, Rule, DslProgram

class DslInterpreter:
    """
    Parses and executes a DSL program.
    """
    def __init__(self):
        # Example DSL string: IF SMA(10) > SMA(50) THEN BUY ELSE SELL
        self.rule_pattern = re.compile(
            r"IF\s+"
            r"(\w+)\((\d+)\)\s*"       # Indicator 1 and param 1
            r"([><]=?)\s*"            # Operator
            r"(\w+)\((\d+)\)\s*"       # Indicator 2 and param 2
            r"THEN\s+(\w+)\s*"        # True Action
            r"ELSE\s+(\w+)"           # False Action
        )

    def parse(self, dsl_string: str) -> DslProgram | None:
        """
        Parses a DSL string into a structured program.
        Returns None if parsing fails.
        """
        match = self.rule_pattern.match(dsl_string.strip())
        if not match:
            return None

        try:
            ind1_str, param1_str, op_str, ind2_str, param2_str, true_action_str, false_action_str = match.groups()

            condition = Condition(
                indicator1=Indicator[ind1_str],
                param1=int(param1_str),
                operator=self._string_to_operator(op_str),
                indicator2=Indicator[ind2_str],
                param2=int(param2_str)
            )

            rule = Rule(
                condition=condition,
                true_action=Action[true_action_str],
                false_action=Action[false_action_str]
            )
            
            # For now, our program is just a single rule
            return [rule]
        except (KeyError, ValueError):
            # Invalid indicator, operator, or action name
            return None

    def _string_to_operator(self, op_str: str) -> Operator:
        for op in Operator:
            if op.value == op_str:
                return op
        raise ValueError(f"Invalid operator: {op_str}")

    def execute(self, program: DslProgram, market_data: dict) -> Action:
        """
        Executes a parsed DSL program against the current market data.
        
        market_data is a placeholder for a single time-step (e.g., one day's OHLCV).
        """
        # For now, we only handle the first rule
        rule = program[0]
        
        # --- Placeholder Indicator Calculation ---
        # In the future, this will calculate real indicator values from market_data.
        # For now, let's use dummy values to make the logic work.
        indicator1_value = 100 # e.g., market_data['sma_10']
        indicator2_value = 98  # e.g., market_data['sma_50']
        
        condition_met = False
        op = rule.condition.operator
        if op == Operator.GREATER_THAN:
            condition_met = indicator1_value > indicator2_value
        elif op == Operator.LESS_THAN:
            condition_met = indicator1_value < indicator2_value
        # (Other operators would be handled here)

        if condition_met:
            return rule.true_action
        else:
            return rule.false_action
