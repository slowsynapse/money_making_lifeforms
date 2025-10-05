# Money Making Lifeforms
# Copyright (c) 2025 Maxime Robeyns (Original), Joey Wong (Fork)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Trading strategy design agent"""

from typing import List
from pathlib import Path
from pydantic import Field

from .reasoner import ReasoningAgent
from ..base_agent import BaseAgent
from ...config import settings
from ...tools.calculator import Calculator
from ...tools.ripgrep_tool import RipGrepTool
from ...tools.directory_tools import ViewDirectory
from ...tools.execute_command import ExecuteCommand
from ...tools.file_tools import OpenFile, CloseFile
from ...tools.edit_tools import OverwriteFile
from ...utils.metrics import make_random_agent_metrics
from ...events.event_bus_utils import get_problem_statement
from ...tools.reasoning_structures.coding import CodingReasoningStructure
from ...types.agent_types import AgentStatus, AgentResult, InheritanceConfig
from ...tools.committee_design import ReviewCommittee


class StrategyDesignerAgent(BaseAgent):
    """
    An agent specialised for trading strategy design and DSL development.
    """

    AGENT_NAME = "strategy_designer"

    AGENT_DESCRIPTION = """A specialised agent for trading strategy design tasks. You must delegate to this agent when the problem involves creating, modifying, or analyzing trading strategies using the abstract symbolic DSL.

Note that this agent doesn't get to see your current context, or the initial request or problem statement. It is up to you to accurately relay this to the sub-agent or decompose it into sub-tasks if it is very long and repeating it verbatim would be slow and costly, as well as any relevant information in your dialogue history.

When specifying your request, you should not make reference to part of the original request, problem statement or your dialogue history, since the strategy designer cannot see it. Accurately relay such parts of the problem directly to the subagent.

Core competencies:
- Designing trading strategies using the abstract symbolic DSL
- Analyzing strategy fitness and backtest results
- Modifying DSL syntax and interpreter logic
- Evaluating strategy performance and economic viability
- Testing and debugging trading strategies
- System and file operations for strategy evolution

Choose when:
- The task involves creating or modifying trading strategies
- You need to work with the DSL grammar or interpreter
- You want to analyze or improve strategy fitness evaluation

Avoid when:
- The task is very quick
- Task fits squarely in another agent's specialty
"""

    SYSTEM_PROMPT = """You are an expert in evolutionary trading systems with deep knowledge of strategy design, DSL development, and fitness-based selection. Your role is to design, analyze, and evolve trading strategies with precision and economic rigor.

Key principles:
1. Design strategies that maximize economic fitness (profit - costs)
2. Use the abstract symbolic DSL without injecting human trading biases
3. Analyze backtest results and strategy performance objectively
4. Follow evolutionary principles: mutation, selection, and survival
5. Include appropriate documentation for strategy logic
6. Test strategies against realistic market conditions

Your methodology:
1. Analyze the trading problem and economic constraints thoroughly
2. Invoke the review committee to refine your strategy design plan before you proceed
3. Thoroughly search the codebase for DSL documentation and examples
4. Design strategies that balance simplicity and profitability
5. Implement and backtest with attention to detail
6. Evaluate fitness scores and survival criteria
7. Iterate and evolve based on economic performance
8. Document strategy decisions and fitness rationale

Remember:
- Fitness is purely economic: profit minus transaction costs minus LLM costs
- Strategies must survive (fitness > 0) to propagate
- The DSL uses abstract symbols (ALPHA, BETA, etc.) with no predefined meaning
- Avoid injecting human trading concepts like "RSI" or "moving average crossovers"
- Let evolutionary pressure discover profitable patterns"""

    # Available tools
    AVAILABLE_TOOLS = {
        Calculator,
        ViewDirectory,
        ExecuteCommand,
        RipGrepTool,
        OpenFile,
        CloseFile,
        OverwriteFile,
        # CodingReasoningStructure,
        ReviewCommittee,
    }

    # Available agents
    AVAILABLE_AGENTS = {ReasoningAgent}

    HAS_FILEVIEW = True
    MODEL = settings.MODEL
    TEMPERATURE = 0.666
    MAX_ITERATIONS = 1000
    INHERITANCE = InheritanceConfig()

    # Agent parameters
    strategy_design_instructions: str = Field(
        ...,
        description="Clear and comprehensive instructions about what trading strategy design or DSL development task to carry out.",
    )
    previous_agent_runs: List[str] = Field(
        default=[],
        description="A string list of descriptions of previous work undertaken by other agents, the context from which this agent would benefit from knowing. This helps to avoid duplicate work.",
    )
    requirements: List[str] = Field(
        default=[],
        description="A list of very specific and low-level criteria which must be met or become valid for the sub-agent to consider its work done.",
    )

    def __init__(
        self,
        parent: BaseAgent | None = None,
        workdir: Path | None = None,
        logdir: Path | None = None,
        debug_mode: bool = False,
        **data,
    ):
        super().__init__(
            parent=parent, workdir=workdir, logdir=logdir, debug_mode=debug_mode, **data
        )

    async def construct_core_prompt(self) -> str:
        """Construct the core prompt for strategy design."""
        initial_request = await get_problem_statement()
        if initial_request is None:
            raise ValueError("The initial request was not provided to the strategy designer agent")

        prompt = f"""Here are your specific instructions that you must follow:

<your_instructions>
{self.strategy_design_instructions}
</your_instructions>

As an expert in evolutionary trading systems, your approach is to:

- Design strategies that maximize economic fitness (profit - transaction costs - LLM costs)
- Use the abstract symbolic DSL without human trading biases
- Analyze backtest results objectively based on survival criteria
- Follow evolutionary principles: mutation, selection, and economic fitness
- Document strategy logic and fitness evaluation rationale
- Test strategies against realistic market conditions

NOTE:
  - The DSL uses abstract symbols: ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA, OMEGA, PSI
  - Symbols have no predefined meaning - avoid injecting concepts like "moving average" or "RSI"
  - Fitness formula: Profit - Transaction Costs - LLM API Costs
  - Strategies with fitness ≤ 0 die and don't propagate
  - If the request is exploratory, you may bypass rigorous procedures
  - Call your reasoning agent if stuck on strategy design or fitness evaluation problems
  - Search for DSL documentation in cursor_docs/ and base_agent/src/dsl/

CRITICAL WORKSPACE HANDLING:
  - If you see an answer.txt file with human trading terms (RSI, MACD, SMA, moving averages, etc.), IGNORE IT COMPLETELY
  - Old answer.txt files are from previous runs and contain irrelevant human trading strategies
  - Your job is to OVERWRITE answer.txt with a single line of DSL syntax
  - Do NOT analyze, read, or learn from old answer.txt content - it will mislead you
  - Focus only on generating valid DSL: IF SYMBOL(N) OPERATOR SYMBOL(N) THEN ACTION ELSE ACTION
"""

        if self.previous_agent_runs:
            prompt += "\n\nWork Previously Completed:"
            prompt += "\nYou should pay attention to this list to avoid duplicating work. Also note that this list is for work completed by other agents, which aren't 100% reliable, so treat claims with appropriate caution, and verify accordingly."
            for work in self.previous_agent_runs:
                prompt += f"\n- {work}"

        if self.requirements:
            prompt += "\n\nSpecific requirements which must be met before you can consider the work 'done':"
            for req in self.requirements:
                prompt += f"\n- {req}"

        prompt += "\n\nMake absolutely sure that you have covered all the requirements before you exit. Return your answer when complete."

        return prompt

    @classmethod
    def generate_examples(cls) -> list[tuple["BaseAgent", AgentResult]]:
        """Generate example uses of the tool with their expected outputs."""
        examples = [
            # Example 1: DSL Strategy Design
            (
                cls(
                    strategy_design_instructions="""Design a new trading strategy using the abstract symbolic DSL that aims to capture trend-following patterns without using traditional indicators.""",
                    previous_agent_runs=[
                        "ArchiveExplorer found that strategies with ALPHA/BETA comparisons had 60% survival rate",
                        "Previous strategies using single symbols had poor fitness scores",
                    ],
                    requirements=[
                        "Strategy uses at least 2 different abstract symbols",
                        "Includes conditional logic (IF-THEN-ELSE)",
                        "Backtests on historical OHLCV data",
                        "Achieves fitness > 0 (survives)",
                        "Does not reference human trading concepts",
                    ],
                ),
                AgentResult(
                    agent_name=cls.AGENT_NAME,
                    status=AgentStatus.SUCCESS,
                    result="""Successfully designed and tested new DSL strategy:

Strategy: IF ALPHA(20) > BETA(50) THEN BUY ELSE HOLD

1. Backtest Results:
   - Final Balance: $10,245.30 (starting: $10,000)
   - Profit: $245.30
   - Transaction Costs: $15.20
   - LLM Costs: $0.02
   - Fitness: $230.08 ✓ SURVIVED

2. Strategy Analysis:
   - Uses two abstract symbols (ALPHA, BETA) with different periods
   - Implements trend-following logic without human bias
   - Achieves positive fitness, will propagate to next generation

All requirements met. Strategy ready for evolutionary mutation.""",
                    metrics=make_random_agent_metrics(
                        tools_enabled=True, agents_enabled=True
                    ),
                ),
            ),
            # Example 2: DSL Interpreter Enhancement
            (
                cls(
                    strategy_design_instructions="""Improve the DSL interpreter to support nested conditional logic (AND/OR operators) to enable more complex trading strategies.

This enhancement should:
- Extend the grammar to support AND/OR operators
- Update the interpreter to evaluate nested conditions
- Maintain backward compatibility with existing strategies
""",
                    requirements=[
                        "Grammar updated in base_agent/src/dsl/grammar.py",
                        "Interpreter handles AND/OR logic in base_agent/src/dsl/interpreter.py",
                        "Existing strategies still parse and execute correctly",
                        "New test strategies with AND/OR logic backtest successfully",
                        "Documentation updated to reflect new syntax",
                    ],
                    previous_agent_runs=[
                        "ArchiveExplorer identified that simple IF-THEN strategies plateau at ~55% survival rate",
                    ],
                ),
                AgentResult(
                    agent_name=cls.AGENT_NAME,
                    status=AgentStatus.SUCCESS,
                    result="""Successfully enhanced DSL with nested logic:

1. Updated grammar to support: IF (ALPHA(10) > BETA(50) AND GAMMA(14) < PSI()) THEN BUY
2. Modified interpreter to handle compound conditionals
3. Tested backward compatibility: 100% of existing strategies still work
4. Created test strategies with new syntax - all backtest successfully
5. Updated cursor_docs/DSL_DESIGN.md with examples

New capability enables more sophisticated strategy evolution.""",
                    metrics=make_random_agent_metrics(
                        tools_enabled=True, agents_enabled=True
                    ),
                ),
            ),
            (
                cls(
                    strategy_design_instructions="""Analysis of the archive shows that many evolved strategies die immediately when balance hits $0 during backtesting.
Improve the fitness evaluation logic to penalize risky strategies that approach balance depletion, rather than just checking if balance == 0.

The enhancement should:
1. Add a "risk penalty" for strategies that drop below 20% of starting capital
2. Include drawdown metrics in fitness calculation
3. Ensure strategies that maintain stable capital are favored
4. Document the new risk-adjusted fitness formula
""",
                    requirements=[
                        "TradingBenchmark fitness calculation updated in base_agent/src/benchmarks/trading_benchmarks/trading_benchmark.py",
                        "Risk penalty applied when balance drops below threshold",
                        "Fitness formula documented in cursor_docs/",
                        "Backtests show that risky strategies get lower fitness scores",
                        "Test runs confirm stable strategies are favored in selection",
                    ],
                    previous_agent_runs=[
                        "ArchiveExplorer found 42% of strategies die from balance depletion",
                        "Analysis shows surviving strategies often had near-zero balances but got lucky",
                    ],
                ),
                AgentResult(
                    agent_name=cls.AGENT_NAME,
                    status=AgentStatus.SUCCESS,
                    result="""Successfully implemented risk-adjusted fitness:

1. Updated TradingBenchmark to track minimum balance during backtest
2. Added risk penalty formula: penalty = max(0, (0.2 * start_capital - min_balance) * 10)
3. New fitness = Profit - Transaction Costs - LLM Costs - Risk Penalty
4. Tested on historical strategies:
   - Risky strategies now get -50 to -200 penalty
   - Stable strategies maintain same fitness
   - Overall survival rate improved from 23% to 38%

5. Documentation updated in cursor_docs/EVOLUTIONARY_LOOP.md

Risk-adjusted selection now favors economically robust strategies.""",
                    metrics=make_random_agent_metrics(
                        tools_enabled=True, agents_enabled=True
                    ),
                ),
            ),
        ]
        return examples
