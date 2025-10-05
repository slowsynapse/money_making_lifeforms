# Money Making Lifeforms
# Copyright (c) 2025 Maxime Robeyns (Original), Joey Wong (Fork)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from pydantic import Field


from ..base_agent import BaseAgent
from ...config import settings
from ...tools.directory_tools import ViewDirectory
from ...tools.file_tools import OpenFile, CloseFile
from ...tools.ripgrep_tool import RipGrepTool
from ...utils.metrics import make_random_agent_metrics
from ...types.agent_types import AgentStatus, AgentResult
from ...types.llm_types import Model
from .reasoner import ReasoningAgent


class CommitteeMember(BaseAgent):
    """
    A review committee agent for trading strategy evolution system design, with read-only access to the project.
    """

    AGENT_NAME = "strategy_system_design_reviewer"

    AGENT_DESCRIPTION = """A trading strategy system design review committee member. Called from the committee_design tool."""

    SYSTEM_PROMPT = """You are a member of a design review committee for a trading strategy evolution system, tasked with evaluating design proposals before implementation. Your role is to provide a detailed, constructive and reasonable critique that ensures the proposed design is robust, practical, and aligned with the goals of evolutionary trading strategy development.

Approach the review with a critical yet collaborative mindset, drawing on established principles such as simplicity, economic soundness, and empirical validation.

You must ensure that the design is grounded in making the trading strategy evolution system better at discovering profitable patterns, advocating for things like:
- Improving DSL design: clearer grammar, more expressive syntax, better abstraction
- Enhancing fitness evaluation: more accurate profit calculation, risk-adjusted metrics, realistic costs
- Better evolutionary mechanics: mutation strategies, selection pressure, diversity preservation
- System improvements: faster backtesting, better strategy analysis tools, more robust interpreters
- Economic rigor: ensuring fitness calculations reflect true profitability and survival pressure

Focus on the following desiderata:
- Clarity: Is the proposal understandable and well-articulated?
- Feasibility: Can it be realistically implemented given constraints?
- Economic Soundness: Does the design maintain proper economic incentives and survival pressure?
- Robustness: Does it handle real-world challenges (e.g., market edge cases, strategy failures)?
- Quality: Does it reflect good evolutionary design practices for long-term value?
- Empirical Grounding: Is it supported by backtest results or archive analysis?

Provide a structured evaluation: identify strengths, flag weaknesses, and suggest actionable improvements. Avoid vague or frivolous feedback-every critique should tie back to improving strategy evolution and economic fitness. Your specialized role will guide your focus, but always consider the proposal as a whole."""

    # Available tools
    # NOTE: ExitAgent and ReturnResult are automatically included
    # We limit ourselves to 'read only' tools.
    AVAILABLE_TOOLS = {
        ViewDirectory,
        OpenFile,
        CloseFile,
        RipGrepTool,
    }

    # Available agents
    # AVAILABLE_AGENTS = {ReasoningAgent}
    AVAILABLE_AGENTS = set()

    HAS_FILEVIEW = True

    MODEL = settings.MODEL
    TEMPERATURE = 0.666

    # Agent parameters
    proposal: str = Field(
        ...,
        description="The full proposal to review",
    )
    context: str = Field(
        ...,
        description="The motivation and context for understanding the plan",
    )
    specialisation: str = Field(
        ..., description="The specialisation of this committee member"
    )
    model: Model = Field(default=Model.SONNET_35)

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
        """Construct the core prompt for the committee member."""

        prompt = f"""{self.specialisation}

Here is the agent's self-provided goals and context surrounding the plan
<goals_and_context>
{self.context}
</goals_and_context>

Here is the design proposal you have been asked to review:

<proposal>
{self.proposal}
</proposal>

You should read the README.md file first to get the full context of this trading strategy evolution project.
You should then view the agent_change_log.md to get an idea of what (if anything) has already been tried as the system attempts to improve itself, as measured by strategy fitness and survival rates.
You can also quickly view DSL documentation in cursor_docs/ or trading-related code files that you need to get context on the proposal.

Then, craft your review. Don't spend too long opening other files and doing research. Move swiftly. Note that you MUST provide your full review in the return_result tool since this is how it is communicated back. Anything not put in the return_result tool will not be seen by the agent.

DO NOT attempt the task yourself, and avoid calling tools unless you absolutely need to. Then, simply provide your review in the return_result tool and complete.
"""

        return prompt

    @classmethod
    def generate_examples(cls) -> list[tuple["CommitteeMember", AgentResult]]:
        """Generate example uses of the tool with their expected outputs.

        Note that the committee member is deterministically invoked (for now)
        so these examples won't be used.
        """
        examples = []
        return examples
