import pytest
import asyncio
import sys
from pathlib import Path

# Add the parent directory of 'agent_code' to the path
# so that we can import `agent_code.agent` as a package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_code.agent import Agent

@pytest.mark.asyncio
async def test_agent_execution(tmp_path: Path):
    """
    Test the main execution flow of the Agent.
    """
    workdir = tmp_path / "workdir"
    logdir = tmp_path / "logdir"

    agent = Agent(workdir=workdir, logdir=logdir, server_enabled=False, debug_mode=True)

    test_prompt = "Write a hello world script in python and then terminate."

    try:
        tokens, cached, cost, duration = await agent.exec(
            problem=test_prompt,
            timeout=120,  # 120-second timeout to prevent hanging
        )
        await agent.create_report()
    except Exception as e:
        pytest.fail(f"Agent execution failed with an exception: {e}")

    assert isinstance(tokens, int)
    assert isinstance(cached, int)
    assert isinstance(cost, float)
    assert isinstance(duration, float)
    assert duration > 0
    assert tokens > 0
    assert cost > 0.0

    assert (logdir / "trace.txt").exists()
    assert (logdir / "execution_tree.txt").exists()
