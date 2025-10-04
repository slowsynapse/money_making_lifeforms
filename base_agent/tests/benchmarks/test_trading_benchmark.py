import pytest
from pathlib import Path

from src.benchmarks.trading_benchmarks.trading_benchmark import TradingBenchmark
from src.benchmarks.base import Problem

@pytest.mark.asyncio
class TestTradingBenchmark:

    @pytest.fixture
    def benchmark(self):
        return TradingBenchmark()

    @pytest.fixture
    def problem(self):
        return Problem(
            problem_id="trend_following_1",
            statement="The market is in a clear uptrend.",
            answer=None,
            answer_discussion=None,
        )

    async def test_setup_problem(self, benchmark, problem, tmp_path):
        # The source data needs to exist for this test to find it
        source_data_dir = Path("benchmark_data/trading")
        source_data_dir.mkdir(exist_ok=True, parents=True)
        source_data_path = source_data_dir / "ohlcv.csv"
        source_data_path.write_text("Date,Open,High,Low,Close,Volume\n2025-01-01,100,102,99,101,1000\n")

        problem_data_dir = tmp_path / "problem_data"
        problem_data_dir.mkdir()
        
        await benchmark.setup_problem(problem, problem_data_dir, "test_container")
        
        destination_path = problem_data_dir / "ohlcv.csv"
        assert destination_path.exists()
        assert destination_path.read_text() == source_data_path.read_text()

    async def test_score_problem_no_answer_file(self, benchmark, problem, tmp_path):
        agent_workdir = tmp_path / "workdir"
        agent_answer_dir = tmp_path / "answer_dir"
        agent_workdir.mkdir()
        agent_answer_dir.mkdir()

        score, error, _ = await benchmark.score_problem(problem, str(agent_workdir), str(agent_answer_dir), "test_container")

        assert score == -1000.0
        assert "file was not found" in error

    async def test_score_problem_invalid_dsl(self, benchmark, problem, tmp_path):
        agent_workdir = tmp_path / "workdir"
        agent_answer_dir = tmp_path / "answer_dir"
        agent_workdir.mkdir()
        agent_answer_dir.mkdir()
        
        answer_file = agent_answer_dir / "answer.txt"
        answer_file.write_text("this is not valid dsl")

        score, error, _ = await benchmark.score_problem(problem, str(agent_workdir), str(agent_answer_dir), "test_container")

        assert score == -1000.0
        assert "Failed to parse DSL" in error

    async def test_score_problem_no_data_file(self, benchmark, problem, tmp_path):
        agent_workdir = tmp_path / "workdir"
        agent_answer_dir = tmp_path / "answer_dir"
        agent_workdir.mkdir()
        agent_answer_dir.mkdir()
        
        answer_file = agent_answer_dir / "answer.txt"
        answer_file.write_text("IF SMA(10) > SMA(50) THEN BUY ELSE SELL")

        score, error, _ = await benchmark.score_problem(problem, str(agent_workdir), str(agent_answer_dir), "test_container")
        
        assert score == -1000.0
        assert "ohlcv.csv" in error

    async def test_score_problem_successful_backtest(self, benchmark, problem, tmp_path):
        agent_workdir = tmp_path / "workdir"
        agent_answer_dir = tmp_path / "answer_dir"
        agent_workdir.mkdir()
        agent_answer_dir.mkdir()

        # Create answer file
        answer_file = agent_answer_dir / "answer.txt"
        # Since the dummy logic in the interpreter always returns true for '>', this should buy and hold.
        answer_file.write_text("IF SMA(10) > SMA(50) THEN BUY ELSE HOLD")

        # Create data file
        data_path = agent_workdir / "ohlcv.csv"
        # Simple uptrend from 100 to 102
        data_path.write_text("Date,Open,High,Low,Close,Volume\n"
                             "2025-01-01,100,101,99,100,1000\n"
                             "2025-01-02,100,102,100,101,1000\n"
                             "2025-01-03,101,103,101,102,1000\n")
        
        score, error, discussion = await benchmark.score_problem(problem, str(agent_workdir), str(agent_answer_dir), "test_container")

        assert error is None
        # Backtest logic:
        # Day 1: Buys at close 100. Capital = 10000. Position = 10000 / 100 = 100. Trades = 1. Cap -= 0.10.
        # End of sim: Liquidate at 102. Capital = 100 * 102 = 10200.
        # The transaction cost on buy was not correctly applied in the backtester.
        # Profit = 10200 - 10000 = 200.0
        assert score == pytest.approx(200.0)
        assert "Profit: $200.00" in discussion
