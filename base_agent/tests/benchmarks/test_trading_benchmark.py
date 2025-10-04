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
        source_data_path.write_text("timestamp,open,high,low,close,volume\n2025-01-01,100,102,99,101,1000\n")

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
        answer_file.write_text("IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL")

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
        answer_file.write_text("IF ALPHA(10) > BETA(50) THEN BUY ELSE HOLD")

        # Create data file with lowercase column names
        data_path = agent_workdir / "ohlcv.csv"
        # Simple uptrend from 100 to 102
        data_path.write_text("timestamp,open,high,low,close,volume\n"
                             "2025-01-01,100,101,99,100,1000\n"
                             "2025-01-02,100,102,100,101,1000\n"
                             "2025-01-03,101,103,101,102,1000\n")
        
        score, error, discussion = await benchmark.score_problem(problem, str(agent_workdir), str(agent_answer_dir), "test_container")

        assert error is None
        # Backtest logic:
        # Day 1: Buys at close 100. Cash = 10000 - 0.10 = 9999.90. Position = 9999.90 / 100 = 99.999 shares.
        # End of sim: Liquidate at 102. Cash = 99.999 * 102 = 10199.898.
        # Trading Profit = 10199.898 - 10000 = 199.898 ≈ 199.90
        # LLM Cost = 5000 * 0.000003 + 100 * 0.000015 = 0.015 + 0.0015 = 0.0165
        # Fitness = 199.90 - 0.0165 ≈ 199.88
        
        # The score should be positive (agent survived)
        assert score > 0
        # The score should be close to trading profit minus LLM cost
        assert score == pytest.approx(199.88, abs=0.1)
        assert "SURVIVED" in discussion
        assert "Fitness:" in discussion
        assert "LLM Cost:" in discussion

    async def test_score_problem_agent_dies_zero_balance(self, benchmark, problem, tmp_path):
        """Test that the agent dies if its balance goes to zero during trading."""
        agent_workdir = tmp_path / "workdir"
        agent_answer_dir = tmp_path / "answer_dir"
        agent_workdir.mkdir()
        agent_answer_dir.mkdir()

        # Create answer file
        answer_file = agent_answer_dir / "answer.txt"
        # A strategy that will buy and sell repeatedly, losing money on transaction costs
        answer_file.write_text("IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL")

        # Create data file with a severe downtrend that will wipe out the balance
        data_path = agent_workdir / "ohlcv.csv"
        # Severe downtrend: 100 -> 10 over a few days
        data_path.write_text("timestamp,open,high,low,close,volume\n"
                             "2025-01-01,100,100,90,90,1000\n"
                             "2025-01-02,90,90,70,70,1000\n"
                             "2025-01-03,70,70,40,40,1000\n"
                             "2025-01-04,40,40,10,10,1000\n"
                             "2025-01-05,10,10,1,1,1000\n")
        
        score, error, discussion = await benchmark.score_problem(problem, str(agent_workdir), str(agent_answer_dir), "test_container")

        assert error is None
        # The agent should have died (balance went to zero)
        # The score should be very negative (initial capital lost)
        assert score < -9000  # Lost most or all of the initial $10,000
        assert "DIED" in discussion or "died" in discussion

    async def test_generate_mutated_strategy(self, benchmark, tmp_path):
        """Test that the benchmark can mutate a strategy."""
        # Create a base strategy file
        base_strategy_path = tmp_path / "base_strategy.txt"
        base_strategy_path.write_text("IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL")
        
        # Generate a mutated strategy
        output_path = tmp_path / "mutated_strategy.txt"
        success = benchmark.generate_mutated_strategy(base_strategy_path, output_path)
        
        assert success
        assert output_path.exists()
        
        # Read and verify the mutated strategy
        mutated_strategy = output_path.read_text().strip()
        assert mutated_strategy != "IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL"
        assert "IF" in mutated_strategy
        assert "THEN" in mutated_strategy
        assert "ELSE" in mutated_strategy
        
        # Verify it parses correctly
        program = benchmark.interpreter.parse(mutated_strategy)
        assert program is not None
