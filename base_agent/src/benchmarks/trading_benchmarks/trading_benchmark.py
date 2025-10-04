import pandas as pd
from typing import ClassVar
from pathlib import Path

from ..base import BaseBenchmark, Problem
from ...dsl.interpreter import DslInterpreter
from ...dsl.grammar import Action

class TradingBenchmark(BaseBenchmark):
    """
    A benchmark for evaluating the agent's ability to generate profitable trading strategies.
    """
    name: ClassVar[str] = "trading"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._problems = self._load_problems()
        self.interpreter = DslInterpreter()

    @property
    def problems(self) -> list[Problem]:
        return self._problems

    def _load_problems(self) -> list[Problem]:
        return [
            Problem(
                problem_id="trend_following_1",
                statement="The market is in a clear uptrend. Develop a strategy to follow the trend.",
                answer=None,
                answer_discussion=None,
            )
        ]

    async def setup_problem(self, problem: Problem, problem_data_dir: Path, container_name: str) -> None:
        """
        Copies the OHLCV data into the agent's working directory.
        """
        source_data_path = Path("benchmark_data/trading/ohlcv.csv")
        destination_data_path = problem_data_dir / "ohlcv.csv"
        destination_data_path.write_bytes(source_data_path.read_bytes())
        print(f"Copied OHLCV data to {destination_data_path}")

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> tuple[float, str | None, str | None]:
        """
        Scores the agent's generated DSL strategy by running a backtest.
        The fitness score is the final profit.
        """
        answer_path = Path(agent_answer_dir) / "answer.txt"
        if not answer_path.exists():
            return -1000.0, "The `answer.txt` file was not found.", "Fitness penalized for missing answer."

        dsl_strategy_string = answer_path.read_text().strip()
        program = self.interpreter.parse(dsl_strategy_string)

        if not program:
            return -1000.0, f"Failed to parse DSL: '{dsl_strategy_string}'", "Fitness penalized for invalid DSL."

        # Load the market data
        data_path = Path(agent_workdir) / "ohlcv.csv"
        if not data_path.exists():
            return -1000.0, "The `ohlcv.csv` file was not found in the workdir.", "Fitness penalized for missing data."
        
        market_data = pd.read_csv(data_path)

        # Run the backtest
        profit, discussion = self._run_backtest(program, market_data)
        
        return profit, None, discussion

    def _run_backtest(self, program: list, market_data: pd.DataFrame) -> tuple[float, str]:
        """
        Runs a simple backtest simulation.
        """
        initial_capital = 10000.0
        capital = initial_capital
        position = 0
        transaction_cost = 0.10
        num_trades = 0

        for i, row in market_data.iterrows():
            # In a real scenario, we would calculate indicators here.
            # For now, the interpreter's execute method uses dummy values.
            market_snapshot = row.to_dict()
            action = self.interpreter.execute(program, market_snapshot)

            if action == Action.BUY and position == 0:
                position = capital / row['Close']
                capital = 0
                capital -= transaction_cost
                num_trades += 1
            elif action == Action.SELL and position > 0:
                capital = position * row['Close']
                position = 0
                capital -= transaction_cost
                num_trades += 1
        
        # At the end, liquidate any open positions
        if position > 0:
            capital = position * market_data.iloc[-1]['Close']
            position = 0

        final_profit = capital - initial_capital
        
        discussion = (
            f"Backtest complete. Initial Capital: ${initial_capital:.2f}, "
            f"Final Capital: ${capital:.2f}, Profit: ${final_profit:.2f}, Trades: {num_trades}"
        )
        return final_profit, discussion
