import pandas as pd
from typing import ClassVar, Dict, Optional, Tuple
from pathlib import Path

from ..base import BaseBenchmark, Problem
from ...dsl.interpreter import DslInterpreter
from ...dsl.mutator import DslMutator
from ...dsl.grammar import Action
from ...storage.models import CellPhenotype

class TradingBenchmark(BaseBenchmark):
    """
    A benchmark for evaluating the agent's ability to generate profitable trading strategies.

    Fitness = Trading Profit - Transaction Costs - LLM API Costs

    The agent must generate more profit than it costs to run, or it "dies".
    """
    name: ClassVar[str] = "trading"

    # Trading parameters - configurable
    INITIAL_CAPITAL: ClassVar[float] = 100.0  # Starting with $100 - HIGH PRESSURE!

    # Hyperliquid perpetuals fees (as of 2025)
    # Taker fee: 0.045% (0.00045), Maker fee: 0.015% (0.00015)
    # We assume all orders are taker orders (market orders)
    TAKER_FEE_RATE: ClassVar[float] = 0.00045  # 0.045% taker fee
    MAKER_FEE_RATE: ClassVar[float] = 0.00015  # 0.015% maker fee (not currently used)
    
    # Approximate Anthropic API pricing (Claude 3.5 Sonnet)
    # These are rough estimates in USD per token
    INPUT_TOKEN_COST = 3.0 / 1_000_000  # $3 per million input tokens
    OUTPUT_TOKEN_COST = 15.0 / 1_000_000  # $15 per million output tokens

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._problems = self._load_problems()
        self.interpreter = DslInterpreter()
        self.mutator = DslMutator()
        
        # Track agent's financial state
        self.agent_balance = self.INITIAL_CAPITAL
        self.total_llm_costs = 0.0
        self.total_trading_profit = 0.0
        self.attempts = 0

    @property
    def problems(self) -> list[Problem]:
        return self._problems

    def _load_problems(self) -> list[Problem]:
        """
        Loads trading strategy problems. For now, we have a single hardcoded problem.
        In the future, this could load from a file or generate multiple problems.
        """
        dsl_statement = """You must generate a trading strategy using a symbolic Domain-Specific Language (DSL).

**CRITICAL INSTRUCTIONS:**

Your ONLY task is to write a single line of text to a file named `answer.txt` containing a strategy in this exact format:

```
IF SYMBOL(N) OPERATOR SYMBOL(N) THEN ACTION ELSE ACTION
```

**Available Symbols:**
You can use any of these symbols with a numeric parameter N in parentheses:
- ALPHA(N)
- BETA(N)
- GAMMA(N)
- DELTA(N)
- EPSILON(N)
- ZETA(N)

Or these symbols without parameters:
- OMEGA()
- PSI()

**Available Operators:**
- > (greater than)
- < (less than)
- >= (greater or equal)
- <= (less or equal)
- == (equal)

**Available Actions:**
- BUY
- SELL
- HOLD

**Example Valid Strategies:**

```
IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL
IF GAMMA(14) < 30 THEN BUY ELSE HOLD
IF OMEGA() > DELTA(20) THEN BUY ELSE SELL
IF PSI() >= EPSILON(100) THEN HOLD ELSE SELL
```

**Requirements:**

1. Write EXACTLY ONE line to `answer.txt`
2. Use the EXACT format: `IF ... THEN ... ELSE ...`
3. Include spaces around operators
4. DO NOT include explanations, comments, or code
5. DO NOT write Python, JSON, or any other language
6. The file must contain ONLY the DSL string

**Fitness Evaluation:**

Your strategy will be tested on market data. You survive if:

```
Fitness = Trading_Profit - Transaction_Costs - API_Costs
```

If Fitness ≤ 0, your strategy dies.

**Action:**

Use the `overwrite_file` tool to create `answer.txt` with your strategy."""

        return [
            Problem(
                problem_id="trend_following_1",
                statement=dsl_statement,
                answer=None,
                answer_discussion=None,
            )
        ]
    
    def get_problem_with_history(self, problem_id: str, history: list[dict]) -> Problem:
        """
        Generate a problem statement that includes performance history from previous attempts.
        
        This allows the agent to learn from its past performance and make informed decisions
        about what strategies to try next.
        
        Args:
            problem_id: The problem identifier
            history: List of dicts with 'strategy', 'fitness', 'survived' keys
            
        Returns:
            Problem with augmented statement including history
        """
        base_problem = self.get_problem(problem_id)
        
        if not history:
            # No history yet - just return the base problem
            return base_problem
        
        # Build history context
        history_text = "\n\n**YOUR PERFORMANCE HISTORY:**\n\n"
        history_text += f"You are currently on attempt #{len(history) + 1}. Here are your previous results:\n\n"
        
        for i, result in enumerate(history[-10:], 1):  # Show last 10 attempts
            strategy = result.get('strategy', 'Unknown')
            fitness = result.get('fitness', 0)
            survived = result.get('survived', False)
            status = "✓ SURVIVED" if survived else "✗ DIED"
            
            history_text += f"Attempt {i}:\n"
            history_text += f"  Strategy: {strategy}\n"
            history_text += f"  Fitness: ${fitness:.2f}\n"
            history_text += f"  Status: {status}\n"
            if not survived:
                reason = result.get('death_reason', 'Unknown')
                history_text += f"  Death Reason: {reason}\n"
            history_text += "\n"
        
        # Add learning guidance with actual state
        estimated_llm_cost = 0.0165  # Rough estimate
        survival_threshold = estimated_llm_cost
        
        history_text += f"""**LEARNING FROM HISTORY:**

Analyze your previous attempts:
- Which symbol combinations led to survival?
- Which operators worked better?
- Did you die from bad trades or from making too few trades?
- What patterns do you notice in successful vs failed strategies?

Use this information to generate a BETTER strategy that has higher fitness.
Remember: Fitness = Profit - Costs. You need positive fitness to survive.

**YOUR CURRENT FINANCIAL STATE:**
- Agent Balance: ${self.agent_balance:.2f}
- Total LLM Costs So Far: ${self.total_llm_costs:.2f}
- Total Trading Profit So Far: ${self.total_trading_profit:.2f}
- Attempts Made: {self.attempts}

**TRADING PARAMETERS (Per Backtest):**
- Starting Capital: ${self.INITIAL_CAPITAL:.2f}
- Transaction Cost: ${self.TRANSACTION_COST:.2f} per trade
- Estimated LLM Cost: ${estimated_llm_cost:.4f} per attempt

**SURVIVAL REQUIREMENT:**
- Minimum profit needed: >${estimated_llm_cost:.4f}
- Your goal: Generate enough profit to cover costs and grow your balance

Now generate your next strategy in `answer.txt`:**"""
        
        # Combine base statement with history
        augmented_statement = base_problem.statement + history_text
        
        return Problem(
            problem_id=problem_id,
            statement=augmented_statement,
            answer=None,
            answer_discussion=None,
        )

    async def setup_problem(self, problem: Problem, problem_data_dir: Path, container_name: str) -> None:
        """
        Copies the OHLCV data into the agent's working directory.
        Also cleans up any old answer.txt files to prevent contamination.
        """
        # Clean up old answer.txt if it exists (prevents contamination from previous runs)
        import shutil
        agent_outputs = problem_data_dir / "agent_outputs"
        if agent_outputs.exists():
            shutil.rmtree(agent_outputs)
            print(f"Cleaned old agent_outputs directory")

        # Copy fresh market data (use PURR data if available, fallback to synthetic)
        purr_data_path = Path("benchmark_data/trading/purr_60d.csv")
        synthetic_data_path = Path("benchmark_data/trading/ohlcv.csv")

        if purr_data_path.exists():
            source_data_path = purr_data_path
            print(f"Using real PURR data from Hyperliquid")
        else:
            source_data_path = synthetic_data_path
            print(f"Using synthetic OHLCV data (PURR data not found)")

        destination_data_path = problem_data_dir / "ohlcv.csv"
        destination_data_path.write_bytes(source_data_path.read_bytes())
        print(f"Copied market data to {destination_data_path}")

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> tuple[float, str | None, str | None]:
        """
        Scores the agent's generated DSL strategy by running a backtest.
        The fitness score is the final profit minus LLM costs.
        
        If fitness <= 0, the agent has failed to sustain itself.
        """
        answer_path = Path(agent_answer_dir) / "answer.txt"
        if not answer_path.exists():
            return -1000.0, "The `answer.txt` file was not found.", "Agent died: No answer submitted."

        dsl_strategy_string = answer_path.read_text().strip()
        program = self.interpreter.parse(dsl_strategy_string)

        if not program:
            return -1000.0, f"Failed to parse DSL: '{dsl_strategy_string}'", "Agent died: Invalid DSL syntax."

        # Load the market data
        data_path = Path(agent_workdir) / "ohlcv.csv"
        if not data_path.exists():
            return -1000.0, "The `ohlcv.csv` file was not found in the workdir.", "Agent died: Missing market data."
        
        market_data = pd.read_csv(data_path)

        # Estimate LLM cost for this run
        # For a simple DSL generation task, estimate ~5000 input tokens and ~100 output tokens
        # This is a rough approximation; in production we'd get actual usage from the agent
        estimated_input_tokens = 5000
        estimated_output_tokens = 100
        llm_cost = (estimated_input_tokens * self.INPUT_TOKEN_COST + 
                   estimated_output_tokens * self.OUTPUT_TOKEN_COST)

        # Run the backtest
        trading_profit, num_trades, survived, backtest_log = self._run_backtest(program, market_data)
        
        # Calculate true fitness: profit - llm_cost
        fitness = trading_profit - llm_cost
        
        survival_status = "SURVIVED" if survived and fitness > 0 else "DIED"
        discussion = (
            f"[{survival_status}] Trading Profit: ${trading_profit:.2f}, "
            f"LLM Cost: ${llm_cost:.4f}, Fitness: ${fitness:.2f}, "
            f"Trades: {num_trades}. {backtest_log}"
        )
        
        return fitness, None, discussion

    def _run_backtest(self, program: list, market_data: pd.DataFrame) -> tuple[float, int, bool, str]:
        """
        Runs a backtest simulation with early termination on zero balance.

        Uses realistic Hyperliquid perpetuals fees (0.045% taker fee).
        Now uses the full DataFrame with lookback support.

        Returns:
            tuple of (profit, num_trades, survived, log_message)
        """
        initial_capital = self.INITIAL_CAPITAL
        cash = initial_capital
        position = 0.0  # Number of shares/contracts held
        num_trades = 0
        survived = True
        total_fees_paid = 0.0

        # Determine minimum start index based on max lookback in program
        min_start_index = self._get_min_required_history(program)

        for i in range(min_start_index, len(market_data)):
            # Execute the DSL strategy with full DataFrame and current index
            action = self.interpreter.execute(program, market_data, current_index=i)

            current_price = market_data.iloc[i]['close']

            # Execute trading actions with percentage-based fees
            if action == Action.BUY and cash > 0:
                # Buy with all available cash
                # Fee is charged on the notional value
                trade_value = cash
                fee = trade_value * self.TAKER_FEE_RATE
                position = (cash - fee) / current_price
                cash = 0
                num_trades += 1
                total_fees_paid += fee

            elif action == Action.SELL and position > 0:
                # Sell entire position
                trade_value = position * current_price
                fee = trade_value * self.TAKER_FEE_RATE
                cash = trade_value - fee
                position = 0
                num_trades += 1
                total_fees_paid += fee

            # Calculate current portfolio value
            portfolio_value = cash + (position * current_price)

            # SURVIVAL CHECK: If portfolio value hits zero or below, agent dies
            if portfolio_value <= 0:
                survived = False
                log = f"Agent died on candle {i+1}/{len(market_data)}. Portfolio value: ${portfolio_value:.2f}. Balance went to zero."
                return -initial_capital, num_trades, survived, log

        # At the end, liquidate any remaining position
        if position > 0:
            final_position_value = position * market_data.iloc[-1]['close']
            final_fee = final_position_value * self.TAKER_FEE_RATE
            cash = final_position_value - final_fee
            position = 0
            total_fees_paid += final_fee

        final_capital = cash
        trading_profit = final_capital - initial_capital

        log = (
            f"Backtest complete. Initial: ${initial_capital:.2f}, "
            f"Final: ${final_capital:.2f}, Profit: ${trading_profit:.2f}, "
            f"Fees: ${total_fees_paid:.2f} ({num_trades} trades), "
            f"Traded on {len(market_data) - min_start_index} candles (skipped first {min_start_index} for lookback)"
        )

        return trading_profit, num_trades, survived, log

    def _get_min_required_history(self, program: list) -> int:
        """
        Calculate the minimum number of historical candles required before we can start trading.

        This is the maximum lookback parameter used in any rule.

        Args:
            program: Parsed DSL program

        Returns:
            Minimum start index (e.g., if max param is 50, return 50)
        """
        from ...dsl.grammar import CompoundCondition, Condition, FunctionCall, IndicatorValue

        def get_lookback_from_condition(cond):
            """Recursively extract max lookback from a condition (handles simple and compound)."""
            if isinstance(cond, CompoundCondition):
                # Recursively get lookback from left and right subconditions
                left_lookback = get_lookback_from_condition(cond.left)
                right_lookback = get_lookback_from_condition(cond.right)
                return max(left_lookback, right_lookback)
            elif isinstance(cond, Condition):
                # Simple condition - extract from left and right expressions
                max_val = 0
                for expr in [cond.left, cond.right]:
                    if isinstance(expr, IndicatorValue):
                        max_val = max(max_val, expr.param or 0)
                    elif isinstance(expr, FunctionCall):
                        max_val = max(max_val, expr.window or 0)
                return max_val
            return 0

        max_lookback = 0
        for rule in program:
            rule_lookback = get_lookback_from_condition(rule.condition)
            max_lookback = max(max_lookback, rule_lookback)

        # Return at least 1 to avoid edge cases
        return max(1, max_lookback)

    def run_multi_timeframe_backtest(
        self,
        program: list,
        multi_tf_data: Dict[str, pd.DataFrame],
        symbol: str,
    ) -> Tuple[float, Dict[str, CellPhenotype], str]:
        """
        Run backtest across multiple timeframes and return best fitness.

        This is the key method for cell-based evolution. It tests a strategy
        on 1H, 4H, and 1D data and selects the best performing timeframe.

        Args:
            program: Parsed DSL program
            multi_tf_data: Dictionary mapping timeframe to DataFrame
                          Example: {'1h': df_1h, '4h': df_4h, '1d': df_1d}
            symbol: Trading symbol (e.g., 'PURR')

        Returns:
            tuple of:
            - best_fitness: Best fitness across all timeframes
            - phenotypes: Dict mapping timeframe to CellPhenotype
            - log: Human-readable summary

        Example:
            data = fetcher.fetch_multi_timeframe('PURR')
            fitness, phenotypes, log = benchmark.run_multi_timeframe_backtest(
                program, data, 'PURR'
            )
            # fitness is the best result across 1H, 4H, 1D
            # phenotypes contains detailed metrics for each timeframe
        """
        phenotypes = {}
        best_fitness = float('-inf')
        best_timeframe = None

        for timeframe, market_data in multi_tf_data.items():
            # Run backtest on this timeframe
            trading_profit, num_trades, survived, backtest_log = self._run_backtest(
                program, market_data
            )

            # Calculate fitness (profit minus LLM cost)
            # Note: We only subtract LLM cost once, not per timeframe
            # The LLM cost is amortized across all timeframe tests
            estimated_llm_cost = 0.0165  # Rough estimate
            fitness = trading_profit  # Don't subtract LLM cost yet - do it at cell level

            # Calculate detailed metrics for phenotype
            phenotype = self._calculate_phenotype(
                program=program,
                market_data=market_data,
                symbol=symbol,
                timeframe=timeframe,
                trading_profit=trading_profit,
                num_trades=num_trades,
                survived=survived,
            )

            phenotypes[timeframe] = phenotype

            # Track best timeframe
            if fitness > best_fitness:
                best_fitness = fitness
                best_timeframe = timeframe

        # Create summary log
        log = f"Multi-timeframe backtest results:\n"
        for tf, phenotype in phenotypes.items():
            marker = "✓" if tf == best_timeframe else " "
            win_rate_str = f"{phenotype.win_rate:.1%}" if phenotype.win_rate is not None else "N/A"
            log += f"{marker} {tf}: ${phenotype.total_profit:.2f} profit, {phenotype.total_trades} trades, {win_rate_str} win rate\n"
        log += f"Best timeframe: {best_timeframe} with ${best_fitness:.2f} profit"

        return best_fitness, phenotypes, log

    def _calculate_phenotype(
        self,
        program: list,
        market_data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        trading_profit: float,
        num_trades: int,
        survived: bool,
    ) -> CellPhenotype:
        """
        Calculate detailed phenotype metrics for a backtest.

        This creates the CellPhenotype object that stores market behavior data.

        Args:
            program: Parsed DSL program
            market_data: Market data DataFrame
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., '1h')
            trading_profit: Total profit from backtest
            num_trades: Number of trades executed
            survived: Whether the strategy survived

        Returns:
            CellPhenotype object (phenotype_id and cell_id will be 0 initially)
        """
        # Run a detailed backtest to collect trade statistics
        initial_capital = self.INITIAL_CAPITAL
        cash = initial_capital
        position = 0.0
        trades = []  # Store individual trade results
        total_fees_paid = 0.0
        profitable_trades = 0
        losing_trades = 0

        min_start_index = self._get_min_required_history(program)
        portfolio_values = []

        for i in range(min_start_index, len(market_data)):
            action = self.interpreter.execute(program, market_data, current_index=i)
            current_price = market_data.iloc[i]['close']

            # Track trade entry/exit
            if action == Action.BUY and cash > 0:
                entry_price = current_price
                trade_value = cash
                fee = trade_value * self.TAKER_FEE_RATE
                position = (cash - fee) / current_price
                cash = 0
                total_fees_paid += fee
                trades.append({'type': 'BUY', 'price': entry_price, 'index': i})

            elif action == Action.SELL and position > 0:
                exit_price = current_price
                trade_value = position * current_price
                fee = trade_value * self.TAKER_FEE_RATE
                cash = trade_value - fee
                position = 0
                total_fees_paid += fee

                # Calculate profit/loss for this trade
                if trades and trades[-1]['type'] == 'BUY':
                    entry = trades[-1]
                    trade_profit = (exit_price - entry['price']) * (trade_value / exit_price)
                    if trade_profit > 0:
                        profitable_trades += 1
                    else:
                        losing_trades += 1

                trades.append({'type': 'SELL', 'price': exit_price, 'index': i})

            # Track portfolio value for drawdown calculation
            portfolio_value = cash + (position * current_price)
            portfolio_values.append(portfolio_value)

        # Calculate max drawdown and max runup
        max_drawdown = 0.0
        max_runup = 0.0
        peak_value = initial_capital
        trough_value = initial_capital

        for value in portfolio_values:
            if value > peak_value:
                # New peak - calculate drawdown from previous peak
                if peak_value > 0:
                    drawdown = (peak_value - trough_value) / peak_value
                    max_drawdown = max(max_drawdown, drawdown)
                peak_value = value
                trough_value = value
            elif value < trough_value:
                trough_value = value

            if value < trough_value:
                # New trough - calculate runup from previous trough
                if trough_value > 0:
                    runup = (peak_value - trough_value) / trough_value
                    max_runup = max(max_runup, runup)

        # Calculate win rate
        total_closed_trades = profitable_trades + losing_trades
        win_rate = profitable_trades / total_closed_trades if total_closed_trades > 0 else 0.0

        # Calculate profit factor (gross profit / gross loss)
        profit_factor = None  # Would need to track gross profits and losses separately

        # Calculate average trade metrics
        avg_profit_per_trade = trading_profit / total_closed_trades if total_closed_trades > 0 else 0.0

        # Data range
        data_start_date = str(market_data.iloc[0]['timestamp']) if 'timestamp' in market_data.columns else None
        data_end_date = str(market_data.iloc[-1]['timestamp']) if 'timestamp' in market_data.columns else None

        return CellPhenotype(
            phenotype_id=0,  # Will be set by repository
            cell_id=0,  # Will be set by repository
            symbol=symbol,
            timeframe=timeframe,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            total_trades=num_trades,
            profitable_trades=profitable_trades,
            losing_trades=losing_trades,
            total_profit=trading_profit,
            total_fees=total_fees_paid,
            max_drawdown=max_drawdown if max_drawdown > 0 else None,
            max_runup=max_runup if max_runup > 0 else None,
            sharpe_ratio=None,  # Would need returns series to calculate
            sortino_ratio=None,  # Would need returns series to calculate
            win_rate=win_rate if total_closed_trades > 0 else None,
            profit_factor=profit_factor,
            avg_trade_duration_hours=None,  # Would need to track trade durations
            avg_profit_per_trade=avg_profit_per_trade if total_closed_trades > 0 else None,
            avg_loss_per_trade=None,  # Would need to track separately
            longest_winning_streak=None,  # Would need to track streaks
            longest_losing_streak=None,  # Would need to track streaks
            trigger_conditions=None,  # Could store JSON of trigger analysis
        )

    def update_state(self, trading_profit: float, llm_cost: float, survived: bool):
        """
        Update the agent's financial state after an attempt.
        
        Args:
            trading_profit: Profit/loss from the trading backtest
            llm_cost: Cost of the LLM call for this attempt
            survived: Whether the strategy survived (fitness > 0)
        """
        self.attempts += 1
        self.total_llm_costs += llm_cost
        self.total_trading_profit += trading_profit
        
        # Update balance: add profit, subtract costs
        fitness = trading_profit - llm_cost
        self.agent_balance += fitness
        
        return fitness
    
    def get_state(self) -> dict:
        """Get the current financial state of the agent."""
        return {
            'balance': self.agent_balance,
            'total_llm_costs': self.total_llm_costs,
            'total_trading_profit': self.total_trading_profit,
            'attempts': self.attempts,
            'initial_capital': self.INITIAL_CAPITAL,
            'transaction_cost': self.TRANSACTION_COST,
            'net_result': self.agent_balance - self.INITIAL_CAPITAL,
        }
    
    def reset_state(self):
        """Reset the agent's financial state to initial conditions."""
        self.agent_balance = self.INITIAL_CAPITAL
        self.total_llm_costs = 0.0
        self.total_trading_profit = 0.0
        self.attempts = 0
    
    def generate_mutated_strategy(self, base_strategy_path: Path, output_path: Path) -> bool:
        """
        Generate a mutated strategy from a base strategy.
        
        This is used for evolutionary improvement instead of LLM-based code modification.
        
        Args:
            base_strategy_path: Path to the file containing the base DSL strategy
            output_path: Path where the mutated strategy should be written
            
        Returns:
            True if mutation succeeded, False otherwise
        """
        try:
            # Read the base strategy
            if not base_strategy_path.exists():
                print(f"Base strategy not found: {base_strategy_path}")
                return False
            
            base_dsl = base_strategy_path.read_text().strip()
            print(f"Base strategy: {base_dsl}")
            
            # Parse it
            program = self.interpreter.parse(base_dsl)
            if not program:
                print(f"Failed to parse base strategy: {base_dsl}")
                return False
            
            # Mutate it
            mutated_program = self.mutator.mutate(program)
            mutated_dsl = self.mutator.to_string(mutated_program)
            print(f"Mutated strategy: {mutated_dsl}")
            
            # Write to output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(mutated_dsl + "\n")
            
            return True
            
        except Exception as e:
            print(f"Error during mutation: {e}")
            import traceback
            traceback.print_exc()
            return False
