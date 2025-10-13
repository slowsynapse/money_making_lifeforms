"""
Intelligent Mutation Proposal using LLM

Proposes strategic mutations to trading cells based on discovered patterns
and performance insights.
"""

import json
from typing import Optional, Dict, Any

from ..storage.cell_repository import CellRepository
from ..storage.models import Cell
from ..llm.llm_factory import analyze_cell_with_llm
from ..dsl.interpreter import DslInterpreter


async def propose_intelligent_mutation(
    cell: Cell,
    patterns: dict,
    repo: CellRepository,
    use_json: bool = True,
    temperature: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to propose an intelligent mutation to a cell.

    The LLM analyzes the cell's strategy, fitness, patterns it uses, and
    performance across timeframes to suggest a targeted mutation that is
    likely to improve performance.

    Args:
        cell: The cell to mutate
        patterns: Pattern taxonomy (from merge_pattern_discoveries)
        repo: Cell repository for database access
        use_json: Whether to expect JSON response

    Returns:
        Dict containing:
            - proposed_strategy: New DSL strategy string
            - rationale: Why this mutation might work
            - expected_improvement: What the LLM expects to improve
            - confidence: "low" | "medium" | "high"

        Returns None if proposal fails or is invalid
    """
    # Fetch cell phenotypes for performance analysis
    phenotypes = repo.get_phenotypes_for_cell(cell.cell_id)

    # Calculate total trades for trading activity guidance
    total_trades = sum(p.total_trades for p in phenotypes) if phenotypes else 0

    # Find which patterns this cell uses
    cell_patterns = []
    pattern_cell_map = patterns.get("pattern_cell_map", {})
    for pattern_name, cell_ids in pattern_cell_map.items():
        if cell.cell_id in cell_ids:
            # Find the pattern details
            for p in patterns.get("patterns", []):
                if p["pattern_name"] == pattern_name:
                    cell_patterns.append(p)
                    break

    # Build context for LLM
    context_parts = []
    context_parts.append(f"Cell #{cell.cell_id} (Generation {cell.generation})")
    context_parts.append(f"Current Strategy: {cell.dsl_genome}")
    context_parts.append(f"Current Fitness: ${cell.fitness:.2f}")

    # Add performance breakdown
    if phenotypes:
        context_parts.append("\nPerformance by Timeframe:")
        for pheno in phenotypes:
            win_rate_str = f"{pheno.win_rate*100:.0f}%" if pheno.win_rate is not None else "N/A"
            context_parts.append(
                f"- {pheno.timeframe}: ${pheno.total_profit:.2f} "
                f"({pheno.total_trades} trades, {win_rate_str} win rate)"
            )
            if pheno.sharpe_ratio:
                context_parts.append(f"  Sharpe: {pheno.sharpe_ratio:.2f}")

    # Add parent comparison
    if cell.parent_cell_id:
        parent = repo.get_cell(cell.parent_cell_id)
        if parent:
            improvement = cell.fitness - parent.fitness
            context_parts.append(f"\nImprovement over parent: ${improvement:+.2f}")
            context_parts.append(f"Parent strategy: {parent.dsl_genome}")

    # Add patterns
    if cell_patterns:
        context_parts.append("\nPatterns Used:")
        for pattern in cell_patterns:
            context_parts.append(
                f"- {pattern['pattern_name']} ({pattern['pattern_category']}): "
                f"{pattern['explanation']}"
            )

    # Add top performing cells for inspiration
    top_cells = repo.get_top_cells(limit=5, status='online')
    if top_cells:
        context_parts.append("\nTop 5 Performing Strategies:")
        for top_cell in top_cells:
            if top_cell.cell_id != cell.cell_id:
                context_parts.append(
                    f"- Cell #{top_cell.cell_id}: ${top_cell.fitness:.2f} - {top_cell.dsl_genome}"
                )

    context = "\n".join(context_parts)

    # Build prompt for mutation proposal
    prompt = f"""{context}

Based on this analysis, propose ONE intelligent mutation to improve this strategy.

DSL Syntax Available:
- Indicators: ALPHA (open), BETA (high), GAMMA (low), DELTA (close), EPSILON (volume)
- Parameters: Lookback periods (e.g., DELTA(20) = close price 20 periods ago)
- Operators: >, <, >=, <=, ==, !=
- Arithmetic: +, -, *, / (e.g., DELTA(0) - DELTA(20))
- Logic: AND, OR, NOT (e.g., IF A > B AND C < D THEN BUY ELSE SELL)
- Parentheses: Group conditions (e.g., IF (A > B OR C < D) AND E > F THEN BUY ELSE SELL)
- Actions: BUY, SELL, HOLD

IMPORTANT: Multi-rule strategies use NEWLINES to separate rules. Do NOT use "ELSE IF" - it's not valid syntax.

CRITICAL Trading Activity Rules:
- Parent strategy generated {total_trades} trades across all timeframes
- Mutations MUST generate trades to be tested effectively
- Avoid strategies where ALL conditions lead to HOLD - this kills trading
- If you add HOLD actions, ensure at least one branch actively BUYs or SELLs
- Zero-trade strategies get penalized $-45 for inactivity
- Aim to maintain or improve trading activity level

Example strategies:
- "IF DELTA(0) > DELTA(20) THEN BUY ELSE SELL"  (simple momentum)
- "IF GAMMA(50) * 1.02 < DELTA(0) THEN BUY ELSE HOLD"  (support breakout)
- "IF DELTA(0) > DELTA(20) AND EPSILON() > 1000 THEN BUY ELSE SELL"  (momentum with volume)
- "IF DELTA(0) < GAMMA(50) OR DELTA(0) > BETA(50) THEN SELL ELSE BUY"  (reversal at extremes)
- "IF NOT DELTA() < DELTA(20) THEN BUY ELSE HOLD"  (negation for clarity)

Multi-rule examples (newline-separated):
- "IF DELTA(0) > DELTA(20) THEN BUY ELSE HOLD\\nIF EPSILON() > 1000 THEN SELL ELSE HOLD"  (two rules)
- "IF GAMMA(14) >= BETA(20) THEN BUY ELSE HOLD\\nIF DELTA(0) < GAMMA(50) THEN SELL ELSE HOLD"  (momentum + support)

Propose a mutation that:
1. Is syntactically valid DSL (use \\n for multiple rules, NOT "ELSE IF")
2. Is DIFFERENT from the current strategy
3. Addresses a weakness or exploits a strength
4. Has a clear rationale

Return JSON:
{{
  "proposed_strategy": "IF ... THEN ... ELSE ...",
  "rationale": "why this mutation might work",
  "expected_improvement": "what specific aspect should improve",
  "confidence": "low|medium|high"
}}"""

    try:
        # Call LLM with temperature control for diversity
        response = await analyze_cell_with_llm(
            cell_context=prompt,
            system_prompt="You are an expert trading strategist. Propose intelligent, targeted mutations to trading strategies.",
            use_json=use_json,
            temperature=temperature,
        )

        if not use_json or not isinstance(response, dict):
            print(f"  ⚠️  Unexpected response format from LLM")
            return None

        # Validate proposed strategy
        proposed_strategy = response.get("proposed_strategy", "").strip()

        if not proposed_strategy:
            print(f"  ❌ No strategy proposed")
            return None

        # Check if it's different from current
        if proposed_strategy == cell.dsl_genome:
            print(f"  ❌ Proposed strategy is identical to current")
            return None

        # Validate DSL syntax
        interpreter = DslInterpreter()
        try:
            program = interpreter.parse(proposed_strategy)
            if not program:
                print(f"  ❌ Proposed strategy failed to parse: {proposed_strategy}")
                return None
        except Exception as e:
            print(f"  ❌ Invalid DSL syntax: {e}")
            return None

        # Looks good!
        print(f"  ✓ Valid mutation proposed: {proposed_strategy}")
        print(f"    Rationale: {response.get('rationale', 'N/A')[:80]}...")
        print(f"    Confidence: {response.get('confidence', 'unknown')}")

        return response

    except Exception as e:
        print(f"  ❌ Mutation proposal failed: {e}")
        return None


async def batch_propose_mutations(
    cells: list[Cell],
    patterns: dict,
    repo: CellRepository,
    max_proposals: int = 10,
) -> list[Dict[str, Any]]:
    """
    Propose mutations for multiple cells and return the best proposals.

    Args:
        cells: List of cells to propose mutations for
        patterns: Pattern taxonomy
        repo: Cell repository
        max_proposals: Maximum number of successful proposals to return

    Returns:
        List of mutation proposals (up to max_proposals)
    """
    proposals = []

    print(f"\n🧬 Generating mutation proposals for {len(cells)} cells...")

    for i, cell in enumerate(cells):
        if len(proposals) >= max_proposals:
            break

        # Vary temperature across batch for diversity
        # Start conservative (0.3), gradually increase creativity (0.8)
        temperature = 0.3 + (i / max(len(cells) - 1, 1)) * 0.5

        print(f"\n  Cell {i+1}/{len(cells)}: #{cell.cell_id} (${cell.fitness:.2f}) [T={temperature:.2f}]")

        proposal = await propose_intelligent_mutation(cell, patterns, repo, temperature=temperature)

        if proposal:
            proposal["cell_id"] = cell.cell_id
            proposal["parent_fitness"] = cell.fitness
            proposal["temperature"] = temperature
            proposals.append(proposal)

    print(f"\n✓ Generated {len(proposals)} valid mutation proposals")
    return proposals
