"""
Cell Analysis Pipeline with Batch Processing

Analyzes trading strategy cells using LLM to discover patterns and build
a pattern taxonomy. Uses batch processing to fit within LLM context limits.
"""

import json
from typing import Optional
from dataclasses import asdict

from ..storage.cell_repository import CellRepository
from ..storage.models import Cell, CellPhenotype
from ..llm.llm_factory import analyze_cell_with_llm


async def prepare_cell_context(cell_id: int, repo: CellRepository) -> str:
    """
    Prepare context about a cell for LLM analysis.

    Fetches cell data, lineage, and phenotypes from database and formats
    them into a concise string for LLM processing.

    Args:
        cell_id: ID of the cell to analyze
        repo: Cell repository for database access

    Returns:
        Formatted string containing cell information
    """
    # Fetch cell
    cell = repo.get_cell(cell_id)
    if not cell:
        raise ValueError(f"Cell #{cell_id} not found")

    # Fetch phenotypes
    phenotypes = repo.get_phenotypes_for_cell(cell_id)

    # Build context string
    context_parts = []

    # Basic info
    context_parts.append(f"Cell #{cell.cell_id}")
    context_parts.append(f"Generation: {cell.generation}")
    context_parts.append(f"Fitness: ${cell.fitness:.2f}")
    context_parts.append(f"Strategy: {cell.dsl_genome}")

    # Parent info
    if cell.parent_cell_id:
        parent = repo.get_cell(cell.parent_cell_id)
        if parent:
            context_parts.append(f"Parent: Cell #{parent.cell_id} (${parent.fitness:.2f})")

    # Phenotype performance
    if phenotypes:
        context_parts.append("\nPerformance:")
        for pheno in phenotypes:
            win_rate_str = f"{pheno.win_rate*100:.0f}%" if pheno.win_rate is not None else "N/A"
            context_parts.append(
                f"- {pheno.timeframe}: ${pheno.total_profit:.2f} "
                f"({pheno.total_trades} trades, {win_rate_str} win rate)"
            )

    return "\n".join(context_parts)


async def analyze_cells_in_batches(
    repo: CellRepository,
    cell_ids: list[int],
    batch_size: int = 30,
    use_json: bool = True,
) -> dict:
    """
    Analyze multiple cells in batches to fit LLM context limits.

    Processes cells in batches of `batch_size` (default 30 for Gemma 3 27B's
    8K context window). Each batch is analyzed independently, then results
    are merged.

    Args:
        repo: Cell repository for database access
        cell_ids: List of cell IDs to analyze
        batch_size: Number of cells per batch (default 30 for 8K context)
        use_json: Whether to expect JSON response from LLM

    Returns:
        Dict containing:
            - patterns: List of discovered patterns
            - cell_analyses: Dict mapping cell_id to analysis
            - failed_batches: List of batch indices that failed
    """
    results = {
        "patterns": [],
        "cell_analyses": {},
        "failed_batches": [],
    }

    # Split into batches
    batches = [
        cell_ids[i:i + batch_size]
        for i in range(0, len(cell_ids), batch_size)
    ]

    print(f"\n🔬 Analyzing {len(cell_ids)} cells in {len(batches)} batches of {batch_size}...")

    for batch_idx, batch in enumerate(batches):
        print(f"\n  Batch {batch_idx + 1}/{len(batches)}: Cells {batch[0]}-{batch[-1]}...")

        try:
            # Prepare batch context
            batch_contexts = []
            for cell_id in batch:
                try:
                    context = await prepare_cell_context(cell_id, repo)
                    batch_contexts.append(context)
                except Exception as e:
                    print(f"    ⚠️  Skipping Cell #{cell_id}: {e}")
                    continue

            if not batch_contexts:
                print(f"    ❌ Batch {batch_idx + 1} empty, skipping")
                results["failed_batches"].append(batch_idx)
                continue

            # Build batch prompt
            batch_text = "\n\n---\n\n".join(batch_contexts)

            prompt = f"""Analyze these {len(batch_contexts)} trading strategy cells and identify patterns.

{batch_text}

For each distinct pattern you identify, provide:
1. pattern_name: Short descriptive name (e.g., "Momentum Crossover")
2. pattern_category: One of [momentum, mean_reversion, trend, breakout, volatility]
3. explanation: What the pattern does and why it might work
4. cells_using_pattern: List of cell IDs that exhibit this pattern

Return JSON:
{{
  "patterns": [
    {{
      "pattern_name": "...",
      "pattern_category": "...",
      "explanation": "...",
      "cells_using_pattern": [cell_id1, cell_id2, ...]
    }},
    ...
  ]
}}"""

            # Call LLM
            response = await analyze_cell_with_llm(
                cell_context=prompt,
                system_prompt="You are an expert trading strategy analyst. Identify common patterns across multiple strategies.",
                use_json=use_json,
            )

            # Extract patterns from response
            if use_json and isinstance(response, dict):
                batch_patterns = response.get("patterns", [])
                results["patterns"].extend(batch_patterns)
                print(f"    ✓ Found {len(batch_patterns)} patterns in batch")

                # Store individual cell analyses (which patterns they use)
                for pattern in batch_patterns:
                    for cell_id in pattern.get("cells_using_pattern", []):
                        if cell_id not in results["cell_analyses"]:
                            results["cell_analyses"][cell_id] = []
                        results["cell_analyses"][cell_id].append({
                            "pattern_name": pattern["pattern_name"],
                            "pattern_category": pattern["pattern_category"],
                        })
            else:
                print(f"    ⚠️  Unexpected response format from LLM")
                results["failed_batches"].append(batch_idx)

        except Exception as e:
            print(f"    ❌ Batch {batch_idx + 1} failed: {e}")
            results["failed_batches"].append(batch_idx)

    print(f"\n✓ Batch analysis complete: {len(results['patterns'])} patterns found")
    if results["failed_batches"]:
        print(f"  ⚠️  {len(results['failed_batches'])} batches failed")

    return results


async def merge_pattern_discoveries(batch_results: dict) -> dict:
    """
    Merge and deduplicate patterns discovered across multiple batches.

    Patterns with similar names or descriptions are merged. This handles
    the case where the LLM identifies the same pattern in different batches
    with slightly different names.

    Args:
        batch_results: Results from analyze_cells_in_batches()

    Returns:
        Dict with deduplicated patterns:
            - patterns: List of unique patterns
            - pattern_cell_map: Dict mapping pattern_name to list of cell_ids
    """
    patterns = batch_results.get("patterns", [])

    if not patterns:
        return {"patterns": [], "pattern_cell_map": {}}

    # Group patterns by category and similar names
    pattern_groups = {}

    for pattern in patterns:
        name = pattern["pattern_name"].lower().strip()
        category = pattern["pattern_category"]
        key = f"{category}:{name}"

        if key not in pattern_groups:
            pattern_groups[key] = {
                "pattern_name": pattern["pattern_name"],
                "pattern_category": category,
                "explanation": pattern["explanation"],
                "cells": set(),
            }

        # Merge cells
        pattern_groups[key]["cells"].update(
            pattern.get("cells_using_pattern", [])
        )

        # Keep the longest explanation
        if len(pattern["explanation"]) > len(pattern_groups[key]["explanation"]):
            pattern_groups[key]["explanation"] = pattern["explanation"]

    # Convert back to list format
    merged_patterns = []
    pattern_cell_map = {}

    for key, group in pattern_groups.items():
        pattern_name = group["pattern_name"]
        merged_patterns.append({
            "pattern_name": pattern_name,
            "pattern_category": group["pattern_category"],
            "explanation": group["explanation"],
            "num_cells": len(group["cells"]),
        })
        pattern_cell_map[pattern_name] = list(group["cells"])

    print(f"\n🔗 Pattern deduplication:")
    print(f"   Before: {len(patterns)} patterns")
    print(f"   After:  {len(merged_patterns)} unique patterns")

    return {
        "patterns": merged_patterns,
        "pattern_cell_map": pattern_cell_map,
    }
