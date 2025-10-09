"""
LLM-based cell analysis for pattern discovery and intelligent mutation.

This module provides batch processing capabilities for analyzing large numbers
of trading strategy cells using local or cloud LLMs.
"""

from .cell_analyzer import (
    prepare_cell_context,
    analyze_cells_in_batches,
    merge_pattern_discoveries,
)
from .mutation_proposer import (
    propose_intelligent_mutation,
    batch_propose_mutations,
)

__all__ = [
    "prepare_cell_context",
    "analyze_cells_in_batches",
    "merge_pattern_discoveries",
    "propose_intelligent_mutation",
    "batch_propose_mutations",
]
