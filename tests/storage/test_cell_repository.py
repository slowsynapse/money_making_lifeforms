"""
Unit tests for CellRepository.

Tests database operations, CRUD methods, and query functionality.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from agent_code.src.storage.cell_repository import CellRepository
from agent_code.src.storage.models import Cell, CellPhenotype, DiscoveredPattern, EvolutionRun


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def repo(temp_db):
    """Create a CellRepository instance with temporary database."""
    return CellRepository(temp_db)


class TestCellBirth:
    """Tests for cell creation."""

    def test_birth_seed_cell(self, repo):
        """Test birthing the first cell (generation 0, no parent)."""
        cell_id = repo.birth_cell(
            generation=0,
            parent_cell_id=None,
            dsl_genome="IF ALPHA(10) > BETA(20) THEN BUY ELSE SELL",
            fitness=5.23,
            status='online',
        )

        assert cell_id == 1  # First cell should have ID 1

        # Verify it was stored correctly
        cell = repo.get_cell(cell_id)
        assert cell is not None
        assert cell.cell_id == 1
        assert cell.generation == 0
        assert cell.parent_cell_id is None
        assert cell.fitness == 5.23
        assert cell.status == 'online'

    def test_birth_child_cell(self, repo):
        """Test birthing a cell with a parent."""
        # Create parent
        parent_id = repo.birth_cell(
            generation=0,
            parent_cell_id=None,
            dsl_genome="IF ALPHA(10) > BETA(20) THEN BUY ELSE SELL",
            fitness=5.23,
        )

        # Create child
        child_id = repo.birth_cell(
            generation=1,
            parent_cell_id=parent_id,
            dsl_genome="IF ALPHA(10) >= BETA(20) THEN BUY ELSE SELL",
            fitness=12.45,
        )

        assert child_id == 2

        child = repo.get_cell(child_id)
        assert child.parent_cell_id == parent_id
        assert child.generation == 1
        assert child.fitness == 12.45

    def test_birth_multiple_cells(self, repo):
        """Test creating multiple cells."""
        cell_ids = []
        for i in range(5):
            cell_id = repo.birth_cell(
                generation=i,
                parent_cell_id=cell_ids[-1] if cell_ids else None,
                dsl_genome=f"IF ALPHA({i}) > BETA({i+1}) THEN BUY ELSE SELL",
                fitness=float(i + 1),
            )
            cell_ids.append(cell_id)

        assert len(cell_ids) == 5
        assert cell_ids == [1, 2, 3, 4, 5]


class TestCellRetrieval:
    """Tests for retrieving cells."""

    def test_get_nonexistent_cell(self, repo):
        """Test getting a cell that doesn't exist."""
        cell = repo.get_cell(999)
        assert cell is None

    def test_get_top_cells_empty(self, repo):
        """Test getting top cells when database is empty."""
        top_cells = repo.get_top_cells(limit=10)
        assert len(top_cells) == 0

    def test_get_top_cells(self, repo):
        """Test getting top cells sorted by fitness."""
        # Create cells with different fitness values
        fitnesses = [5.23, 12.45, 3.14, 18.90, 7.65]
        for i, fitness in enumerate(fitnesses):
            repo.birth_cell(
                generation=i,
                parent_cell_id=None,
                dsl_genome=f"IF ALPHA({i}) > BETA({i}) THEN BUY ELSE SELL",
                fitness=fitness,
            )

        # Get top 3 cells
        top_cells = repo.get_top_cells(limit=3)
        assert len(top_cells) == 3
        assert top_cells[0].fitness == 18.90
        assert top_cells[1].fitness == 12.45
        assert top_cells[2].fitness == 7.65

    def test_get_top_cells_filter_by_status(self, repo):
        """Test filtering cells by status."""
        # Create online and deprecated cells
        repo.birth_cell(0, None, "GENOME1", 10.0, status='online')
        repo.birth_cell(1, None, "GENOME2", 20.0, status='online')
        repo.birth_cell(2, None, "GENOME3", 30.0, status='deprecated')

        # Should only get online cells
        online_cells = repo.get_top_cells(limit=10, status='online')
        assert len(online_cells) == 2
        assert all(c.status == 'online' for c in online_cells)


class TestLineage:
    """Tests for lineage tracking."""

    def test_get_lineage_seed(self, repo):
        """Test getting lineage of seed cell."""
        seed_id = repo.birth_cell(0, None, "SEED", 5.0)
        lineage = repo.get_lineage(seed_id)

        assert len(lineage) == 1
        assert lineage[0].cell_id == seed_id

    def test_get_lineage_multi_generation(self, repo):
        """Test getting lineage across multiple generations."""
        # Create lineage: Gen0 -> Gen1 -> Gen2 -> Gen3
        gen0_id = repo.birth_cell(0, None, "GEN0", 5.0)
        gen1_id = repo.birth_cell(1, gen0_id, "GEN1", 10.0)
        gen2_id = repo.birth_cell(2, gen1_id, "GEN2", 15.0)
        gen3_id = repo.birth_cell(3, gen2_id, "GEN3", 20.0)

        lineage = repo.get_lineage(gen3_id)

        assert len(lineage) == 4
        # Should be ordered from oldest to newest
        assert lineage[0].generation == 0
        assert lineage[1].generation == 1
        assert lineage[2].generation == 2
        assert lineage[3].generation == 3

    def test_get_lineage_nonexistent(self, repo):
        """Test getting lineage of nonexistent cell."""
        lineage = repo.get_lineage(999)
        assert len(lineage) == 0


class TestLLMAnalysis:
    """Tests for LLM analysis storage."""

    def test_update_llm_analysis(self, repo):
        """Test storing LLM analysis for a cell."""
        cell_id = repo.birth_cell(0, None, "GENOME", 10.0)

        # Initially, no LLM analysis
        cell = repo.get_cell(cell_id)
        assert cell.llm_name is None
        assert cell.llm_category is None

        # Add LLM analysis
        repo.update_cell_llm_analysis(
            cell_id=cell_id,
            llm_name="Volume Spike Reversal",
            llm_category="Volume Analysis",
            llm_hypothesis="Detects institutional accumulation during local dips",
            llm_confidence=0.85,
        )

        # Verify it was stored
        cell = repo.get_cell(cell_id)
        assert cell.llm_name == "Volume Spike Reversal"
        assert cell.llm_category == "Volume Analysis"
        assert cell.llm_hypothesis == "Detects institutional accumulation during local dips"
        assert cell.llm_confidence == 0.85
        assert cell.llm_analyzed_at is not None

    def test_find_unanalyzed_cells(self, repo):
        """Test finding cells that need LLM analysis."""
        # Create cells with and without analysis
        cell1_id = repo.birth_cell(0, None, "GENOME1", 10.0)
        cell2_id = repo.birth_cell(1, None, "GENOME2", 20.0)
        cell3_id = repo.birth_cell(2, None, "GENOME3", 30.0)

        # Analyze only cell2
        repo.update_cell_llm_analysis(
            cell2_id, "Pattern", "Category", "Hypothesis", 0.9
        )

        # Find unanalyzed cells
        unanalyzed = repo.find_unanalyzed_cells(limit=10, min_fitness=0.0)
        assert len(unanalyzed) == 2
        unanalyzed_ids = {c.cell_id for c in unanalyzed}
        assert cell1_id in unanalyzed_ids
        assert cell3_id in unanalyzed_ids
        assert cell2_id not in unanalyzed_ids

    def test_find_unanalyzed_cells_min_fitness(self, repo):
        """Test filtering unanalyzed cells by minimum fitness."""
        repo.birth_cell(0, None, "GENOME1", 5.0)
        repo.birth_cell(1, None, "GENOME2", 15.0)
        repo.birth_cell(2, None, "GENOME3", 25.0)

        # Only get cells with fitness >= 10
        unanalyzed = repo.find_unanalyzed_cells(limit=10, min_fitness=10.0)
        assert len(unanalyzed) == 2
        assert all(c.fitness >= 10.0 for c in unanalyzed)


class TestCellDeprecation:
    """Tests for cell lifecycle management."""

    def test_deprecate_cell(self, repo):
        """Test marking a cell as deprecated."""
        cell_id = repo.birth_cell(0, None, "OLD_GENOME", 10.0)

        # Deprecate the cell
        repo.deprecate_cell(
            cell_id=cell_id,
            reason="Superseded by better strategy",
            superseded_by_cell_id=None,
        )

        # Verify deprecation
        cell = repo.get_cell(cell_id)
        assert cell.status == 'deprecated'
        assert cell.deprecated_reason == "Superseded by better strategy"

    def test_deprecate_cell_with_successor(self, repo):
        """Test deprecating a cell and linking to its successor."""
        old_id = repo.birth_cell(0, None, "OLD_GENOME", 10.0)
        new_id = repo.birth_cell(1, old_id, "NEW_GENOME", 20.0)

        repo.deprecate_cell(
            cell_id=old_id,
            reason="Better version found",
            superseded_by_cell_id=new_id,
        )

        old_cell = repo.get_cell(old_id)
        assert old_cell.superseded_by_cell_id == new_id


class TestPhenotypes:
    """Tests for phenotype storage."""

    def test_store_phenotype(self, repo):
        """Test storing phenotype data."""
        cell_id = repo.birth_cell(0, None, "GENOME", 10.0)

        phenotype = CellPhenotype(
            phenotype_id=0,
            cell_id=cell_id,
            symbol='PURR',
            timeframe='1h',
            total_trades=45,
            profitable_trades=28,
            losing_trades=17,
            total_profit=23.45,
            total_fees=1.12,
            win_rate=0.622,
        )

        phenotype_id = repo.store_phenotype(phenotype)
        assert phenotype_id > 0

    def test_get_phenotypes(self, repo):
        """Test retrieving phenotypes for a cell."""
        cell_id = repo.birth_cell(0, None, "GENOME", 10.0)

        # Store multiple phenotypes (different timeframes)
        for tf in ['1h', '4h', '1d']:
            phenotype = CellPhenotype(
                phenotype_id=0,
                cell_id=cell_id,
                symbol='PURR',
                timeframe=tf,
                total_trades=10,
                profitable_trades=6,
                losing_trades=4,
                total_profit=5.0,
                total_fees=0.5,
            )
            repo.store_phenotype(phenotype)

        phenotypes = repo.get_phenotypes(cell_id)
        assert len(phenotypes) == 3
        timeframes = {p.timeframe for p in phenotypes}
        assert timeframes == {'1h', '4h', '1d'}


class TestFailedMutations:
    """Tests for failed mutation tracking."""

    def test_record_failed_mutation(self, repo):
        """Test recording a failed mutation."""
        parent_id = repo.birth_cell(0, None, "PARENT", 10.0)

        failure_id = repo.record_failed_mutation(
            parent_cell_id=parent_id,
            attempted_genome="BAD GENOME",
            generation=1,
            failure_reason='negative_fitness',
            fitness_achieved=-5.0,
        )

        assert failure_id > 0

    def test_get_failure_statistics(self, repo):
        """Test getting failure statistics."""
        parent_id = repo.birth_cell(0, None, "PARENT", 10.0)

        # Record multiple failures
        repo.record_failed_mutation(parent_id, "BAD1", 1, 'negative_fitness', -5.0)
        repo.record_failed_mutation(parent_id, "BAD2", 2, 'negative_fitness', -3.0)
        repo.record_failed_mutation(parent_id, "BAD3", 3, 'lower_than_parent', 8.0)

        stats = repo.get_failure_statistics(parent_id)
        assert stats['negative_fitness'] == 2
        assert stats['lower_than_parent'] == 1

    def test_get_failure_statistics_all(self, repo):
        """Test getting failure statistics across all cells."""
        parent1_id = repo.birth_cell(0, None, "PARENT1", 10.0)
        parent2_id = repo.birth_cell(0, None, "PARENT2", 15.0)

        repo.record_failed_mutation(parent1_id, "BAD1", 1, 'negative_fitness', -5.0)
        repo.record_failed_mutation(parent2_id, "BAD2", 1, 'execution_error', None)

        # Get stats for all cells
        stats = repo.get_failure_statistics()
        assert stats['negative_fitness'] == 1
        assert stats['execution_error'] == 1


class TestEvolutionRuns:
    """Tests for evolution run tracking."""

    def test_start_evolution_run(self, repo):
        """Test starting an evolution run."""
        run_id = repo.start_evolution_run(
            run_type='evolution',
            max_generations=50,
            fitness_goal=100.0,
            symbol='PURR',
            timeframe='1h',
            initial_capital=100.0,
            transaction_fee_rate=0.00045,
        )

        assert run_id > 0

        run = repo.get_evolution_run(run_id)
        assert run.run_type == 'evolution'
        assert run.max_generations == 50
        assert run.symbol == 'PURR'

    def test_complete_evolution_run(self, repo):
        """Test completing an evolution run."""
        # Start a run
        run_id = repo.start_evolution_run(
            run_type='evolution',
            max_generations=50,
            fitness_goal=100.0,
        )

        # Create a cell to reference
        best_cell_id = repo.birth_cell(0, None, "BEST", 50.0)

        # Complete the run
        repo.complete_evolution_run(
            run_id=run_id,
            best_cell_id=best_cell_id,
            total_cells_birthed=25,
            total_mutations_failed=75,
            final_best_fitness=50.0,
            termination_reason='max_generations_reached',
            generations_without_improvement=10,
        )

        run = repo.get_evolution_run(run_id)
        assert run.completed_at is not None
        assert run.best_cell_id == best_cell_id
        assert run.total_cells_birthed == 25
        assert run.total_mutations_failed == 75


class TestPatterns:
    """Tests for discovered pattern management."""

    def test_create_pattern(self, repo):
        """Test creating a discovered pattern."""
        cell_id = repo.birth_cell(0, None, "GENOME", 10.0)

        pattern_id = repo.create_pattern(
            pattern_name="Volume Spike Reversal",
            category="Volume Analysis",
            description="Detects institutional accumulation",
            discovered_by_cell_id=cell_id,
        )

        assert pattern_id > 0

    def test_link_cell_to_pattern(self, repo):
        """Test linking a cell to a pattern."""
        cell_id = repo.birth_cell(0, None, "GENOME", 10.0)
        pattern_id = repo.create_pattern(
            pattern_name="Test Pattern",
            category="Test",
            description="Test",
            discovered_by_cell_id=cell_id,
        )

        # Link cell to pattern
        repo.link_cell_to_pattern(cell_id, pattern_id, confidence=0.85)

        # Verify link
        cells = repo.get_cells_by_pattern(pattern_id)
        assert len(cells) == 1
        assert cells[0].cell_id == cell_id

    def test_get_patterns_by_category(self, repo):
        """Test retrieving patterns by category."""
        cell_id = repo.birth_cell(0, None, "GENOME", 10.0)

        # Create patterns in different categories
        repo.create_pattern("Pattern1", "Volume", "Desc1", cell_id)
        repo.create_pattern("Pattern2", "Volume", "Desc2", cell_id)
        repo.create_pattern("Pattern3", "Momentum", "Desc3", cell_id)

        volume_patterns = repo.get_patterns_by_category("Volume")
        assert len(volume_patterns) == 2

    def test_pattern_statistics_update(self, repo):
        """Test that pattern statistics update when cells are linked."""
        # Create cells with different fitness
        cell1_id = repo.birth_cell(0, None, "GENOME1", 10.0)
        cell2_id = repo.birth_cell(1, None, "GENOME2", 20.0)
        cell3_id = repo.birth_cell(2, None, "GENOME3", 15.0)

        # Create pattern
        pattern_id = repo.create_pattern(
            pattern_name="Test Pattern",
            category="Test",
            description="Test",
            discovered_by_cell_id=cell1_id,
        )

        # Link all cells to pattern
        repo.link_cell_to_pattern(cell1_id, pattern_id)
        repo.link_cell_to_pattern(cell2_id, pattern_id)
        repo.link_cell_to_pattern(cell3_id, pattern_id)

        # Get pattern and check statistics
        patterns = repo.get_patterns_by_category("Test")
        assert len(patterns) == 1
        pattern = patterns[0]

        assert pattern.cells_using_pattern == 3
        assert pattern.avg_fitness == 15.0  # (10 + 20 + 15) / 3
        assert pattern.best_fitness == 20.0
        assert pattern.worst_fitness == 10.0
        assert pattern.best_cell_id == cell2_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
