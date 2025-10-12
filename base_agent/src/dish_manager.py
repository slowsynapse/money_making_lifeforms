"""
Dish Manager - Manages multiple "petri dish" experiments

Each dish represents a named experiment with its own cell culture (database).
Dishes can be resumed and extended over multiple runs.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class DishManager:
    """
    Manages experiment dishes (named evolutionary runs).

    Directory structure:
        experiments/
        ├── baseline_purr/
        │   ├── dish_config.json
        │   ├── cells.db
        │   └── runs/
        │       └── run_001_gen0-100/
        └── dsl_v2_test/
            └── ...
    """

    def __init__(self, experiments_dir: Path):
        """
        Initialize dish manager.

        Args:
            experiments_dir: Root directory for all experiments (e.g., experiments/)
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def create_dish(
        self,
        dish_name: str,
        symbol: str = "PURR",
        initial_capital: float = 1000.0,
        mutation_rate: str = "standard",
        dsl_version: str = "v2_phase1",
        description: str = ""
    ) -> Path:
        """
        Create a new experiment dish.

        Args:
            dish_name: Name of the experiment (e.g., 'baseline_purr')
            symbol: Trading symbol
            initial_capital: Starting capital
            mutation_rate: Mutation strategy
            dsl_version: DSL version to use
            description: Human-readable description

        Returns:
            Path to the dish directory

        Raises:
            ValueError: If dish already exists
        """
        dish_path = self.experiments_dir / dish_name

        if dish_path.exists():
            raise ValueError(
                f"Dish '{dish_name}' already exists at {dish_path}. "
                "Use --resume to continue this dish."
            )

        # Create dish directory structure
        dish_path.mkdir(parents=True, exist_ok=True)
        (dish_path / "runs").mkdir(exist_ok=True)

        # Create dish config
        config = {
            "dish_name": dish_name,
            "created_at": datetime.now().isoformat(),
            "symbol": symbol,
            "initial_capital": initial_capital,
            "mutation_rate": mutation_rate,
            "dsl_version": dsl_version,
            "description": description,
            "total_generations": 0,
            "total_cells": 0,
            "total_runs": 0,
            "best_fitness": None,
            "best_cell_name": None,
            "last_updated": datetime.now().isoformat()
        }

        config_path = dish_path / "dish_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Created new dish: {dish_name}")
        print(f"  Location: {dish_path}")

        return dish_path

    def load_dish(self, dish_name: str) -> tuple[Path, dict]:
        """
        Load an existing dish.

        Args:
            dish_name: Name of the dish to load

        Returns:
            Tuple of (dish_path, config_dict)

        Raises:
            FileNotFoundError: If dish doesn't exist
        """
        dish_path = self.experiments_dir / dish_name

        if not dish_path.exists():
            raise FileNotFoundError(
                f"Dish '{dish_name}' not found at {dish_path}. "
                f"Available dishes: {self.list_dish_names()}"
            )

        config_path = dish_path / "dish_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Dish config not found at {config_path}. "
                "This dish may be corrupted."
            )

        with open(config_path, 'r') as f:
            config = json.load(f)

        print(f"✓ Loaded dish: {dish_name}")
        print(f"  Created: {config['created_at']}")
        print(f"  Generations: {config['total_generations']}")
        print(f"  Cells: {config['total_cells']}")
        if config['best_fitness']:
            print(f"  Best Fitness: ${config['best_fitness']:.2f} ({config['best_cell_name']})")

        return dish_path, config

    def update_dish_config(
        self,
        dish_name: str,
        total_generations: Optional[int] = None,
        total_cells: Optional[int] = None,
        best_fitness: Optional[float] = None,
        best_cell_name: Optional[str] = None
    ):
        """
        Update dish configuration after a run.

        Args:
            dish_name: Name of the dish
            total_generations: New total generation count
            total_cells: New total cell count
            best_fitness: New best fitness (if improved)
            best_cell_name: New best cell name (if improved)
        """
        dish_path = self.experiments_dir / dish_name
        config_path = dish_path / "dish_config.json"

        with open(config_path, 'r') as f:
            config = json.load(f)

        if total_generations is not None:
            config['total_generations'] = total_generations
        if total_cells is not None:
            config['total_cells'] = total_cells
        if best_fitness is not None:
            config['best_fitness'] = best_fitness
        if best_cell_name is not None:
            config['best_cell_name'] = best_cell_name

        config['total_runs'] = config.get('total_runs', 0) + 1
        config['last_updated'] = datetime.now().isoformat()

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def list_dishes(self) -> list[dict]:
        """
        List all experiment dishes with their summaries.

        Returns:
            List of dish summary dicts
        """
        dishes = []

        if not self.experiments_dir.exists():
            return dishes

        for dish_dir in sorted(self.experiments_dir.iterdir()):
            if not dish_dir.is_dir():
                continue

            config_path = dish_dir / "dish_config.json"
            if not config_path.exists():
                continue

            with open(config_path, 'r') as f:
                config = json.load(f)

            dishes.append({
                'dish_name': config['dish_name'],
                'created_at': config['created_at'],
                'total_generations': config['total_generations'],
                'total_cells': config['total_cells'],
                'total_runs': config.get('total_runs', 0),
                'best_fitness': config.get('best_fitness'),
                'best_cell_name': config.get('best_cell_name'),
                'description': config.get('description', ''),
                'path': str(dish_dir)
            })

        return dishes

    def list_dish_names(self) -> list[str]:
        """Get list of all dish names."""
        return [d['dish_name'] for d in self.list_dishes()]

    def get_next_run_number(self, dish_name: str) -> int:
        """
        Get the next run number for a dish.

        Args:
            dish_name: Name of the dish

        Returns:
            Next run number (1-indexed)
        """
        dish_path = self.experiments_dir / dish_name
        runs_dir = dish_path / "runs"

        if not runs_dir.exists():
            return 1

        existing_runs = list(runs_dir.iterdir())
        if not existing_runs:
            return 1

        # Extract run numbers from run directories
        run_numbers = []
        for run_dir in existing_runs:
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                try:
                    num_str = run_dir.name.split("_")[1]
                    run_numbers.append(int(num_str))
                except (IndexError, ValueError):
                    continue

        if not run_numbers:
            return 1

        return max(run_numbers) + 1

    def create_run_directory(
        self,
        dish_name: str,
        start_gen: int,
        end_gen: int
    ) -> Path:
        """
        Create a directory for this evolution run.

        Args:
            dish_name: Name of the dish
            start_gen: Starting generation
            end_gen: Ending generation

        Returns:
            Path to the run directory
        """
        run_number = self.get_next_run_number(dish_name)
        run_name = f"run_{run_number:03d}_gen{start_gen}-{end_gen}"

        dish_path = self.experiments_dir / dish_name
        run_path = dish_path / "runs" / run_name
        run_path.mkdir(parents=True, exist_ok=True)

        # Create generations subdirectory
        (run_path / "generations").mkdir(exist_ok=True)

        return run_path
