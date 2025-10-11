#!/usr/bin/env python3
"""
Verification script for trading module refactoring.
Checks that all expected functions exist and are properly exported.
"""

import ast
import sys
from pathlib import Path

def check_file_exports(filepath, expected_names):
    """Check if a Python file defines the expected names."""
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())

    defined_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined_names.add(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            defined_names.add(node.name)

    return expected_names.issubset(defined_names)

def check_init_exports(init_file, expected_names):
    """Check if __init__.py exports the expected names."""
    with open(init_file, 'r') as f:
        content = f.read()

    # Check if all expected names appear in the file
    for name in expected_names:
        if name not in content:
            return False
    return True

def main():
    base_path = Path(__file__).parent / "base_agent"

    print("🔍 Verifying trading module refactoring...")
    print()

    # Expected trading functions
    trading_functions = {
        'run_trading_demo',
        'run_trading_test',
        'run_trading_evolve',
        'run_trading_learn',
    }

    # Check 1: trading_evolution.py defines all functions
    evolution_file = base_path / "src/trading/trading_evolution.py"
    print(f"✓ Checking {evolution_file.relative_to(base_path.parent)}")
    if not evolution_file.exists():
        print(f"  ✗ File does not exist!")
        return 1

    if not check_file_exports(evolution_file, trading_functions):
        print(f"  ✗ Missing expected functions!")
        return 1
    print(f"  ✓ All {len(trading_functions)} trading functions defined")

    # Check 2: __init__.py exports all functions
    init_file = base_path / "src/trading/__init__.py"
    print(f"✓ Checking {init_file.relative_to(base_path.parent)}")
    if not init_file.exists():
        print(f"  ✗ File does not exist!")
        return 1

    if not check_init_exports(init_file, trading_functions):
        print(f"  ✗ Not all functions are exported!")
        return 1
    print(f"  ✓ All {len(trading_functions)} functions exported")

    # Check 3: agent.py imports all functions
    agent_file = base_path / "agent.py"
    print(f"✓ Checking {agent_file.relative_to(base_path.parent)}")
    if not agent_file.exists():
        print(f"  ✗ File does not exist!")
        return 1

    with open(agent_file, 'r') as f:
        agent_content = f.read()

    if "from .src.trading import" not in agent_content:
        print(f"  ✗ Missing trading import statement!")
        return 1

    # Check all functions are imported
    for func in trading_functions:
        if func not in agent_content:
            print(f"  ✗ Function '{func}' not found in agent.py!")
            return 1
    print(f"  ✓ All {len(trading_functions)} functions imported and used")

    # Check 4: Verify file sizes
    print()
    print("📊 File size verification:")
    agent_lines = len(open(agent_file).readlines())
    evolution_lines = len(open(evolution_file).readlines())
    print(f"  • agent.py: {agent_lines} lines")
    print(f"  • trading_evolution.py: {evolution_lines} lines")

    if agent_lines > 1000:
        print(f"  ⚠ agent.py is still quite large ({agent_lines} lines)")
    else:
        print(f"  ✓ agent.py is reasonably sized")

    print()
    print("✅ All verification checks passed!")
    print("   The trading module refactoring is structurally sound.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
