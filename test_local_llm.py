#!/usr/bin/env python3
"""Test script for local LLM setup with Ollama."""

import asyncio
import sys
import os

# Add base_agent to path
sys.path.insert(0, 'base_agent/src')

from llm.llm_factory import get_llm_client, analyze_cell_with_llm


async def test_basic_generation():
    """Test basic text generation."""
    print("\n" + "="*60)
    print("TEST 1: Basic Text Generation")
    print("="*60)

    client = get_llm_client()

    if client is None:
        print("❌ Local LLM not available, will use Anthropic API")
        print("Set USE_LOCAL_LLM=true in .env to use local LLM")
        return False

    print("✓ Local LLM client created")
    print(f"Model: {client.model}")
    print(f"Base URL: {client.base_url}")

    # Simple test prompt
    prompt = "Explain what a moving average is in trading in one sentence."

    print(f"\nPrompt: {prompt}")
    print("\nGenerating...")

    try:
        response = await client.generate(
            prompt=prompt,
            system="You are a helpful trading expert.",
            temperature=0.7,
            max_tokens=100,
        )

        print(f"\nResponse:\n{response}")
        print("\n✓ Basic generation works!")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def test_json_generation():
    """Test JSON generation (for cell analysis)."""
    print("\n" + "="*60)
    print("TEST 2: JSON Generation")
    print("="*60)

    client = get_llm_client()

    if client is None:
        print("Skipping (using Anthropic)")
        return True

    # Test JSON prompt
    prompt = """
Analyze this trading strategy: "IF DELTA(0) > DELTA(20) THEN BUY ELSE SELL"

Where DELTA is the close price and the parameter is lookback periods.

Return a JSON object with:
{
  "pattern_name": "short descriptive name",
  "pattern_category": "momentum|mean_reversion|trend|breakout",
  "explanation": "what the strategy does",
  "strength": "weak|moderate|strong"
}
"""

    print(f"\nPrompt: {prompt[:100]}...")
    print("\nGenerating JSON...")

    try:
        response = await client.generate_with_json(
            prompt=prompt,
            system="You are a trading strategy analyst. Respond with valid JSON only.",
            temperature=0.3,
        )

        print(f"\nJSON Response:")
        import json
        print(json.dumps(response, indent=2))

        # Verify structure
        required_keys = ["pattern_name", "pattern_category", "explanation", "strength"]
        for key in required_keys:
            if key not in response:
                print(f"❌ Missing key: {key}")
                return False

        print("\n✓ JSON generation works!")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_cell_analysis():
    """Test the analyze_cell_with_llm helper function."""
    print("\n" + "="*60)
    print("TEST 3: Cell Analysis Helper")
    print("="*60)

    cell_context = """
Analyze this trading cell:

Cell ID: #42
Generation: 15
Fitness: $27.81
Strategy: IF GAMMA(50) >= GAMMA(100) THEN BUY ELSE HOLD

Where:
- GAMMA = low price
- Parameter = lookback periods

Performance:
- 1H timeframe: $-7.50 (1 trade)
- 4H timeframe: $27.81 (1 trade) ← Best
- 1D timeframe: $0.00 (0 trades)

Respond with JSON:
{
  "pattern_name": "short name",
  "pattern_category": "momentum|mean_reversion|trend|breakout",
  "explanation": "what it does",
  "timeframe_preference": "which timeframe works best and why"
}
"""

    print("Analyzing cell...")

    try:
        response = await analyze_cell_with_llm(
            cell_context=cell_context,
            system_prompt="You are an expert trading strategy analyst.",
            use_json=True,
        )

        print(f"\nAnalysis Result:")
        import json
        print(json.dumps(response, indent=2))

        print("\n✓ Cell analysis works!")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LOCAL LLM SETUP TEST")
    print("="*60)
    print(f"\nUSE_LOCAL_LLM: {os.getenv('USE_LOCAL_LLM', 'not set')}")
    print(f"OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL', 'not set')}")
    print(f"OLLAMA_MODEL: {os.getenv('OLLAMA_MODEL', 'not set')}")

    results = []

    # Run tests
    results.append(("Basic Generation", await test_basic_generation()))
    results.append(("JSON Generation", await test_json_generation()))
    results.append(("Cell Analysis", await test_cell_analysis()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 All tests passed! Local LLM is ready for Sprint 3.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

    return all_passed


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
