#!/usr/bin/env python3
"""
Fetch PURR perpetual data from Hyperliquid and save for training.

This script fetches 60 days of hourly PURR data and saves it to benchmark_data/trading/
"""

import sys
from pathlib import Path

# Add base_agent to path
sys.path.insert(0, str(Path(__file__).parent / "base_agent"))

from base_agent.src.data.hyperliquid_fetcher import fetch_purr_data


def main():
    print("=" * 70)
    print("FETCHING PURR DATA FROM HYPERLIQUID")
    print("=" * 70)
    print()

    # Fetch 60 days of hourly data
    df = fetch_purr_data(
        lookback_days=60,
        interval="1h",
        output_path=Path("benchmark_data/trading/purr_60d.csv")
    )

    print()
    print("=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Total candles: {len(df)}")
    print(f"Date range: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
    print()
    print("Columns:", list(df.columns))
    print()
    print("First 5 rows:")
    print(df.head())
    print()
    print("Last 5 rows:")
    print(df.tail())
    print()
    print("Data statistics:")
    print(df.describe())
    print()
    print("=" * 70)
    print("✓ PURR data fetched successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
