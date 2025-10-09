"""
Hyperliquid API data fetcher for OHLCV historical data.

Fetches candle data from Hyperliquid's perpetuals exchange API.
Supports multi-timeframe data fetching for evolution testing.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import time


class HyperliquidDataFetcher:
    """Fetches historical OHLCV data from Hyperliquid API."""

    API_URL = "https://api.hyperliquid.xyz/info"

    def __init__(self):
        self.session = requests.Session()
        self._cache: Dict[str, pd.DataFrame] = {}  # Cache for avoiding re-fetches

    def fetch_historical_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        lookback_days: int = 60,
        save_path: Path | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "PURR")
            interval: Candle interval - "1m", "5m", "15m", "1h", "4h", "1d"
            lookback_days: Number of days of historical data to fetch
            save_path: Optional path to save the CSV file

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume,
                                   trades, funding_rate, open_interest
        """
        print(f"Fetching {lookback_days} days of {interval} data for {symbol}...")

        # Calculate timestamps
        end_time = int(time.time() * 1000)  # Current time in milliseconds
        start_time = end_time - (lookback_days * 24 * 60 * 60 * 1000)

        # Prepare API request
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
            }
        }

        try:
            response = self.session.post(self.API_URL, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                raise ValueError(f"No data returned for {symbol}")

            # Parse the response into DataFrame
            df = self._parse_candles(data)

            print(f"✓ Fetched {len(df)} candles from {symbol}")
            print(f"  Date range: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")

            # Save to CSV if path provided
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path, index=False)
                print(f"✓ Saved to {save_path}")

            return df

        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching data from Hyperliquid: {e}")
            raise
        except (KeyError, ValueError) as e:
            print(f"❌ Error parsing Hyperliquid response: {e}")
            raise

    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str] = ['1h', '4h', '1d'],
        lookback_days: int = 30,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes at once.

        This is the primary method for evolution testing. It fetches data for
        1H, 4H, and 1D timeframes and aligns them to the same time period.

        Args:
            symbol: Trading pair symbol (e.g., "PURR")
            timeframes: List of timeframes to fetch (default: ['1h', '4h', '1d'])
            lookback_days: Number of days of data to fetch (default: 30)

        Returns:
            Dictionary mapping timeframe to DataFrame
            Example: {'1h': df_1h, '4h': df_4h, '1d': df_1d}

        Example:
            fetcher = HyperliquidDataFetcher()
            data = fetcher.fetch_multi_timeframe('PURR', lookback_days=30)

            # Test strategy on all timeframes
            for tf, df in data.items():
                fitness = backtest_strategy(strategy, df)
                print(f"{tf}: ${fitness:.2f}")
        """
        print(f"Fetching multi-timeframe data for {symbol} ({', '.join(timeframes)})...")

        result = {}
        for tf in timeframes:
            # Check cache first
            cache_key = f"{symbol}_{tf}_{lookback_days}d"
            if cache_key in self._cache:
                print(f"  Using cached data for {tf}")
                result[tf] = self._cache[cache_key]
                continue

            # Fetch fresh data
            try:
                df = self.fetch_historical_ohlcv(
                    symbol=symbol,
                    interval=tf,
                    lookback_days=lookback_days,
                )
                result[tf] = df
                self._cache[cache_key] = df

            except Exception as e:
                print(f"  ⚠️ Failed to fetch {tf} data: {e}")
                # Continue with other timeframes even if one fails
                continue

        # Verify we have at least some data
        if not result:
            raise ValueError(f"Failed to fetch any timeframe data for {symbol}")

        # Align timestamps to common end date
        result = self._align_timeframes(result)

        print(f"✓ Fetched {len(result)} timeframes for {symbol}")
        for tf, df in result.items():
            print(f"  {tf}: {len(df)} candles")

        return result

    def _align_timeframes(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align timeframes to the same end timestamp.

        Ensures all timeframes end at the same time, so strategies can be
        tested fairly across different granularities.

        Args:
            data: Dictionary of timeframe -> DataFrame

        Returns:
            Dictionary with aligned DataFrames
        """
        if not data:
            return data

        # Find the earliest end timestamp across all timeframes
        min_end_timestamp = min(df['timestamp'].iloc[-1] for df in data.values())

        # Truncate all dataframes to end at the same timestamp
        aligned = {}
        for tf, df in data.items():
            aligned_df = df[df['timestamp'] <= min_end_timestamp].copy()
            aligned[tf] = aligned_df

        return aligned

    def clear_cache(self):
        """Clear the internal cache of fetched data."""
        self._cache.clear()

    def _parse_candles(self, data: list) -> pd.DataFrame:
        """
        Parse Hyperliquid candle data into standardized DataFrame.

        Expected format from API:
        [
            {
                "t": timestamp_ms,
                "T": timestamp_ms_close,
                "s": symbol,
                "i": interval,
                "o": open_price,
                "h": high_price,
                "l": low_price,
                "c": close_price,
                "v": volume,
                "n": num_trades
            },
            ...
        ]
        """
        rows = []
        for candle in data:
            rows.append({
                'timestamp': candle['t'],
                'open': float(candle['o']),
                'high': float(candle['h']),
                'low': float(candle['l']),
                'close': float(candle['c']),
                'volume': float(candle['v']),
                'trades': int(candle.get('n', 0)),
                # Hyperliquid doesn't provide funding_rate or open_interest in candles
                # We'll set these to 0 for now, or fetch separately if needed
                'funding_rate': 0.0,
                'open_interest': 0.0,
            })

        df = pd.DataFrame(rows)

        # Sort by timestamp ascending
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def fetch_meta_info(self) -> dict:
        """
        Fetch metadata about available perpetuals markets.

        Returns:
            Dict with universe info about all available markets
        """
        payload = {
            "type": "meta"
        }

        try:
            response = self.session.post(self.API_URL, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching meta info: {e}")
            raise


def fetch_purr_data(
    lookback_days: int = 60,
    interval: str = "1h",
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Convenience function to fetch PURR perpetual data.

    Args:
        lookback_days: Number of days to fetch (default: 60)
        interval: Candle interval (default: "1h")
        output_path: Where to save the CSV (default: benchmark_data/trading/purr_60d.csv)

    Returns:
        DataFrame with OHLCV data
    """
    if output_path is None:
        output_path = Path("benchmark_data/trading/purr_60d.csv")

    fetcher = HyperliquidDataFetcher()
    df = fetcher.fetch_historical_ohlcv(
        symbol="PURR",
        interval=interval,
        lookback_days=lookback_days,
        save_path=output_path,
    )

    return df


if __name__ == "__main__":
    # Test the fetcher
    print("Testing Hyperliquid data fetcher...")
    df = fetch_purr_data()
    print(f"\nFetched data shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nLast few rows:\n{df.tail()}")
    print(f"\nData info:\n{df.info()}")
