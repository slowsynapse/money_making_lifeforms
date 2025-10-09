"""
Hyperliquid API data fetcher for OHLCV historical data.

Fetches candle data from Hyperliquid's perpetuals exchange API.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time


class HyperliquidDataFetcher:
    """Fetches historical OHLCV data from Hyperliquid API."""

    API_URL = "https://api.hyperliquid.xyz/info"

    def __init__(self):
        self.session = requests.Session()

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
