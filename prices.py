"""
Daily stock prices from Nasdaq.com's free historical-quotes endpoint (no API key).

Note: Nasdaq returns UNADJUSTED closes (not corrected for stock splits). Step 4 will add
split adjustment before computing returns.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "application/json",
}


def daily_prices(ticker: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Return a DataFrame indexed by date with columns open, high, low, close, volume (oldest first)."""
    end = end or date.today()
    start = start or end - timedelta(days=365 * 6)
    url = (f"https://api.nasdaq.com/api/quote/{ticker.upper().replace('-', '.')}/historical"
           f"?assetclass=stocks&fromdate={start}&todate={end}&limit=9999")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = (r.json() or {}).get("data") or {}
    rows = ((data.get("tradesTable") or {}).get("rows")) or []
    if not rows:
        raise ValueError(f"No price data from Nasdaq for {ticker}.")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    return df.set_index("date").sort_index()[["open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    df = daily_prices(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
    print(df.head(2), "\n...\n", df.tail(2), f"\n{len(df)} rows")
