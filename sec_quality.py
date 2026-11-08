"""
Step 1: Financial red-flag screen from SEC EDGAR "company facts" data.

QUALITY SCORE (0-100) — four components, 25 points each
=======================================================
Window: the latest fiscal year plus up to 5 prior years (6 annual 10-K data points).
"Start" = first year in the window, "End" = latest year.

1. Cash conversion (25 pts)
   ratio = sum(operating cash flow) / sum(net income) over the window
   points = 25 * clamp((ratio - 0.5) / (1.0 - 0.5), 0, 1)
       -> ratio >= 1.0 gives 25 (every $1 of profit backed by >= $1 of cash)
       -> ratio <= 0.5 gives 0  (half or more of reported profit never showed up as cash)
   penalty: -10 for each year where net income > 0 but operating cash flow < 0
   (floor 0). If cumulative net income <= 0 the test isn't meaningful -> N/A.

2. Inventory build-up (25 pts)
   days_inv = inventory / revenue * 365   ("days of sales sitting in inventory")
   change = days_inv(End) - days_inv(Start)
   points = 25 * clamp(1 - change / 30, 0, 1)
       -> no increase gives 25; +30 days or more gives 0.
   Days rise exactly when inventory grows faster than revenue, so this IS the
   "inventory growth vs revenue growth" test, expressed in days so that a tiny
   inventory balance growing fast can't dominate the score (materiality).
   No inventory reported (banks, software) -> N/A.

3. Receivables build-up (25 pts)
   Same formula with accounts receivable (days_ar = receivables / revenue * 365).
   Rising days = customers paying slower, or revenue recognized before cash.

4. Gross margin trend (25 pts)
   gm = gross profit / revenue;  change = gm(End) - gm(Start), in percentage points
   points = 25 * clamp(1 + change / 5, 0, 1)
       -> flat or rising gives 25; a drop of 5 pp or more gives 0.
   No gross profit / cost of revenue reported -> N/A.

Total = sum of available components, rescaled to 100:
   score = 100 * (points earned) / (25 * number of available components)
A component scoring under half its points (12.5) is listed as a red flag.
Financial firms (SIC 6000-6799: banks, insurers, REITs) are excluded entirely.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

import pandas as pd
import requests

# SEC asks for "Name contact@email" in the User-Agent. Set SEC_USER_AGENT to your own.
USER_AGENT = os.environ.get("SEC_USER_AGENT", "StockQualityScreener class-project admin@example.com")
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

WINDOW_YEARS = 6

# Companies use different XBRL tag names for the same concept; first match wins per year.
TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "op_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "inventory": ["InventoryNet", "InventoryFinishedGoodsNetOfReserves", "InventoryFinishedGoods", "InventoryGross"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
}
FLOW_ITEMS = {"revenue", "net_income", "op_cash_flow", "gross_profit", "cost_of_revenue"}


# ---------------------------------------------------------------- data fetching
@lru_cache(maxsize=1)
def ticker_map() -> dict[str, dict]:
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return {v["ticker"].upper(): {"cik": int(v["cik_str"]), "name": v["title"]} for v in r.json().values()}


def lookup(ticker: str) -> dict:
    t = ticker.strip().upper().replace(".", "-")
    m = ticker_map()
    if t not in m:
        raise ValueError(f"Ticker '{ticker}' not found in SEC's ticker list.")
    return m[t]


@lru_cache(maxsize=64)
def company_facts(cik: int) -> dict:
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=64)
def sic_code(cik: int) -> int | None:
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=HEADERS, timeout=30)
    r.raise_for_status()
    sic = r.json().get("sic")
    return int(sic) if sic else None


def is_financial(cik: int) -> bool:
    """SIC 6000-6799 = banks, insurers, brokers, REITs. Their operating cash flow mixes in
    deposits, loans and trading, and they have no inventory or gross margin, so this
    screen does not apply to them."""
    sic = sic_code(cik)
    return sic is not None and 6000 <= sic <= 6799


def _annual_series(gaap: dict, tags: list[str], flow: bool) -> pd.Series:
    """Return {fiscal-year-end date: value} from 10-K filings, merging tags in priority order."""
    out: dict[date, float] = {}
    for tag in tags:
        if tag not in gaap:
            continue
        rows = []
        for x in gaap[tag]["units"].get("USD", []):
            if not x.get("form", "").startswith("10-K"):
                continue
            end = date.fromisoformat(x["end"])
            if flow:
                if "start" not in x:
                    continue
                days = (end - date.fromisoformat(x["start"])).days
                if not 350 <= days <= 380:  # 10-Ks also report quarters; keep full years only
                    continue
            elif "start" in x:
                continue
            rows.append((end, x["filed"], x["val"]))
        # if a year was reported several times (restatements), keep the most recently filed value
        latest: dict[date, tuple] = {}
        for end, filed, val in rows:
            if end not in latest or filed > latest[end][0]:
                latest[end] = (filed, val)
        for end, (_, val) in latest.items():
            out.setdefault(end, float(val))  # higher-priority tag already set it -> keep that
    return pd.Series(out, dtype=float).sort_index()


def annual_financials(ticker: str) -> tuple[dict, pd.DataFrame]:
    info = lookup(ticker)
    gaap = company_facts(info["cik"]).get("facts", {}).get("us-gaap", {})
    series = {k: _annual_series(gaap, tags, k in FLOW_ITEMS) for k, tags in TAGS.items()}
    rev = series["revenue"]
    if rev.empty:
        raise ValueError(f"No annual revenue found in SEC data for {ticker}.")
    # fiscal-year-end dates are defined by the revenue series; align everything to them
    df = pd.DataFrame(index=rev.index)
    for k, s in series.items():
        df[k] = s.reindex(df.index)
    if df["gross_profit"].isna().all() and df["cost_of_revenue"].notna().any():
        df["gross_profit"] = df["revenue"] - df["cost_of_revenue"]
    else:
        df["gross_profit"] = df["gross_profit"].fillna(df["revenue"] - df["cost_of_revenue"])
    df = df.drop(columns="cost_of_revenue")
    # drop years missing the core items, keep the most recent WINDOW_YEARS
    df = df.dropna(subset=["revenue", "net_income", "op_cash_flow"]).tail(WINDOW_YEARS)
    df.index = [f"FY{d.year}" for d in df.index]
    df.index.name = "fiscal_year"
    df["gross_margin"] = df["gross_profit"] / df["revenue"]
    df["days_inventory"] = df["inventory"] / df["revenue"] * 365
    df["days_receivables"] = df["receivables"] / df["revenue"] * 365
    df["cash_conversion"] = df["op_cash_flow"] / df["net_income"]
    return info, df


# ---------------------------------------------------------------- scoring
def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _cagr(a: float, b: float, years: int) -> float | None:
    if a is None or b is None or a <= 0 or b <= 0 or years <= 0:
        return None
    return (b / a) ** (1 / years) - 1


@dataclass
class Component:
    name: str
    points: float | None  # None = not applicable
    detail: str
    flags: list[str] = field(default_factory=list)


@dataclass
class QualityResult:
    ticker: str
    company: str
    cik: int
    data: pd.DataFrame
    components: list[Component]
    score: float

    @property
    def red_flags(self) -> list[str]:
        return [f for c in self.components for f in c.flags]


def _first_last(s: pd.Series):
    s = s.dropna()
    if len(s) < 2:
        return None
    return s.index[0], s.iloc[0], s.index[-1], s.iloc[-1]


def score_cash_conversion(df: pd.DataFrame) -> Component:
    ni, cfo = df["net_income"].sum(), df["op_cash_flow"].sum()
    if ni <= 0:
        return Component("Cash conversion", None, "Cumulative net income is not positive; test not meaningful.")
    ratio = cfo / ni
    pts = 25 * _clamp((ratio - 0.5) / 0.5)
    bad_years = [fy for fy, r in df.iterrows() if r.net_income > 0 and r.op_cash_flow < 0]
    pts = max(0.0, pts - 10 * len(bad_years))
    flags = []
    if bad_years:
        flags.append(f"Reported a profit but had negative operating cash flow in {', '.join(bad_years)}.")
    years = len(df) - 1
    g_ni = _cagr(df["net_income"].iloc[0], df["net_income"].iloc[-1], years)
    g_cfo = _cagr(df["op_cash_flow"].iloc[0], df["op_cash_flow"].iloc[-1], years)
    growth = ""
    if g_ni is not None and g_cfo is not None:
        growth = f" Net income grew {g_ni:.1%}/yr vs operating cash flow {g_cfo:.1%}/yr."
        if g_ni - g_cfo > 0.10:
            flags.append(f"Earnings growing much faster than cash flow ({g_ni:.1%}/yr vs {g_cfo:.1%}/yr).")
    if pts < 12.5 and not bad_years:
        flags.append(f"Only {ratio:.2f} of operating cash per $1 of reported profit over {df.index[0]}-{df.index[-1]}.")
    return Component("Cash conversion", pts,
                     f"${ratio:.2f} of operating cash flow per $1 of net income ({df.index[0]}-{df.index[-1]}).{growth}",
                     flags)


def score_days(df: pd.DataFrame, col: str, label: str, what: str, growth_col: str) -> Component:
    fl = _first_last(df[col])
    if fl is None:
        return Component(label, None, f"No {what} reported; not applicable.")
    y0, d0, y1, d1 = fl
    change = d1 - d0
    pts = 25 * _clamp(1 - change / 30)
    years = int(y1[2:]) - int(y0[2:])
    g_item = _cagr(df.loc[y0, growth_col], df.loc[y1, growth_col], years)
    g_rev = _cagr(df.loc[y0, "revenue"], df.loc[y1, "revenue"], years)
    growth = ""
    if g_item is not None and g_rev is not None:
        growth = f" {what.capitalize()} grew {g_item:.1%}/yr vs revenue {g_rev:.1%}/yr."
    flags = []
    if pts < 12.5:
        flags.append(f"{what.capitalize()} piling up: {d0:.0f} -> {d1:.0f} days of sales ({y0} to {y1}).{growth}")
    return Component(label, pts, f"{d0:.1f} -> {d1:.1f} days of sales ({change:+.1f} days, {y0} to {y1}).{growth}", flags)


def score_gross_margin(df: pd.DataFrame) -> Component:
    fl = _first_last(df["gross_margin"])
    if fl is None:
        return Component("Gross margin trend", None, "No gross profit / cost of revenue reported; not applicable.")
    y0, m0, y1, m1 = fl
    change_pp = (m1 - m0) * 100
    pts = 25 * _clamp(1 + change_pp / 5)
    flags = []
    if pts < 12.5:
        flags.append(f"Gross margin fell {m0:.1%} -> {m1:.1%} ({y0} to {y1}).")
    return Component("Gross margin trend", pts, f"{m0:.1%} -> {m1:.1%} ({change_pp:+.1f} pp, {y0} to {y1}).", flags)


def analyze(ticker: str) -> QualityResult:
    info = lookup(ticker)
    if is_financial(info["cik"]):
        raise ValueError(f"{ticker.upper()} is a bank/insurer/financial firm (SIC {sic_code(info['cik'])}); "
                         "this cash-flow/inventory/margin screen does not apply to financials.")
    info, df = annual_financials(ticker)
    if len(df) < 2:
        raise ValueError(f"Need at least 2 years of annual data for {ticker}; found {len(df)}.")
    comps = [
        score_cash_conversion(df),
        score_days(df, "days_inventory", "Inventory build-up", "inventory", "inventory"),
        score_days(df, "days_receivables", "Receivables build-up", "receivables", "receivables"),
        score_gross_margin(df),
    ]
    avail = [c for c in comps if c.points is not None]
    score = 100 * sum(c.points for c in avail) / (25 * len(avail)) if avail else float("nan")
    return QualityResult(ticker.upper(), info["name"], info["cik"], df, comps, score)


# ---------------------------------------------------------------- CLI
def _report(res: QualityResult) -> str:
    d = res.data
    tbl = pd.DataFrame({
        "Revenue $B": d["revenue"] / 1e9,
        "Net inc $B": d["net_income"] / 1e9,
        "Op CF $B": d["op_cash_flow"] / 1e9,
        "CFO/NI": d["cash_conversion"],
        "Inventory $B": d["inventory"] / 1e9,
        "Recv $B": d["receivables"] / 1e9,
        "Gross mgn %": d["gross_margin"] * 100,
        "Inv days": d["days_inventory"],
        "AR days": d["days_receivables"],
    }).round(2)
    lines = [f"{res.company} ({res.ticker}, CIK {res.cik})", "", tbl.to_string(), "", "Components:"]
    for c in res.components:
        p = "  N/A" if c.points is None else f"{c.points:5.1f}/25"
        lines.append(f"  {c.name:<22} {p}  {c.detail}")
    lines += ["", f"QUALITY SCORE: {res.score:.0f} / 100", "",
              "Red flags:" if res.red_flags else "Red flags: none"]
    lines += [f"  - {f}" for f in res.red_flags]
    return "\n".join(lines)


if __name__ == "__main__":
    for t in sys.argv[1:] or ["AAPL"]:
        try:
            print(_report(analyze(t)))
        except ValueError as e:
            print(f"{t}: {e}")
        print("\n" + "=" * 100 + "\n")
