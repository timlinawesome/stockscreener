"""Stock Quality Screener — early test version (Steps 1-2 + a connection check)."""
import time

import requests
import streamlit as st

import prices
import sec_quality
import tone

st.set_page_config(page_title="Stock Quality Screener", page_icon="📊", layout="centered")
st.title("Stock Quality Screener")
st.caption("Early test build: financial red flags + earnings-call tone. Verdict and backtest coming next.")


# ---------------------------------------------------------------- connection check
def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        return {"Source": name, "Status": "✅ OK", "Detail": detail, "Seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001 - we want to display any failure
        return {"Source": name, "Status": "❌ Failed", "Detail": f"{type(e).__name__}: {e}"[:200],
                "Seconds": round(time.time() - t0, 1)}


def _sec():
    r = requests.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
                     headers=sec_quality.HEADERS, timeout=30)
    r.raise_for_status()
    return f"HTTP {r.status_code}, {len(r.content)//1024} KB of Apple financial data"


def _fool():
    refs = tone.find_transcripts("AAPL")
    if not refs:
        raise RuntimeError("quote page loaded but no transcript links found (possibly blocked)")
    r = requests.get(refs[0].url, headers=tone.BROWSER_HEADERS, timeout=30)
    r.raise_for_status()
    if "article-body" not in r.text:
        raise RuntimeError(f"HTTP {r.status_code} but page has no transcript body (possibly a block page)")
    return f"{len(refs)} AAPL transcripts listed; latest {refs[0].quarter} downloaded"


def _nasdaq():
    df = prices.daily_prices("AAPL")
    return f"{len(df)} daily prices, last close ${df['close'].iloc[-1]:.2f} on {df.index[-1].date()}"


with st.expander("🔌 Connection check (can this server reach the data sources?)"):
    if st.button("Run connection check"):
        with st.spinner("Contacting SEC, Motley Fool and Nasdaq..."):
            st.table([check("SEC EDGAR", _sec), check("Motley Fool", _fool), check("Nasdaq prices", _nasdaq)])


# ---------------------------------------------------------------- analysis
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def run_quality(t):
    return sec_quality.analyze(t)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def run_tone(t, company):
    return tone.analyze_tone(t, company)


ticker = st.text_input("Stock ticker", value="AAPL", max_chars=10).strip().upper()
if st.button("Analyze", type="primary") and ticker:
    with st.spinner("Reading SEC filings..."):
        try:
            q = run_quality(ticker)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.stop()
    st.subheader(f"{q.company} ({q.ticker})")

    st.markdown(f"### Financial quality score: **{q.score:.0f} / 100**")
    for c in q.components:
        pts = "not applicable" if c.points is None else f"{c.points:.1f} / 25"
        st.markdown(f"- **{c.name}** — {pts}: {c.detail}")
    if q.red_flags:
        st.warning("**Red flags**\n\n" + "\n".join(f"- {f}" for f in q.red_flags))
    else:
        st.success("No financial red flags.")
    d = q.data
    st.line_chart((d[["revenue", "net_income", "op_cash_flow"]] / 1e9).rename(columns={
        "revenue": "Revenue ($B)", "net_income": "Net income ($B)", "op_cash_flow": "Operating cash flow ($B)"}))

    with st.spinner("Reading the last 4 earnings call transcripts..."):
        try:
            t = run_tone(ticker, q.company)
        except Exception as e:  # noqa: BLE001
            t = None
            st.error(f"Could not score earnings calls: {e}")
    if t and t.tone_score is not None:
        st.markdown(f"### Management directness: **{t.tone_score:.0f} / 100**")
        st.caption("Average directness of management's answers to analyst questions over the last 4 calls.")
        worst = sorted((a | {"quarter": c["quarter"]} for c in t.calls for a in c["answers"]),
                       key=lambda a: a["score"])[:3]
        for a in worst:
            st.markdown(f"**{a['score']:.0f}/100** — {a['quarter']}, {a['analyst']} → {', '.join(a['respondents'])}  \n"
                        f"_{a['reason']}_  \n> {a['quote']}")
    elif t:
        st.info("No earnings call transcripts found for this ticker on fool.com.")
