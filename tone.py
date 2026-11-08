"""
Step 2: Earnings-call tone — how directly does management answer analysts' questions?

PIPELINE
  1. Find the ticker's transcripts on fool.com (listed on its quote page).
  2. For each of the most recent 4 calls, split the transcript into speaker turns and
     keep only the Q&A section (everything from the first analyst question onward).
  3. Group into question/answer pairs: one analyst turn + the management turns that follow.
  4. Score every answer 0-100 for directness, with a plain-English reason and an example quote.
     DEFAULT  - rule-based (free, deterministic): see "rule-based scoring" section for the formula.
     OPTIONAL - Claude scores each answer 1-5 against RUBRIC (needs ANTHROPIC_API_KEY and
                TONE_METHOD=llm); mapped to 0-100 as (score - 1) / 4 * 100.

TONE SCORE (0-100) = average answer score across all Q&A answers in the last 4 calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

MODEL = os.environ.get("TONE_MODEL", "claude-opus-5-5")
CACHE_DIR = Path(os.environ.get("TONE_CACHE_DIR", Path(__file__).parent / "cache"))
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
}
FOOL = "https://www.fool.com"

RUBRIC = """Score how DIRECTLY management answered the analyst's question, 1-5:
5 = Fully direct: answers exactly what was asked, with specifics (numbers, timing, a clear yes/no or commitment).
4 = Mostly direct: answers the core question, but with some hedging or a missing specific.
3 = Partial: addresses the topic in general terms without really answering, OR clearly and honestly declines
    ("we don't guide on that", "we won't break that out") — an explicit, upfront decline is a 3, not lower.
2 = Evasive: pivots to prepared talking points, answers a different question, or buries the question in generalities.
1 = Non-answer: ignores the question entirely or gives pure boilerplate.
Judge only directness, not whether the news is good or bad. A direct answer with bad news is still a 5."""


# ---------------------------------------------------------------- HTTP with a disk cache
def _get(url: str) -> str:
    """GET a fool.com page, caching the HTML on disk so re-runs and backtests don't re-scrape."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / ("html_" + hashlib.sha1(url.encode()).hexdigest()[:16] + ".html")
    if f.exists():
        return f.read_text(encoding="utf-8")
    time.sleep(1.0)  # be polite
    r = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
    r.raise_for_status()
    f.write_text(r.text, encoding="utf-8")
    return r.text


# ---------------------------------------------------------------- finding transcripts
@dataclass
class TranscriptRef:
    url: str
    quarter: str  # e.g. "Q3 2026" (company's fiscal quarter)
    published: date


_LINK = re.compile(r"/earnings/call-transcripts/(\d{4})/(\d{2})/(\d{2})/([a-z0-9\-]+)/")


def find_transcripts(ticker: str) -> list[TranscriptRef]:
    """Transcripts listed on the ticker's fool.com quote page, newest first."""
    t = ticker.lower().replace(".", "-")
    html = None
    for exch in ("nasdaq", "nyse", "nysemkt", "otc"):
        try:
            page = requests.get(f"{FOOL}/quote/{exch}/{t}/", headers=BROWSER_HEADERS, timeout=30)
        except requests.RequestException:
            continue
        if page.status_code == 200 and _LINK.search(page.text):
            html = page.text
            break
    if html is None:
        return []
    refs = {}
    for m in _LINK.finditer(html):
        y, mo, d, slug = m.groups()
        q = re.search(r"-q(\d)-(\d{4})-earnings", slug)
        refs[m.group(0)] = TranscriptRef(
            url=FOOL + m.group(0),
            quarter=f"Q{q.group(1)} {q.group(2)}" if q else slug,
            published=date(int(y), int(mo), int(d)),
        )
    return sorted(refs.values(), key=lambda r: r.published, reverse=True)


# ---------------------------------------------------------------- parsing a transcript
@dataclass
class QAPair:
    analyst: str
    question: str
    respondents: list[str]
    answer: str


@dataclass
class Transcript:
    url: str
    quarter: str
    call_date: date
    management: dict[str, str]  # name -> title
    qa: list[QAPair] = field(default_factory=list)


_SPEAKER = re.compile(r"^((?:[A-Z][\w.'\-]*)(?: (?:[A-Z][\w.'\-]*|de|van|von|di|da|del|la|le)){1,4}|Operator):\s+(.*)$", re.S)


def parse_transcript(ref: TranscriptRef) -> Transcript:
    soup = BeautifulSoup(_get(ref.url), "html.parser")
    body = soup.select_one(".article-body")
    if body is None:
        raise ValueError(f"Unrecognized transcript layout: {ref.url}")

    # Call date (falls back to publish date)
    call_date = ref.published
    for h in body.find_all("h2"):
        if h.get_text(strip=True).upper() == "DATE":
            p = h.find_next_sibling("p")
            if p:
                try:
                    call_date = datetime.strptime(p.get_text(strip=True).split(" at ")[0], "%A, %B %d, %Y").date()
                except ValueError:
                    pass

    # Management names from the CALL PARTICIPANTS list ("Chief Executive Officer - Tim Cook")
    mgmt: dict[str, str] = {}
    for h in body.find_all("h2"):
        if "PARTICIPANTS" in h.get_text().upper():
            ul = h.find_next_sibling("ul")
            for li in ul.find_all("li") if ul else []:
                parts = re.split(r"\s+[-–—]\s+", li.get_text(" ", strip=True))
                if len(parts) < 2:
                    continue
                title, name = " - ".join(parts[:-1]), parts[-1]
                if name and "analyst" not in title.lower():
                    mgmt[name.strip()] = title.strip()

    # Speaker turns (only after the "Full Conference Call Transcript" header if present)
    start = next((h for h in body.find_all("h2") if "TRANSCRIPT" in h.get_text().upper()), None)
    paras = start.find_all_next("p") if start else body.find_all("p")
    turns: list[list] = []  # [speaker, text]
    for p in paras:
        if body not in p.parents:
            break
        txt = p.get_text(" ", strip=True)
        if not txt:
            continue
        m = _SPEAKER.match(txt)
        if m:
            turns.append([m.group(1).strip(), m.group(2).strip()])
        elif turns:
            turns[-1][1] += "\n" + txt

    def is_mgmt(name: str) -> bool:
        return name in mgmt or any(name.split()[-1] == n.split()[-1] for n in mgmt)

    def is_host(name: str) -> bool:  # operator / investor-relations moderator: not an "answer"
        return name == "Operator" or "investor relations" in mgmt.get(name, "").lower()

    # Q&A = from the first turn by someone who is neither management nor the operator
    qa_start = next((i for i, (s, _) in enumerate(turns) if not is_mgmt(s) and s != "Operator"), None)
    pairs: list[QAPair] = []
    if qa_start is not None:
        cur: QAPair | None = None
        for speaker, text in turns[qa_start:]:
            if speaker == "Operator" or is_host(speaker):
                continue
            if is_mgmt(speaker):
                if cur is not None:
                    cur.respondents.append(speaker) if speaker not in cur.respondents else None
                    cur.answer += f"\n[{speaker}] {text}"
            else:  # analyst speaks -> new question
                if cur is not None:
                    pairs.append(cur)
                cur = QAPair(speaker, text, [], "")
        if cur is not None:
            pairs.append(cur)
    # drop pleasantries ("Thanks, Tim.") and unanswered questions
    pairs = [q for q in pairs if len(q.question.split()) >= 12 and len(q.answer.split()) >= 20]
    return Transcript(ref.url, ref.quarter, call_date, mgmt, pairs)


# ---------------------------------------------------------------- rule-based scoring (default, free)
# Each answer gets 0-100 points from three parts:
#   Specificity    (40): concrete facts per 100 words — percentages, $ amounts, "X billion",
#                        basis points, years / named quarters, or a plain "Yes"/"No" opening.
#                        3+ per 100 words = full 40.
#   Low hedging    (30): hedge/filler words per 100 words ("may", "might", "we'll see", "over time",
#                        plus promotional filler like "excited", "thrilled").
#                        0 = full 30, 4+ per 100 words = 0.
#   Responsiveness (30): share of the question's key terms that the answer also uses.
#                        35%+ overlap = full 30. Catches answers that pivot to a different topic.
#   Non-answer phrase:  -15 if the answer contains an explicit deflection
#                        ("we don't break that out", "not going to speculate", "too early to tell").
# Ticker tone score = average of all answer scores over the 4 calls.

SPECIFIC_PATTERNS = [
    r"\d[\d,.]*\s*(?:%|percent\b|basis points?\b|bps\b|points?\b)",
    r"\$\s?\d[\d,.]*",
    r"\b\d[\d,.]*\s*(?:billion|million|thousand)\b",
    r"\b(?:single|double|triple)[- ]digits?\b",
    r"\b(?:19|20)\d{2}\b",
    r"\b(?:March|June|September|December|first|second|third|fourth|next|this|last) quarter\b",
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|fifty|hundred)\s+(?:percent|billion|million)\b",
]
HEDGES = [
    "may", "might", "could", "possibly", "perhaps", "potentially", "uncertain", "uncertainty",
    "somewhat", "roughly", "probably", "likely", "unlikely", "hopefully", "generally", "kind of",
    "sort of", "we'll see", "we will see", "over time", "at this point", "it depends", "depending on",
    "hard to say", "difficult to say", "hard to predict", "difficult to predict", "remains to be seen",
    "a lot of moving pieces", "a number of factors", "we continue to evaluate", "evaluating all options",
    "we're evaluating", "we are evaluating", "early days", "not sure",
    # promotional filler: enthusiasm in place of information
    "excited", "thrilled", "incredibly", "incredible", "couldn't be more", "could not be more",
]
NON_ANSWERS = [
    "don't break", "do not break", "not going to break", "don't disclose", "do not disclose",
    "won't disclose", "not going to disclose", "don't provide", "do not provide", "don't guide",
    "do not guide", "not going to guide", "won't guide", "not going to comment", "won't comment",
    "don't comment", "do not comment", "can't comment", "cannot comment", "not going to speculate",
    "won't speculate", "don't want to speculate", "not going to get into", "won't get into",
    "too early to tell", "too early to say", "too early to comment", "not something we",
    "we're not going to talk", "we don't talk about", "want to go beyond", "not really want to talk",
    "n't really want to talk", "not want to talk", "n't want to talk", "get ahead of ourselves",
    "get ahead of that", "not going to go into", "won't go into", "make any comment",
    "comment on future", "comments about future", "comment on our road map", "comment on the road map",
]
STOP = set("""a an the and or but if then than that this these those there their they them we our us you your
i me my he she it its is are was were be been being have has had do does did of in on at to for from by with
about as into over under up down out so not no yes can could would should will just also more most very much
many some any all each what which who whom how why when where think thank thanks question questions follow
maybe kind sort little bit like well first second guess wanted want know talk talking comment comments color
quarter quarters year years going get give terms look looking around really great helpful""".split())

_HEDGE_RE = re.compile(r"\b(?:" + "|".join(re.escape(h) for h in HEDGES) + r")\b", re.I)
_NONANS_RE = re.compile("|".join(re.escape(n) for n in NON_ANSWERS), re.I)
_SPEC_RE = re.compile("|".join(SPECIFIC_PATTERNS), re.I)


def _key_terms(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z\-]+", text.lower())
    return {w.rstrip("s") for w in words if len(w) > 3 and w not in STOP}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def rule_score(question: str, answer: str) -> dict:
    ans = re.sub(r"\[[^\]]+\]\s*", "", answer).replace("’", "'")  # drop [Speaker] tags
    n = max(len(ans.split()), 1)
    specifics = _SPEC_RE.findall(ans)
    if re.match(r"\s*(?:yes|no)\b(?![.,!]?\s*(?:thank|thanks|look|so|well|i think)\b)", ans, re.I):  # not "Yes. Thanks..."
        specifics.append("yes/no")
    hedges = _HEDGE_RE.findall(ans)
    nonans = _NONANS_RE.findall(ans)
    qk, ak = _key_terms(question), _key_terms(ans)
    overlap = len(qk & ak) / len(qk) if qk else 1.0

    spec_pts = 40 * _clamp01((len(specifics) / n * 100) / 3)
    hedge_pts = 30 * _clamp01(1 - (len(hedges) / n * 100) / 4)
    resp_pts = 30 * _clamp01(overlap / 0.35)
    score = max(0.0, spec_pts + hedge_pts + resp_pts - (15 if nonans else 0))

    # plain-English reason + the most telling sentence as the example quote
    bits = []
    if spec_pts < 20:
        bits.append(f"few concrete facts ({len(specifics)} in {n} words)")
    if hedge_pts < 15:
        bits.append(f"heavy hedging ({len(hedges)} hedge words: {', '.join(sorted(set(h.lower() for h in hedges))[:4])})")
    if resp_pts < 15:
        bits.append(f"addresses only {overlap:.0%} of the question's key terms")
    if nonans:
        bits.append(f"explicit deflection (\"{nonans[0]}\")")
    sentences = re.split(r"(?<=[.!?])\s+", ans.strip())
    flagged = [s for s in sentences if _NONANS_RE.search(s)] or [s for s in sentences if _HEDGE_RE.search(s)] or sentences
    quote = flagged[0]
    quote = " ".join(quote.split()[:35]) + ("..." if len(quote.split()) > 35 else "")
    return {"score": round(score, 1), "spec_pts": round(spec_pts, 1), "hedge_pts": round(hedge_pts, 1),
            "resp_pts": round(resp_pts, 1), "deflection": bool(nonans),
            "reason": "; ".join(bits) if bits else "specific, on-topic answer with little hedging",
            "quote": quote}


def score_transcript_rules(tr: Transcript) -> list[dict]:
    out = []
    for i, q in enumerate(tr.qa, 1):
        r = rule_score(q.question, q.answer)
        out.append({"index": i, **r, "analyst": q.analyst, "respondents": q.respondents, "question": q.question})
    return out


# ---------------------------------------------------------------- LLM scoring (optional; needs ANTHROPIC_API_KEY)
class AnswerScore(BaseModel):
    index: int = Field(description="The [Q#] number of the exchange")
    score: int = Field(description="Directness 1-5 per the rubric")
    reason: str = Field(description="One sentence explaining the score")
    quote: str = Field(description="A short verbatim excerpt (<= 30 words) from the answer that best shows the score")


class TranscriptScores(BaseModel):
    scores: list[AnswerScore]


def score_transcript(tr: Transcript, company: str) -> list[dict]:
    """Score every Q&A exchange in one call. Results are cached on disk by transcript URL + model."""
    if not tr.qa:
        return []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{tr.url}|{MODEL}|{RUBRIC}".encode()).hexdigest()[:16]
    cf = CACHE_DIR / f"scores_{key}.json"
    if cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))

    import anthropic  # imported here so the rest of the module works without the SDK/key

    exchanges = "\n\n".join(
        f"[Q{i}] ANALYST ({q.analyst}): {q.question}\nMANAGEMENT ANSWER:{q.answer}"
        for i, q in enumerate(tr.qa, 1)
    )
    prompt = (
        f"Below is the Q&A section of {company}'s {tr.quarter} earnings call ({tr.call_date}).\n\n"
        f"{RUBRIC}\n\nScore every exchange [Q1]..[Q{len(tr.qa)}]. Quotes must be copied verbatim "
        f"from the management answer.\n\n{exchanges}"
    )
    client = anthropic.Anthropic()
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
        output_format=TranscriptScores,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"Model returned no scores for {tr.url} (stop_reason={resp.stop_reason}).")
    out = []
    for s in resp.parsed_output.scores:
        if 1 <= s.index <= len(tr.qa):
            q = tr.qa[s.index - 1]
            s5 = max(1, min(5, s.score))
            out.append({"index": s.index, "llm_score": s5, "score": to_100(s5), "reason": s.reason,
                        "quote": s.quote, "analyst": q.analyst, "respondents": q.respondents,
                        "question": q.question})
    cf.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------- per-ticker tone
@dataclass
class ToneResult:
    ticker: str
    method: str  # "rules" or "llm"
    tone_score: float | None  # 0-100
    calls: list[dict]  # per call: quarter, date, url, n_answers, avg, answers[]


def to_100(score_1to5: float) -> float:
    return (score_1to5 - 1) / 4 * 100


def default_method() -> str:
    return "llm" if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("TONE_METHOD") == "llm" else "rules"


def analyze_tone(ticker: str, company: str = "", n_calls: int = 4,
                 refs: list[TranscriptRef] | None = None, method: str | None = None) -> ToneResult:
    method = method or default_method()
    refs = (refs if refs is not None else find_transcripts(ticker))[:n_calls]
    calls, all_scores = [], []
    for ref in refs:
        tr = parse_transcript(ref)
        answers = (score_transcript(tr, company or ticker.upper()) if method == "llm"
                   else score_transcript_rules(tr))
        s = [a["score"] for a in answers]
        all_scores += s
        calls.append({"quarter": tr.quarter, "date": tr.call_date.isoformat(), "url": tr.url,
                      "n_answers": len(s), "avg": sum(s) / len(s) if s else None, "answers": answers})
    tone = sum(all_scores) / len(all_scores) if all_scores else None
    return ToneResult(ticker.upper(), method, tone, calls)


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ticker = args[0] if args else "AAPL"
    refs = find_transcripts(ticker)[:4]
    if "--parse-only" in sys.argv:  # check extraction without calling the LLM
        for ref in refs:
            tr = parse_transcript(ref)
            print(f"\n=== {tr.quarter}  call {tr.call_date}  {tr.url}")
            print(f"management: {', '.join(f'{n} ({t})' for n, t in tr.management.items())}")
            print(f"{len(tr.qa)} Q&A exchanges")
            for i, q in enumerate(tr.qa, 1):
                print(f"  [Q{i}] {q.analyst} -> {', '.join(q.respondents)} | "
                      f"Q {len(q.question.split())}w: {q.question[:90]!r} | A {len(q.answer.split())}w")
        sys.exit()
    res = analyze_tone(ticker, refs=refs, method="llm" if "--llm" in sys.argv else "rules")
    print(f"{res.ticker} TONE SCORE ({res.method}): {res.tone_score:.0f}/100" if res.tone_score is not None else "no scores")
    for c in res.calls:
        sc = sorted(a["score"] for a in c["answers"])
        print(f"\n{c['quarter']} ({c['date']}): {c['n_answers']} answers, avg {c['avg']:.0f}/100 "
              f"(min {sc[0]:.0f}, median {sc[len(sc)//2]:.0f}, max {sc[-1]:.0f})")
        for a in sorted(c["answers"], key=lambda a: a["score"])[:2]:
            print(f"  {a['score']:.0f}/100 [Q{a['index']} {a['analyst']} -> {', '.join(a['respondents'])}] {a['reason']}\n"
                  f"      Q: {a['question'][:160]}...\n      A: \"{a['quote']}\"")
