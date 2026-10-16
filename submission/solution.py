"""
Time-aware knowledge system for the Dupaco snapshots.

ingest()     keeps every version of every page and builds a view of the site per snapshot date,
             with a rate-table index (rows, tiers, effective dates) and a BM25 text index.
answer()     routes the question (rates, hours, address, link, phone, minimum, general text) and
             answers from the snapshot that was current on the as_of date.
conflicts()  every rate tier whose value differs across pages or snapshots, plus branch count and
             Forbes recognitions, with the value I trust and why.

See NOTES.md for the rules.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

# --- contract types --------------------------------------------------------------------
# Use the harness's own dataclasses when available (the scorer adds harness/ to sys.path),
# otherwise fall back to identical local copies so the module also imports standalone.
try:
    _here = Path(__file__).resolve()
    for _d in _here.parents:
        if (_d / "harness" / "contract.py").exists():
            sys.path.insert(0, str(_d / "harness"))
            break
    from contract import Answer, Conflict, Snapshot, Source  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - fallback only
    from dataclasses import dataclass, field

    @dataclass
    class Source:  # type: ignore[no-redef]
        url: str
        snapshot_id: Optional[str] = None
        quote: Optional[str] = None

    @dataclass
    class Answer:  # type: ignore[no-redef]
        text: str
        sources: list = field(default_factory=list)
        confident: bool = True
        conflict_note: Optional[str] = None

    @dataclass
    class Conflict:  # type: ignore[no-redef]
        topic: str
        values: list
        sources: list
        resolution: Optional[str] = None
        reason: Optional[str] = None

    Snapshot = object  # type: ignore


# =======================================================================================
# Text utilities
# =======================================================================================

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
MONTH_ABBR = {k[:3]: v for k, v in MONTHS.items()}

STOP = set("""
a an the and or of to in on for at by with from is are was were be been being do does did
i im i'm me my we our you your yours it its this that these those what whats what's which who
whom how when where why can could would should will shall may might must there their they
them he she his her as if than then so not no yes any all some about into over under up down
out tell please give get got know let need want like just also much many more most own same
dupaco dupacos cu currently current now today right still have has had does
""".split())

# Words that carry little product meaning in a rate question.
RATE_WORDS = {"rate", "rates", "apr", "apy", "dividend", "interest", "yield", "percent",
              "percentage", "annual", "earn", "pay", "pays", "paying", "charge", "charges",
              "offer", "offered", "offering", "price", "pricing", "cost"}

SYN = {
    "cd": "certificate", "cds": "certificate", "certificates": "certificate", "term-share": "certificate",
    "share-certificate": "certificate",
    "yr": "year", "yrs": "year", "years": "year", "mo": "month", "mos": "month", "months": "month",
    "car": "auto", "cars": "auto", "vehicle": "auto", "vehicles": "auto", "truck": "auto",
    "automobile": "auto", "autos": "auto",
    "mortgages": "mortgage", "home": "home", "homes": "home",
    "savings": "saving", "save": "saving",
    "cards": "card", "visas": "visa",
    "loans": "loan", "lending": "loan", "borrow": "loan",
    "accounts": "account",
    "checkings": "checking",
    "branches": "branch", "locations": "location", "offices": "office",
    "jumbos": "jumbo", "specials": "special", "promo": "special", "promotional": "special",
    "promotion": "special",
    "moneymarket": "money",
    "iras": "ira",
    "heloc": "heloc", "helocs": "heloc",
    "fees": "fee",
    "hrs": "hours", "hour": "hours",
    "atms": "atm",
    "rv": "rv", "rvs": "rv", "boats": "boat", "motorcycles": "motorcycle",
    "students": "student",
    "recognized": "recognition", "recognize": "recognition", "recognizes": "recognition",
    "recognitions": "recognition", "honored": "recognition", "award": "recognition", "awarded": "recognition",
    "closing": "close", "closed": "close", "closes": "close", "closure": "close",
    "opened": "open", "opening": "open", "opens": "open",
    "eligibility": "eligible", "qualify": "eligible", "qualifies": "eligible",
    "joined": "join", "joining": "join", "joins": "join",
    "volunteered": "volunteer", "volunteering": "volunteer", "volunteers": "volunteer",
    "used": "use", "using": "use", "uses": "use",
    "updated": "update", "updating": "update", "change": "update", "changing": "update",
    "applied": "apply", "applying": "apply", "application": "apply",
    "requirements": "require", "required": "require", "requirement": "require",
    "employee": "employee", "employees": "employee", "staff": "employee",
    "located": "location", "locate": "location",
    "wallets": "wallet", "intro": "introductory", "payments": "pay", "payment": "pay", "paying": "pay", "apps": "app",
    "application": "apply",
}

WORD_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


def stem(w: str) -> str:
    w = SYN.get(w, w)
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith("sses"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    return SYN.get(w, w)


def tokens(text: str, drop_stop: bool = False) -> list[str]:
    text = re.sub(r"credit union", " ", (text or "").lower())
    text = text.replace("home$tart", "homestart")
    text = (text).replace("’", "'").replace("–", "-").replace("—", "-")
    text = text.replace("$", " ").replace("%", " ")
    out = []
    for w in WORD_RE.findall(text):
        w = w.strip(".'")
        if w.endswith("'s"):
            w = w[:-2]
        if not w:
            continue
        if drop_stop and (w in STOP or (len(w) < 2 and not w.isdigit())):
            continue
        out.append(stem(w))
    return out


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text, drop_stop=True) if t not in STOP or t.isdigit()]


def parse_date(s: str) -> Optional[date]:
    s = s.strip()
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    m = re.match(r"([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})", s)
    if m:
        mon = MONTHS.get(m.group(1).lower()) or MONTH_ABBR.get(m.group(1).lower()[:3])
        if mon:
            try:
                return date(int(m.group(3)), mon, int(m.group(2)))
            except ValueError:
                return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def norm_pct(v: str) -> Optional[float]:
    m = PCT_RE.search(v or "")
    return round(float(m.group(1)), 4) if m else None


def fmt_pct(x: float) -> str:
    t = f"{x:.3f}".rstrip("0")
    if len(t.split(".")[1]) < 2:
        t = f"{x:.2f}"
    return t + "%"


# =======================================================================================
# Page model
# =======================================================================================

COUNTRY_LINE = re.compile(r"^\s*- [\U0001F1E6-\U0001F1FF]{2}")
NOISE_LINE = re.compile(r"^\s*(!\[|\[Continue\]|✕|\* \* \*|Previous$|Next$|Loading locations|Go$)")
MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
URL_DATE = re.compile(r"/(20\d\d|19\d\d)/(\d{2})/(\d{2})/")
HEAD_DATE = re.compile(
    r"\n\s*((?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s*\d{4})\s*\n")


def page_kind(url: str) -> str:
    u = url.lower()
    path = re.sub(r"^https?://[^/]+", "", u)
    if "privacy-policy?" in u or "?campaignid" in u:
        return "junk"
    if path.rstrip("/") in ("/rates", "/business-rates"):
        return "ratesheet"
    if "/newsroom/tag/" in path or "/resources/tag/" in path or "/author/" in path or "?category=" in u:
        return "listing"
    if "/calculators/" in path or "calculator" in path:
        return "calculator"
    if path.endswith("-lp") or "-lp?" in path or path.rstrip("/") in ("/switch-and-earn", "/offer-expired"):
        return "landing"
    if "/newsroom/" in path:
        return "news"
    if "/learn/blog/" in path or URL_DATE.search(path) or "/business/resources/" in path:
        return "blog"
    if "/hubfs/" in path:
        return "pdf"
    if not u.startswith("https://www.dupaco.com"):
        return "other"
    return "product"


KIND_PRIOR = {"ratesheet": 1.0, "product": 1.0, "pdf": 0.8, "news": 0.85, "blog": 0.75,
              "landing": 0.55, "calculator": 0.45, "listing": 0.25, "junk": 0.05, "other": 0.6}
TABLE_PRIOR = {"ratesheet": 1.0, "product": 0.97, "pdf": 0.7, "news": 0.6, "blog": 0.5,
               "landing": 0.55, "calculator": 0.45, "listing": 0.2, "junk": 0.05, "other": 0.5}


FOOTER_CUT = re.compile(r"\n#+\s*(Handpicked for You|Related (Articles|Posts|Stories)|You May Also Like)\b.*", re.I | re.S)


def clean_markdown(md: str) -> str:
    md = FOOTER_CUT.sub("\n", md or "")
    lines = []
    for ln in md.replace("\\r\\n", " ").replace("\r", "").split("\n"):
        if COUNTRY_LINE.match(ln) or NOISE_LINE.match(ln):
            continue
        if "data:image" in ln or len(ln) > 4000:
            continue
        lines.append(ln.rstrip())
    return "\n".join(lines)


def pub_date(url: str, md: str) -> Optional[date]:
    m = URL_DATE.search(url)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = HEAD_DATE.search(md[:4000])
    if m:
        return parse_date(m.group(1))
    return None


class PageVersion:
    __slots__ = ("url", "title", "md", "snap_idx", "snapshot_id", "kind", "pub", "_tables", "_chunks",
                 "_links")

    def __init__(self, url, title, md, snap_idx, snapshot_id):
        self.url = url
        self.title = title or ""
        self.md = clean_markdown(md or "")
        self.snap_idx = snap_idx
        self.snapshot_id = snapshot_id
        self.kind = page_kind(url)
        self.pub = pub_date(url, self.md) if self.kind in ("news", "blog") else None
        self._tables = None
        self._chunks = None
        self._links = None


# =======================================================================================
# Tables
# =======================================================================================

RATE_COL = re.compile(r"rate|apr|apy|yield|dividend", re.I)
FOOTNOTE_TAIL = re.compile(r"(%|APR|APY)\s*\d\b")
EFFECTIVE = re.compile(r"effective (?:as of )?(\d{1,2}/\d{1,2}/\d{2,4}|[A-Z][a-z]+\.? \d{1,2},? \d{4})", re.I)

DISCRIM = {"business", "jumbo", "refinance", "roth", "traditional", "ira", "introductory", "intro",
           "transfer", "kid", "youth", "hsa", "health", "mini", "max", "equity", "improvement",
           "boat", "rv", "atv", "snowmobile", "jet", "ski", "special", "rewards", "platinum", "heloc",
           "launchpad", "bright", "holiday", "high", "investor", "choice", "achieve", "name", "share",
           "checking", "personal", "auto", "motorcycle", "purchase", "variable", "regular", "money",
           "market", "student"}
PENALIZE = {"business", "jumbo", "refinance", "roth", "traditional", "ira"}
REQUIRED = {"business", "jumbo", "refinance", "roth", "traditional", "ira", "introductory", "intro", "transfer",
            "hsa", "health", "boat", "rv", "atv", "snowmobile", "jet", "ski", "rewards", "platinum", "heloc",
            "launchpad", "bright", "holiday", "achieve", "checking", "auto", "motorcycle", "student", "equipment",
            "commercial", "construction", "land", "equity", "improvement", "personal", "mortgage", "visa",
            "certificate", "saving"}
ROW_OPTIONAL = re.compile(r"intro|balance transfer|mini-?max|relief program", re.I)


def clean_cell(c: str) -> str:
    c = c.strip().replace("\\*", "*").replace("\\$", "$").replace("\\#", "#")
    c = MD_LINK.sub(r"\1", c)
    c = FOOTNOTE_TAIL.sub(r"\1", c)
    c = re.sub(r"\*+$", "", c).strip()
    return re.sub(r"\s+", " ", c)


class Table:
    __slots__ = ("page", "caption", "header", "rows", "effective", "rate_cols", "label_cols", "doc_tokens",
                 "signature", "details_url", "cap_tokens", "pos")

    def __init__(self, page, caption, header, rows, effective, details_url, pos):
        self.page = page
        self.caption = caption
        self.header = header
        self.rows = rows
        self.effective = effective
        self.details_url = details_url
        self.pos = pos
        self.rate_cols = [i for i, h in enumerate(header) if RATE_COL.search(h) and not re.search(
            r"payment|min\.?|points", h, re.I)]
        if not self.rate_cols and re.search(r"rate|apr|apy", " ".join([caption] + header), re.I):
            self.rate_cols = [i for i in range(len(header)) if rows and any("%" in r[i] for r in rows if i < len(r))]
        self.label_cols = [i for i in range(len(header)) if i not in self.rate_cols and not re.search(
            r"min\.?|payment|points", header[i], re.I)]
        label_text = " ".join(" ".join(r[i] for i in self.label_cols if i < len(r)) for r in rows)
        self.cap_tokens = set(content_tokens(caption))
        if re.search(r"home loan|conventional|mortgage", caption + " " + label_text, re.I):
            self.cap_tokens |= {"mortgage"}
        self.doc_tokens = set(content_tokens(caption + " " + " ".join(header) + " " + label_text)) | self.cap_tokens
        self.signature = (tuple(re.sub(r"[^a-z0-9%]", "", h.lower()) for h in header),
                          tuple(re.sub(r"[^a-z0-9]", "", self.row_label(r).lower().replace("purchase", ""))
                                for r in rows))

    def row_label(self, r) -> str:
        return ", ".join(r[i] for i in self.label_cols if i < len(r) and r[i] and r[i] != "—")

    def row_rates(self, r) -> list[tuple[str, str]]:
        out = []
        for i in self.rate_cols:
            if i < len(r) and r[i] and r[i] != "—":
                out.append((self.header[i], r[i]))
        return out


def parse_tables(pv: PageVersion) -> list[Table]:
    lines = pv.md.split("\n")
    tables = []
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].lstrip().startswith("|") and i + 1 < n and re.match(r"^\s*\|\s*:?-{3}", lines[i + 1]):
            header = [clean_cell(c) for c in lines[i].strip().strip("|").split("|")]
            j = i + 2
            rows = []
            while j < n and lines[j].lstrip().startswith("|"):
                cells = [clean_cell(c) for c in lines[j].strip().strip("|").split("|")]
                if any(cells):
                    rows.append(cells + [""] * (len(header) - len(cells)))
                j += 1
            # caption: up to 3 short non-table lines above
            cap = []
            k = i - 1
            seen_lines = 0
            while k >= 0 and len(cap) < 3 and seen_lines < 8:
                s = lines[k].strip()
                k -= 1
                if not s:
                    continue
                seen_lines += 1
                if s.startswith("|") or len(s) >= 90:
                    break
                if s.startswith("[") or re.search(r"(Rates|Rate) are effective", s) or re.match(r"^\d[A-Z]", s):
                    continue
                s = re.sub(r"^[-*]\s+", "", s)
                cap.append(re.sub(r"^#+\s*", "", MD_LINK.sub(r"\1", s)).strip("*").strip())
            cap = list(reversed(cap))
            if len(cap) == 1 and tables and " / " in tables[-1].caption and i - tables[-1].pos < 60:
                cap = [tables[-1].caption.split(" / ")[0]] + cap
            caption = " / ".join(cap)
            eff = None
            details = None
            after = []
            for t in lines[j:j + 60]:
                if t.strip().startswith("|"):
                    break
                if t.strip():
                    after.append(t)
                if len(after) >= 8:
                    break
            for t in after:
                if eff is None:
                    m = EFFECTIVE.search(t)
                    if m and not re.match(r"^\s*\d", t.strip()):
                        eff = parse_date(m.group(1))
                for m in MD_LINK.finditer(t):
                    if "see details" in m.group(1).lower() or "learn more" in m.group(1).lower():
                        if "dupaco.com" in m.group(2) and not details:
                            details = m.group(2)
            if eff is None:
                for t in after:
                    m = EFFECTIVE.search(t)
                    if m:
                        eff = parse_date(m.group(1))
                        break
            if rows and any("%" in c for r in rows for c in r):
                tb = Table(pv, caption, header, rows, eff, details, i)
                if tb.rate_cols:
                    tables.append(tb)
            i = j
        else:
            i += 1
    return tables


# --- qualifiers ---------------------------------------------------------------------------

def _money(s: str) -> Optional[float]:
    s = s.replace(",", "").strip()
    m = re.match(r"(\d+(?:\.\d+)?)\s*(k|thousand|m|million)?", s, re.I)
    if not m:
        return None
    v = float(m.group(1))
    suf = (m.group(2) or "").lower()
    if suf in ("k", "thousand"):
        v *= 1000
    elif suf in ("m", "million"):
        v *= 1_000_000
    return v


def row_quals(label: str) -> dict:
    q = {}
    l = label.replace("–", "-").replace("—", "-")
    m = re.search(r"(\d+)\s*(?:-|\s)?\s*(?:month|mo\b)", l, re.I)
    if m:
        q["months"] = int(m.group(1))
    m = re.search(r"(\d+)\s*-?\s*year", l, re.I)
    if m and "model" not in l.lower()[max(0, m.start() - 12):m.end() + 6]:
        q["months"] = int(m.group(1)) * 12
    m = re.search(r"\b((?:19|20)\d\d)\s*-\s*(\d{2,4})\s*model", l, re.I) or re.search(
        r"model year\s*((?:19|20)\d\d)\s*-\s*(\d{2,4})", l, re.I)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        if b < 100:
            b += (a // 100) * 100
        q["model"] = (min(a, b), max(a, b))
    else:
        m = re.search(r"\b((?:19|20)\d\d)\s*model year\s*\(and older\)", l, re.I)
        if m:
            q["model"] = (1900, int(m.group(1)))
    m = re.search(r"\$\s?([\d,]+(?:\.\d+)?)\s*-\s*\$?\s?([\d,]+(?:\.\d+)?)", l)
    if m:
        q["amount"] = (_money(m.group(1)), _money(m.group(2)))
    else:
        m = re.search(r"\$\s?([\d,]+(?:\.\d+)?)\s*\+", l)
        if m:
            q["amount"] = (_money(m.group(1)), float("inf"))
        else:
            m = re.search(r"\$\s?([\d,]+(?:\.\d+)?)\s*or less", l, re.I)
            if m:
                q["amount_max"] = _money(m.group(1))
    return q


NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fifteen": 15, "seventeen": 17, "eighteen": 18,
             "twenty": 20, "twenty-four": 24, "twenty-five": 25, "thirty": 30, "thirty-six": 36, "forty-eight": 48,
             "sixty": 60, "seventy-two": 72, "eighty-four": 84}
NUM_WORD_RE = re.compile(r"\b(" + "|".join(sorted(NUM_WORDS, key=len, reverse=True)) + r")(?=[\s-]+(?:month|year|yr))")


def question_quals(q: str) -> dict:
    out = {}
    ql = q.lower().replace("–", "-")
    ql = NUM_WORD_RE.sub(lambda m: str(NUM_WORDS[m.group(1)]), ql)
    m = re.search(r"(\d+)\s*(?:-|\s)?\s*(?:month|mo\b|mos\b)", ql)
    if m:
        out["months"] = int(m.group(1))
    m = re.search(r"\b(\d{1,2})\s*(?:-|\s)?\s*(?:year|yr)s?\b", ql)
    if m and "months" not in out:
        out["months"] = int(m.group(1)) * 12
    m = re.search(r"\b((?:19|20)\d\d)\b", ql)
    if m and re.search(r"model|vehicle|car\b|auto|truck|motorcycle|atv|snowmobile|jet ski|suv", ql):
        out["model"] = int(m.group(1))
    amts = []
    for m in re.finditer(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|million)?\b", ql):
        amts.append(_money(m.group(1) + (m.group(2) or "")))
    if not amts:
        for m in re.finditer(r"\b(\d{1,3}(?:,\d{3})+|\d+k)\b", ql):
            amts.append(_money(m.group(1)))
    amts = [a for a in amts if a is not None]
    if amts:
        out["amount"] = amts[0]
        if len(amts) > 1:
            out["amount_hi"] = amts[1]
    pm = PCT_RE.search(ql)
    if pm:
        out["pct"] = float(pm.group(1))
    if re.search(r"\b(highest|best|top|max(imum)?|most)\b", ql):
        out["extreme"] = "max"
    elif re.search(r"\b(lowest|cheapest|min(imum)? rate|least)\b", ql):
        out["extreme"] = "min"
    return out


def row_matches(rq: dict, qq: dict) -> bool:
    if "months" in qq and "months" in rq and rq["months"] != qq["months"]:
        return False
    if "model" in qq and "model" in rq:
        lo, hi = rq["model"]
        if not (lo <= qq["model"] <= hi):
            return False
    if "amount" in qq:
        a = qq["amount"]
        if "amount" in rq:
            lo, hi = rq["amount"]
            if not (lo - 0.5 <= a <= hi + 0.5):
                return False
        if "amount_max" in rq and a > rq["amount_max"] + 0.5:
            return False
    return True


# =======================================================================================
# Text chunks + BM25
# =======================================================================================

def count_hit(noun: str, text: str):
    """'22 branch office locations' or 'membership has grown to more than 181,000'."""
    n = re.escape(noun) + r"\w{0,4}\b"
    num = r"(?<![\d,.$])\d{1,3}(?:,\d{3})*\+?(?![\d,]*%)"
    tail = (r"(?:\s+[a-z]+){0,2}?\s*(?:[.,;:)]|\band\b|\bor\b|\bin\b|\bat\b|\bacross\b|\bwho\b|\bthat\b|\bwith\b|"
            r"\b(?:used|use|are|is|were|was|have|has|volunteered|served|received|work|worked|participated|took|"
            r"earned|joined|attended|completed|gave|donated|logged|chose|signed)\b|$)")
    return re.search(num + r"\s+(?:(?!of\b|in\b|outside\b|than\b|per\b|from\b|at\b|to\b)[A-Za-z’'-]+\s){0,2}" + n + tail,
                     text, re.I) or re.search(
        n + r"\s+(?:has grown to|grew to|of|totals?|reached|is|are|now)\s+(?:more than |over |nearly |almost |about |"
            r"approximately )?" + num, text, re.I)


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(-])|\n+")


def make_chunks(pv: PageVersion) -> list[tuple[str, str]]:
    """Return (heading, text) chunks. Tables are rendered as one line per row."""
    out = []
    heading = pv.title
    buf: list[str] = []

    def flush():
        if buf:
            txt = "\n".join(buf).strip()
            if len(txt) > 40 or (len(txt) > 12 and re.search(r"\d", txt)):
                out.append((heading, txt))
            buf.clear()

    lines = pv.md.split("\n")
    for ln in lines:
        s = ln.strip()
        if not s:
            if sum(len(b) for b in buf) > 350:
                flush()
            continue
        if s.startswith("#"):
            flush()
            heading = re.sub(r"^#+\s*", "", MD_LINK.sub(r"\1", s)).strip("* ")
            continue
        if s.startswith("|"):
            flush()
            if re.match(r"^\|\s*:?-{3}", s):
                continue
            cells = [clean_cell(c) for c in s.strip("|").split("|")]
            out.append((heading, " | ".join(c for c in cells if c)))
            continue
        if re.match(r"^\d+ min read$", s) or HEAD_DATE.match("\n" + s + "\n"):
            continue  # article byline bits
        if not MD_LINK.sub("", s).strip(" -*|·•>") and MD_LINK.search(s):
            continue  # a line made only of links/buttons
        if re.match(r"^[^\[]*\]\(https?://", s):
            continue  # tail of a link broken across lines
        s = MD_LINK.sub(r"\1", s)
        s = re.sub(r"\*\*|__", "", s)
        # fee-schedule style lines ("Item ... $5.00") become their own chunks
        if pv.kind == "pdf" and re.search(r"\$\d|\d%|\bFree\b", s) and len(s) < 220:
            flush()
            out.append((heading, s))
            continue
        buf.append(s)
        if sum(len(b) for b in buf) > 700:
            flush()
    flush()
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / self.N) if self.N else 1.0
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                self.post[t].append((i, c))
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def search(self, q: list[str]) -> dict[int, float]:
        sc: dict[int, float] = defaultdict(float)
        for t in set(q):
            p = self.post.get(t)
            if not p:
                continue
            idf = self.idf[t]
            for i, c in p:
                sc[i] += idf * c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return sc


# =======================================================================================
# A view of the website at one snapshot
# =======================================================================================

class View:
    def __init__(self, idx: int, taken_on: date, pages: dict[str, tuple[PageVersion, bool]]):
        self.idx = idx
        self.taken_on = taken_on
        self.pages = pages  # url -> (version, present_in_this_snapshot)
        self._build_tables()
        self._build_text()
        self._build_links()

    # ---- tables ----
    def _build_tables(self):
        self.tables: list[Table] = []
        for url, (pv, present) in self.pages.items():
            if pv.kind in ("junk", "listing"):
                continue
            if pv._tables is None:
                pv._tables = parse_tables(pv)
            for t in pv._tables:
                self.tables.append(t)
        self.present = {u for u, (_, p) in self.pages.items() if p}
        # table-vocabulary idf
        df = Counter()
        for t in self.tables:
            df.update(t.doc_tokens)
        n = max(1, len(self.tables))
        self.tidf = {w: math.log(1 + n / (1 + c)) for w, c in df.items()}
        self.table_vocab = set(df)

    # ---- text ----
    def _build_text(self):
        self.chunks: list[tuple[str, str, PageVersion, bool]] = []
        seen: dict[str, int] = {}
        for url, (pv, present) in sorted(self.pages.items()):
            if pv.kind == "junk":
                continue
            if pv._chunks is None:
                pv._chunks = make_chunks(pv)
            for heading, txt in pv._chunks:
                key = txt[:300]
                rank = (present, KIND_PRIOR[pv.kind], pv.pub or date(2030, 1, 1))
                if key in seen:
                    j = seen[key]
                    _, _, opv, opres = self.chunks[j]
                    orank = (opres, KIND_PRIOR[opv.kind], opv.pub or date(2030, 1, 1))
                    if rank > orank:
                        self.chunks[j] = (heading, txt, pv, present)
                    continue
                seen[key] = len(self.chunks)
                self.chunks.append((heading, txt, pv, present))
        docs = []
        for heading, txt, pv, _ in self.chunks:
            docs.append(content_tokens(txt) + content_tokens(heading) + content_tokens(pv.title))
        self.bm25 = BM25(docs)
        self.vocab = set(self.bm25.post)

    # ---- links ----
    def _build_links(self):
        self.links: dict[str, Counter] = defaultdict(Counter)   # url -> link-text token counts
        self.link_freq: Counter = Counter()
        for url, (pv, present) in self.pages.items():
            if not present or pv.kind in ("junk", "listing"):
                continue
            for m in MD_LINK.finditer(pv.md):
                target = m.group(2).split("?")[0]
                if "dupaco.com" not in target or target.endswith((".png", ".jpg", ".svg", ".jpeg")):
                    continue
                self.link_freq[target] += 1
                self.links[target].update(content_tokens(m.group(1)))
        for url in self.present:
            pv = self.pages[url][0]
            if pv.kind in ("product", "ratesheet"):
                self.links[url.split("?")[0]].update(content_tokens(pv.title) * 2)
                self.link_freq[url.split("?")[0]] += 1


# =======================================================================================
# The knowledge system
# =======================================================================================

UNPUBLISHED_PRODUCTS = {
    # product words that appear on the site but never with a published rate
    "construction", "arm", "adjustable", "fha", "va", "usda", "land", "lot", "bridge", "physician",
    "commercial", "agricultural", "farm", "sba", "jumbo-mortgage", "balloon", "reverse",
}

REFUSE_TEXT = ("I don't know: the website doesn't list that, so I can't confirm it. "
               "Please contact Dupaco directly for a current quote.")


class TimeAwareKnowledge:
    def __init__(self):
        self.snaps: list[tuple[str, date]] = []
        self.versions: dict[str, list[PageVersion]] = defaultdict(list)
        self.views: list[View] = []
        self._cache: dict = {}
        self._conflicts = None

    # ------------------------------------------------------------------ ingest
    def ingest(self, snapshot) -> None:
        idx = len(self.snaps)
        taken = parse_date(snapshot.taken_on) or date(2026, 1, 1)
        self.snaps.append((snapshot.snapshot_id, taken))
        present = set()
        for p in snapshot.pages:
            url = p.url
            if url in present:
                continue
            present.add(url)
            prev = self.versions[url][-1] if self.versions[url] else None
            if prev is not None and prev.md == clean_markdown(p.markdown or ""):
                pv = prev  # unchanged page: reuse parsed structures
            else:
                pv = PageVersion(url, p.title, p.markdown, idx, snapshot.snapshot_id)
            self.versions[url].append(pv)
        pages = {}
        for url, vs in self.versions.items():
            pages[url] = (vs[-1], url in present)
        self.views.append(View(idx, taken, pages))
        self._cache.clear()
        self._conflicts = None

    def _view(self, as_of: Optional[str]) -> View:
        if not self.views:
            raise RuntimeError("no snapshots ingested")
        if not as_of:
            return self.views[-1]
        d = parse_date(str(as_of)[:10]) or parse_date(str(as_of))
        if d is None:
            return self.views[-1]
        best = self.views[0]
        for v in self.views:
            if v.taken_on <= d:
                best = v
        return best

    # ------------------------------------------------------------------ answer
    def answer(self, question: str, as_of: Optional[str] = None):
        key = (re.sub(r"\s+", " ", question.strip().lower()), as_of)
        if key in self._cache:
            return self._cache[key]
        view = self._view(as_of)
        try:
            ans = self._route(question, view, as_of)
        except Exception as e:  # never crash the run
            ans = Answer(text=REFUSE_TEXT, confident=False, conflict_note=f"internal error: {e}")
        self._cache[key] = ans
        return ans

    def _route(self, q: str, view: View, as_of):
        ql = q.lower()
        qt = content_tokens(q)
        # 1. explicit link / url questions
        if (re.search(r"\b(link|url|web ?address|website address|web page|webpage|page)\b", ql) or re.search(
                r"\b(website|site) (to|for)\b|\bwhat'?s the website\b|\bwhere (can|do|should) i (find|see|view|check|look)\b"
                r".*\bonline\b|\bwhere online\b", ql)) and not re.search(r"\brate\b.*\bwhat is\b", ql):
            a = self._answer_link(q, view)
            if a:
                return a
        # 2. hours
        if (re.search(r"\bhours?\b|\bopen (on|until|at)\b|\bwhat time\b|\bclose[sd]? at\b", ql) or (
                re.search(r"\blobb(y|ies)\b|\bdrive[- ]?(up|thru|through)s?\b", ql) and
                re.search(r"\bopen|\bclose|\bwhen\b|\btime", ql))) and not re.search(
                r"\bvolunteer|hours (per|a) (week|month)|24-hour atm", ql):
            a = self._answer_hours(q, view)
            if a:
                return a
        # 3. branch addresses
        if re.search(r"\baddress\b|\bwhere is\b|\blocated\b|\blocation of\b", ql) and re.search(
                r"branch|office|atm|location", ql):
            a = self._answer_address(q, view)
            if a:
                return a
        # 4. rates
        if self._is_rate_question(ql, qt):
            a = self._answer_rate(q, view, as_of)
            if a:
                return a
        # 4. join / sign up online questions are link questions in disguise
        if re.search(r"\b(join|become a member|open an account) online\b|\bhow (do|can) i join\b", ql):
            a = self._answer_link(q, view)
            if a:
                return a
        # 5. phone numbers
        if re.search(r"\bphone\b|\bcall\b|\btelephone\b|\bcontact number\b|\btoll[- ]free\b", ql):
            a = self._answer_phone(q, view)
            if a:
                return a
        # 6. minimum to open / earn (table columns)
        if re.search(r"\bminimum\b|\bmin\.? to\b|\bhow much (do i need|to open)\b", ql) and re.search(
                r"\bopen|deposit|earn|balance|start\b", ql):
            a = self._answer_minimum(q, view)
            if a:
                return a
        # 7. general passage retrieval
        return self._answer_text(q, view)

    PHONE_RE = re.compile(r"(?<!\d)(?:\(?\d{3}\)?[-. ]?)\d{3}[-.]\d{4}(?!\d)")

    def _answer_phone(self, q: str, view: View):
        generic = {"phone", "number", "call", "contact", "reach", "main", "general", "customer", "service",
                   "telephone", "toll", "free", "talk", "speak", "someone", "person", "question"}
        specific = [w for w in content_tokens(q) if w not in generic]
        if specific:
            return None  # department-specific: let passage search find the right number
        freq = Counter()
        where = {}
        for url, (pv, present) in view.pages.items():
            if not present or pv.kind in ("junk", "listing"):
                continue
            for m in self.PHONE_RE.finditer(pv.md):
                num = re.sub(r"\D", "", m.group(0))
                if len(num) == 10:
                    freq[num] += 1
                    where.setdefault(num, url)
        if not freq:
            return None
        num = freq.most_common(1)[0][0]
        pretty = f"{num[:3]}-{num[3:6]}-{num[6:]}"
        src = "https://www.dupaco.com/contact" if "https://www.dupaco.com/contact" in view.present else where[num]
        return Answer(text=f"You can reach Dupaco at {pretty}.", sources=[Source(url=src)])

    def _answer_minimum(self, q: str, view: View):
        ql = q.lower()
        col_pat = r"min\.? to earn" if re.search(r"\bearn\b", ql) else r"min\.? to open"
        qset = {w for w in content_tokens(q) if w not in RATE_WORDS | {"minimum", "min", "open", "deposit",
                                                                        "balance", "earn", "start", "need"}}
        if not qset:
            return None
        best = None
        for t in view.tables:
            ci = [i for i, h in enumerate(t.header) if re.search(col_pat, h, re.I)]
            if not ci:
                continue
            sc = self._score_table(t, qset, view, {})
            if best is None or sc > best[0]:
                best = (sc, t, ci[0])
        if not best or best[0] < 0.4:
            return None
        _, t, ci = best
        vals = [r[ci] for r in t.rows if ci < len(r) and r[ci] and r[ci] != "—"]
        if not vals:
            return None
        what = "open" if "open" in col_pat else "earn dividends"
        uniq = list(dict.fromkeys(vals))
        if len(uniq) == 1:
            text = f"The minimum to {what} a {self._product_name(t)} account is {uniq[0]}."
        else:
            text = f"{self._product_name(t)} minimum to {what}: " + "; ".join(
                f"{t.row_label(r)}: {r[ci]}" for r in t.rows if ci < len(r)) + "."
        return Answer(text=text, sources=[Source(url=t.page.url)])

    # ------------------------------------------------------------------ rate engine
    def _is_rate_question(self, ql: str, qt: list[str]) -> bool:
        if re.search(r"\d+\s*-?\s*(month|mo\b|year|yr)", ql) and re.search(
                r"certificate|\bcds?\b|special|\bira\b|loan|mortgage", ql):
            return True
        if re.search(r"\b(pay|pays|earn|earns|paying)\b", ql) and re.search(
                r"certificate|\bcds?\b|saving|\bira\b|money market|checking|account|hsa", ql) and not re.search(
                r"\bpay (off|back|my|for|bill)", ql):
            return True
        if re.search(r"\b(aprs?|apys?|dividends?|rates?|interest|yields?)\b", ql):
            if re.search(r"\b(what is|what's|what are|how much|tell me|give me|get me|provide|current|"
                         r"do any|does|is the|are the|rates?\b.*\?|know the|show me|list)\b", ql) or ql.endswith("?") \
                    or len(qt) < 12:
                return True
        return False

    def _score_table(self, t: Table, qset: set[str], view: View, qq: dict) -> float:
        if not qset:
            return 0.0
        hit = sum(view.tidf.get(w, 0) for w in qset if w in t.doc_tokens)
        cap_hit = sum(view.tidf.get(w, 0) for w in qset if w in t.cap_tokens)
        tot = sum(view.tidf.get(w, 0) for w in qset if w in view.table_vocab) or 1e-9
        s = (hit + 0.5 * cap_hit) / (1.5 * tot)
        # discriminator words in the table that the question did not ask for
        for d in PENALIZE:
            if (d in t.doc_tokens or (d == "business" and "business" in t.page.url)) and d not in qset:
                s *= 0.86
        s *= TABLE_PRIOR.get(t.page.kind, 0.5)
        if t.page.url not in view.present:
            s *= 0.6
        return s

    def _answer_rate(self, q: str, view: View, as_of):
        ql = q.lower()
        qq = question_quals(q)
        qset = {w for w in content_tokens(q) if w not in RATE_WORDS and not w.isdigit()}
        if "months" in qq:
            qset -= {"month", "year"}
        # domain hints
        if "mortgage" in qset or ("home" in qset and "loan" in qset and "equity" not in qset
                                  and "improvement" not in qset):
            qset |= {"fixed", "conventional", "mortgage"}
        if "card" in qset or "visa" in qset:
            qset |= {"visa", "card"}
        if "auto" in qset:
            qset |= {"auto", "model"}
        if "share" in qset and "certificate" in qset:
            qset.discard("share")
        if "checking" in qset:
            qset.discard("account")
        qset -= {"account", "loan"} if len(qset) > 2 else set()

        # 1. honest refusal for products the site never prices
        unpub = [w for w in qset if w in UNPUBLISHED_PRODUCTS]
        if re.search(r"\b\d+/\d+\b", ql) and re.search(r"arm|adjustable|mortgage", ql):
            unpub.append("arm")
        good_vocab = getattr(view, "_good_vocab", None)
        if good_vocab is None:
            good_vocab = set()
            for t in view.tables:
                if TABLE_PRIOR[t.page.kind] >= 0.9:
                    good_vocab |= t.doc_tokens
            view._good_vocab = good_vocab
        if unpub and not any(w in good_vocab for w in unpub if w not in {"arm", "adjustable"}):
            return Answer(text=REFUSE_TEXT.replace("that", "a published rate for that product"),
                          confident=False, sources=[])

        cands = []
        unmatched = []   # well-matching tables whose rows don't have the asked-for term/tier
        required = {w for w in qset if w in REQUIRED}
        qkind = None
        if re.search(r"\bloans?\b|mortgage|\baprs?\b|visa|credit card|heloc|borrow|financ", ql):
            qkind = "loan"
        elif re.search(r"\bapys?\b|dividend|saving|certificate|\bcds?\b|checking|money market|\bira\b|yield|hsa",
                       ql):
            qkind = "deposit"
        if "mortgage" in required and "refinance" in required:
            required.discard("mortgage")
        if "saving" in required and (required & {"health", "hsa", "bright", "launchpad", "holiday", "certificate",
                                                 "ira", "checking"}):
            required.discard("saving")
        for t in view.tables:
            s = self._score_table(t, qset, view, qq)
            if s <= 0.05:
                continue
            if qkind and self._table_kind(t) != qkind:
                continue
            if re.search(r"visa|card", (t.caption + " " + t.page.url).lower()) and not re.search(
                    r"visa|card|credit", ql):
                continue
            have = t.doc_tokens | ({"business"} if "business" in t.page.url else set())
            if "certificate" in required and "ira" in have:
                have = have | {"certificate"}
            if required - have:
                continue
            rows = [r for r in t.rows if row_matches(row_quals(t.row_label(r)), qq)]
            if "months" in qq and not any("months" in row_quals(t.row_label(r)) for r in t.rows):
                s *= 0.8   # the question names a term; this table has no terms
            rows = self._row_discriminate(t, rows, qset)
            rows = self._smallest_cap(t, rows, qq)
            if not rows:
                unmatched.append((s, t))
                continue
            cands.append((s, t, rows))
        if unmatched and "months" in qq:
            ub = max(unmatched, key=lambda x: x[0])
            cb = cands[0][0] if cands else 0.0
            if cands:
                cb = max(c[0] for c in cands)
            if ub[0] >= 0.6 and ub[0] > 1.25 * cb:
                term = qq["months"]
                tdesc = f"{term // 12}-year" if term % 12 == 0 and term >= 60 and "year" in ql else f"{term}-month"
                when = "" if not as_of else " on that date"
                last = self._last_seen_term(view, ub[1], term)
                if last:
                    return Answer(text=f"No, that is no longer available: the current website doesn't list a {tdesc} "
                                       f"option for {self._product_name(ub[1])} any more (it was last listed in the "
                                       f"{last} snapshot). Please contact Dupaco for current terms.",
                                  confident=False,
                                  sources=[Source(url=ub[1].page.url, snapshot_id=ub[1].page.snapshot_id)],
                                  conflict_note="term dropped from the rate sheet between snapshots")
                return Answer(text=f"I don't know of one: the website{when} doesn't list a {tdesc} option for "
                                   f"{self._product_name(ub[1])}, so I can't confirm a rate. Please contact Dupaco "
                                   f"for current terms.", confident=False,
                              sources=[Source(url=ub[1].page.url, snapshot_id=ub[1].page.snapshot_id)])
        if not cands:
            good = getattr(view, "_good_vocab", set())
            if (required and required <= good and qset - required - {"account", "loan", "card"} == set()) or (
                    required & {"jumbo", "business", "student", "equipment", "commercial", "construction", "land"}):
                return Answer(text=REFUSE_TEXT.replace("that", "a published rate for that product"),
                              confident=False, sources=[])
            return None
        cands.sort(key=lambda x: (-x[0], -(x[1].effective or date(1900, 1, 1)).toordinal(), x[1].page.url, x[1].pos))
        best = cands[0][0]
        # must explain most of the product words in the question
        prod_words = [w for w in qset if w in view.table_vocab]
        missing = [w for w in qset if w not in view.table_vocab and w in view.vocab and len(w) > 3
                   and w not in {"model", "fixed", "conventional", "special", "standard", "regular", "balance",
                                 "tier", "between", "term", "length", "length", "option", "type", "kind",
                                 "program", "new", "used", "sign", "open", "opening", "anyone", "member",
                                 "deposit", "minimum", "require", "required", "amount", "personal", "consumer",
                                 "people", "someone", "receive", "earned", "earning", "available", "different",
                                 "difference", "repeat", "general", "standard", "purchase", "buy", "buying",
                                 "vehicle", "auto", "month", "year", "advertised", "listed", "website", "site",
                                 "rn", "nowadays", "moment"}]
        if best < 0.3 or (missing and best < 0.55 and len(missing) >= len(prod_words)):
            return None  # not a clean rate question; let text search try

        has_qual = any(k in qq for k in ("months", "model", "amount"))
        list_mode = bool(re.search(r"\b(all|every|offer|options|kinds|types|list|each|compare|which)\b", ql)) or \
            bool(re.search(r"\b(cards|certificates|cds|accounts|loans)\b", ql)) or \
            (bool(re.search(r"\brates\b", ql)) and not has_qual)
        thresh = 0.7 if list_mode else 0.9
        chosen: list[tuple[float, Table, list]] = []
        for s, t, rows in cands:
            if s < thresh * best:
                break
            if any(self._same_product(t, c[1]) or t.signature == c[1].signature for c in chosen):
                continue
            chosen.append((s, t, rows))
            if len(chosen) >= (5 if list_mode else 2):
                break
        # For each chosen product, the freshest authoritative copy wins (stale calculators,
        # landing pages and old guidance pages lose to a newer rate sheet).
        upgraded = []
        for s, t, rows in chosen:
            best_t, best_rows = t, rows
            for s2, t2, rows2 in cands:
                if s2 < 0.5 * best or t2 is best_t or t2.page.url not in view.present:
                    continue
                if TABLE_PRIOR[t2.page.kind] < 0.9 or not self._same_product(t2, t):
                    continue
                if (t2.effective or date(1900, 1, 1)) > (best_t.effective or date(1900, 1, 1)):
                    best_t, best_rows = t2, rows2
            if any(best_t is u[1] or best_t.signature == u[1].signature for u in upgraded):
                continue
            upgraded.append((s, best_t, best_rows))
        chosen = upgraded
        # optional rows (intro offers, relief programs, mini-max) only when asked for
        trimmed = []
        for s, t, rows in chosen:
            keep = [r for r in rows if not ROW_OPTIONAL.search(t.row_label(r)) or ROW_OPTIONAL.search(ql)]
            trimmed.append((s, t, keep or rows))
        chosen = trimmed
        # freshest version of each chosen table (same rows on another page with a newer date)
        final = []
        for s, t, rows in chosen:
            fresher = self._freshest_copy(t, view)
            if fresher is not t:
                rows = [r for r in fresher.rows if row_matches(row_quals(fresher.row_label(r)), qq)] or rows
                t = fresher
            final.append((t, rows))

        # extreme (highest / lowest)
        if qq.get("extreme") and len(final) >= 1:
            t, rows = final[0]
            vals = []
            for r in rows:
                rr = t.row_rates(r)
                if rr:
                    v = norm_pct(rr[-1][1])
                    if v is not None:
                        vals.append((v, r))
            if vals:
                pick = max(vals, key=lambda x: x[0]) if qq["extreme"] == "max" else min(vals, key=lambda x: x[0])
                final = [(t, [pick[1]])]

        # yes/no "does any tier pay X%"
        prefix = ""
        if "pct" in qq and re.search(r"\b(do|does|is|are|any|can)\b", ql):
            target = round(qq["pct"], 4)
            hits = [(t, [r for r in rows if any(norm_pct(v) == target for _, v in t.row_rates(r))]) for t, rows in final]
            hits = [(t, rs) for t, rs in hits if rs]
            if hits:
                prefix = "Yes. "
                final = hits
            else:
                t, rows = final[0]
                tops = [norm_pct(v) for r in rows for _, v in t.row_rates(r) if norm_pct(v) is not None]
                if tops:
                    prefix = f"No. The {self._product_name(t)} tiers top out at {fmt_pct(max(tops))}. "
                final = final[:1]

        text, sources = self._render_tables(final, view, prefix)
        note = self._stale_note(final, view)
        return Answer(text=text, sources=sources, confident=True, conflict_note=note)

    @staticmethod
    def _table_kind(t: Table) -> str:
        txt = (t.caption + " " + " ".join(t.header) + " " + t.page.url).lower()
        if re.search(r"\bapr\b|annual percentage rate|loan|visa|card|mortgage|heloc|conventional", txt) and not \
                re.search(r"\bapy\b|annual percentage yield|dividend", " ".join(t.header).lower()):
            return "loan"
        return "deposit"

    def _last_seen_term(self, view: View, like: Table, months: int) -> Optional[str]:
        """Name of the most recent earlier snapshot whose version of this product listed the term."""
        fam = self._family(like)
        for v in reversed(self.views[:view.idx]):
            for t in v.tables:
                if TABLE_PRIOR[t.page.kind] < 0.9 or self._family(t) != fam:
                    continue
                if any(row_quals(t.row_label(r)).get("months") == months for r in t.rows):
                    return v.taken_on.strftime("%B")
        return None

    @staticmethod
    def _row_discriminate(t: Table, rows: list, qset: set) -> list:
        """If the question names an option that only some rows carry (Roth, jumbo, refinance...),
        keep just those rows."""
        if len(rows) < 2:
            return rows
        toks = [set(content_tokens(t.row_label(r))) for r in rows]
        for w in qset:
            if w in DISCRIM or w in {"roth", "traditional", "jumbo", "purchase", "refinance", "standard"}:
                have = [r for r, tk in zip(rows, toks) if w in tk]
                if have and len(have) < len(rows):
                    rows = have
                    toks = [set(content_tokens(t.row_label(r))) for r in rows]
        return rows

    @staticmethod
    def _smallest_cap(t: Table, rows: list, qq: dict) -> list:
        """'$300,000 or less' style tiers are nested caps: a $250k loan belongs to the smallest cap."""
        if "amount" not in qq or len(rows) < 2:
            return rows
        capped = [(row_quals(t.row_label(r)).get("amount_max"), r) for r in rows]
        caps = [c for c, _ in capped if c is not None]
        if not caps:
            return rows
        m = min(caps)
        return [r for c, r in capped if c is None or c == m]

    @staticmethod
    def _product_name(t: Table) -> str:
        parts = [p for p in t.caption.split(" / ") if p]
        if not parts:
            return t.page.title.split("|")[0].strip() or "rate table"
        if len(parts[-1].split()) <= 1 and len(parts) > 1:
            return f"{parts[-2]} - {parts[-1]}"
        return parts[-1]

    @staticmethod
    def _same_product(a: Table, b: Table) -> bool:
        da = (a.doc_tokens & DISCRIM) | ({"business"} if "business" in a.page.url else set())
        db = (b.doc_tokens & DISCRIM) | ({"business"} if "business" in b.page.url else set())
        if ("business" in da) != ("business" in db):
            return False
        ca, cb = a.cap_tokens - {"rate", "loan"}, b.cap_tokens - {"rate", "loan"}
        if not ca or not cb:
            return False
        jac = len(ca & cb) / len(ca | cb)
        core = {"business", "jumbo", "refinance", "roth", "traditional", "rewards", "platinum", "equity",
                "improvement", "boat", "atv", "special", "heloc", "purchase", "share", "checking", "money",
                "bright", "launchpad", "holiday", "health", "high", "investor", "name", "regular", "variable",
                "certificate", "ira"}
        if (da & core) != (db & core) and not (a.signature[1] and a.signature[1] == b.signature[1]):
            # allow same row-set (identical table on 2 pages) even with different captions
            lbl_a = {re.sub(r"[^a-z0-9]", "", a.row_label(r).lower()) for r in a.rows}
            lbl_b = {re.sub(r"[^a-z0-9]", "", b.row_label(r).lower()) for r in b.rows}
            return len(lbl_a & lbl_b) >= max(2, 0.6 * min(len(lbl_a), len(lbl_b)))
        return jac >= 0.34 or a.signature == b.signature

    def _freshest_copy(self, t: Table, view: View) -> Table:
        best = t
        for o in view.tables:
            if o is t or o.signature[1] != t.signature[1] or o.page.url not in view.present:
                continue
            if TABLE_PRIOR[o.page.kind] < 0.9:
                continue
            if (o.effective or date(1900, 1, 1)) > (best.effective or date(1900, 1, 1)):
                best = o
        return best

    def _render_tables(self, final, view: View, prefix: str):
        parts = []
        sources: list = []
        seen_src = set()

        def add_src(url, quote=None):
            k = url.split("#")[0].rstrip("/")
            if k in seen_src or len(sources) >= 6:
                return
            seen_src.add(k)
            sources.append(Source(url=url, snapshot_id=view.pages[url][0].snapshot_id if url in view.pages else None,
                                  quote=quote))

        for t, rows in final:
            name = self._product_name(t)
            lines = []
            for r in rows[:16]:
                lbl = t.row_label(r)
                rates = t.row_rates(r)
                if not rates:
                    continue
                rtxt = ", ".join(self._fmt_rate(h, v) for h, v in rates)
                lines.append(f"{lbl}: {rtxt}" if lbl and lbl.lower() != name.lower() else rtxt)
            if not lines:
                continue
            eff = f" (rates effective {t.effective.strftime('%m/%d/%Y')})" if t.effective else ""
            head = name if name else "Rates"
            if len(lines) == 1:
                parts.append(f"{head} - {lines[0]}{eff}.")
            else:
                parts.append(f"{head}{eff}: " + "; ".join(lines) + ".")
            add_src(t.page.url, lines[0][:200])
            # the same table on the product page (rate sheets link to it)
            if t.details_url:
                add_src(t.details_url.split("?")[0])
            for o in view.tables:
                if o.signature == t.signature and o.page.url in view.present and o.effective == t.effective \
                        and TABLE_PRIOR[o.page.kind] >= 0.9:
                    add_src(o.page.url)
        text = prefix + " ".join(parts)
        return text.strip(), sources

    @staticmethod
    def _fmt_rate(h: str, v: str) -> str:
        hl = h.lower()
        v = v.strip()
        if "apy" in hl or "annual percentage yield" in hl:
            return v if "apy" in v.lower() else f"{v} APY"
        if "apr" in hl or "annual percentage rate" in hl:
            return v if "apr" in v.lower() else f"{v} APR"
        if "dividend" in hl:
            return f"{v} dividend rate"
        if hl.strip() == "rate":
            return f"{v} rate"
        return v

    def _stale_note(self, final, view: View) -> Optional[str]:
        notes = []
        for t, rows in final:
            for r in rows[:6]:
                lbl = re.sub(r"[^a-z0-9]", "", t.row_label(r).lower())
                cur = {norm_pct(v) for _, v in t.row_rates(r)}
                for o in view.tables:
                    if o is t or not self._same_product(o, t):
                        continue
                    for r2 in o.rows:
                        if re.sub(r"[^a-z0-9]", "", o.row_label(r2).lower()) == lbl:
                            other = {norm_pct(v) for _, v in o.row_rates(r2)}
                            diff = other - cur - {None}
                            if diff and len(notes) < 3:
                                notes.append(f"{o.page.url} still shows {', '.join(fmt_pct(x) for x in sorted(diff))} "
                                             f"for '{t.row_label(r)}' (effective "
                                             f"{o.effective.isoformat() if o.effective else 'undated'}); "
                                             f"used the newer/authoritative {t.page.url}.")
        return " ".join(notes) or None

    # ------------------------------------------------------------------ hours
    HOURS_BLOCK = re.compile(r"\*\*(Lobby|Drive[- ]?up|Drive[- ]?thru)\*\*\s*\n?(.*?)(?=\*\*|ABOUT DUPACO|$)", re.I | re.S)

    def _collect_hours(self, view: View):
        if hasattr(view, "_hours"):
            return view._hours
        recs = []  # (branch_text, kind, schedule_lines, url, pub)
        for url, (pv, present) in view.pages.items():
            if pv.kind not in ("news", "blog", "product"):
                continue
            md = pv.md
            if "Lobby" not in md and "Drive-up" not in md:
                continue
            for m in self.HOURS_BLOCK.finditer(md):
                kind = "lobby" if m.group(1).lower().startswith("lobby") else "drive"
                body = m.group(2)
                sched = []
                for ln in re.split(r"\n+", body):
                    ln = ln.strip()
                    if re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*", ln) and ":" in ln:
                        ln = re.sub(r"\s*[–—-]\s*", "–", re.sub(r"\s+", " ", ln))
                        ln = re.sub(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)–", r"\1–", ln)
                        sched.append(ln)
                    elif sched and not ln:
                        continue
                    elif sched:
                        break
                if sched:
                    ctx = md[max(0, m.start() - 600):m.start()]
                    recs.append((ctx, kind, tuple(sched), url, pv.pub))
        view._hours = recs
        return recs

    def _answer_hours(self, q: str, view: View):
        ql = q.lower()
        recs = self._collect_hours(view)
        if not recs:
            return None
        kind = "drive" if re.search(r"drive|drive-?(up|thru|through)", ql) else "lobby"
        # specific branch?
        m = re.search(r"\b(?:for|at|of|in) (?:the |your |dupaco'?s? )?([a-z0-9.'-]+(?: [a-z0-9.'-]+){0,3}?) "
                      r"(?:branch|office|location)", ql)
        branch = None
        if m:
            branch = m.group(1).strip()
            branch = re.sub(r"^(dupaco'?s?|your) ", "", branch)
            if branch in ("general", "main", "typical", "usual", "each", "every", "all", "any", "the", "your",
                          "dupaco"):
                branch = None
        if branch:
            btoks = [w for w in content_tokens(branch) if w not in {"avenue", "street", "st", "ave", "road", "rd",
                                                                    "drive", "dr", "lane", "ln"}]
            hits = [r for r in recs if r[1] == kind and btoks and all(w in content_tokens(r[0]) for w in btoks)]
            if not hits:
                # is this branch mentioned anywhere at all?
                known = any(all(w in self._page_tokens(pv) for w in btoks) and pv.kind != "junk"
                            for pv, _ in view.pages.values()) if btoks else False
                if True:
                    if known:
                        return Answer(text=f"I don't know the hours for the {branch.title()} branch: the website "
                                           f"doesn't list them. Please contact Dupaco or check the branch locator at "
                                           f"https://www.dupaco.com/contact.", confident=False,
                                      sources=[Source(url="https://www.dupaco.com/contact")])
                    return Answer(text=f"I don't know: I can't find a Dupaco branch called '{branch.title()}' on the "
                                       f"website, so I can't confirm its hours. Please contact Dupaco or use the "
                                       f"branch locator at https://www.dupaco.com/contact.",
                                  confident=False, sources=[Source(url="https://www.dupaco.com/contact")])
                return None
            hits.sort(key=lambda r: r[4] or date(1900, 1, 1), reverse=True)
            r = hits[0]
            label = "Lobby" if kind == "lobby" else "Drive-up"
            return Answer(text=f"{branch.title()} branch {label.lower()} hours: " + "; ".join(r[2]) + ".",
                          sources=[Source(url=r[3])])
        lob = Counter(r[2] for r in recs if r[1] == "lobby")
        drv = Counter(r[2] for r in recs if r[1] == "drive")
        urls = [r[3] for r in sorted(recs, key=lambda r: r[4] or date(1900, 1, 1), reverse=True)]

        def summarize(counter):
            items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            return [list(s) for s, _ in items[:3]]

        srcs = [Source(url="https://www.dupaco.com/contact"), Source(url="https://www.dupaco.com/")]
        main_lobby = summarize(lob)[0] if lob else None
        main_drive = summarize(drv)[0] if drv else None
        for r in sorted(recs, key=lambda r: r[4] or date(1900, 1, 1), reverse=True):
            target = main_drive if kind == "drive" else main_lobby
            if r[1] == kind and target and list(r[2]) == target and all(x.url != r[3] for x in srcs):
                srcs.append(Source(url=r[3]))
            if len(srcs) >= 6:
                break
        if kind == "drive":
            sch = summarize(drv)
            if not sch:
                return None
            main = "; ".join(sch[0])
            sats = sorted({ln for s in sch for ln in s if ln.lower().startswith("sat")})
            alt = (" Saturday drive-up hours vary by branch (" + " or ".join(
                ln.split(":", 1)[1].strip() for ln in sats) + ").") if len(sats) > 1 else ""
            text = (f"Drive-up hours at most Dupaco branches: {main}.{alt} "
                    f"Hours can differ by branch, so tell me which branch and I'll check that one.")
            return Answer(text=text, sources=srcs)
        sch = summarize(lob)
        if not sch:
            return None
        main = "; ".join(sch[0])
        others = [" ; ".join(s) for s in sch[1:]]
        text = f"Lobby hours vary by branch. The most common schedule is {main}."
        if others:
            text += " Other branches use: " + " | ".join(o.replace(" ; ", "; ") for o in others) + "."
        if re.search(r"\bsame\b|\ball\b|\bevery\b|\bdiffer", ql):
            text = "No, lobby hours are not the same at every branch. " + text
        text += " Which branch are you interested in?"
        return Answer(text=text, sources=srcs)

    @staticmethod
    def _page_tokens(pv: PageVersion) -> set:
        if not hasattr(pv, "_tokset_cache"):
            pass
        return set(content_tokens(pv.md[:20000] + " " + pv.title))


    # ------------------------------------------------------------------ addresses
    ADDR_RE = re.compile(r"^\s*(\d{1,6}\s+[^,\n]{3,60}),\s*([A-Za-z .'-]+),\s*(IA|IL|WI|MN|Iowa|Illinois|Wisconsin)"
                         r",?\s*(\d{5})", re.M)

    def _answer_address(self, q: str, view: View):
        ql = q.lower()
        m = re.search(r"\b(?:of|for|is|at|in) (?:the |your |dupaco'?s? )?([a-z0-9.'-]+(?: [a-z0-9.'-]+){0,3}?) "
                      r"(?:branch|office|location|atm)", ql)
        name = m.group(1).strip() if m else ""
        name = re.sub(r"^(dupaco'?s?|your|the) ", "", name)
        ntoks = [w for w in content_tokens(name) if w not in {"dupaco", "main"}]
        if not ntoks:
            return None
        best = None
        for url, (pv, present) in view.pages.items():
            if not present or pv.kind in ("junk", "listing"):
                continue
            for am in self.ADDR_RE.finditer(pv.md):
                ctx = pv.md[max(0, am.start() - 120):am.end()]
                label = ctx.lower()
                atoks = set(content_tokens(am.group(0) + " " + ctx[-200:]))
                if not all(w in atoks for w in ntoks):
                    continue
                s = 1.0
                if "branch" not in label:
                    continue
                if "dupaco branch" in label:
                    s *= 3
                if "surcharge-free atm" in label or "shared branch" in label:
                    s *= 0.3
                if url.rstrip("/").endswith("/contact"):
                    s *= 1.5
                if best is None or s > best[0]:
                    best = (s, am.group(0).strip(), url)
        if not best:
            return None
        return Answer(text=f"The {name.title()} branch is at {best[1]}.", sources=[Source(url=best[2])])

    # ------------------------------------------------------------------ links
    def _answer_link(self, q: str, view: View):
        qset = {w for w in content_tokens(q) if w not in {"link", "url", "use", "see", "page", "website", "site",
                                                          "web", "address", "online", "find", "go"}}
        if not qset:
            return None
        if "certificate" in qset or "saving" in qset:
            qset |= {"saving", "certificate"}
        if re.search(r"locator|locations?|branch(es)? and atm|find a branch|atm", q.lower()):
            qset |= {"location", "contact", "branch", "atm", "find"}
        if re.search(r"\bjoin|member", q.lower()):
            qset |= {"join", "member", "become"}
        best = None
        for url, toks in view.links.items():
            if url.endswith((".pdf", ".xlsx", ".xls")):
                continue
            ut = set(content_tokens(re.sub(r"[/_.:-]", " ", url.replace("https://", "").replace("www.dupaco.com", ""))))
            s = 0.0
            tot = sum(toks.values()) or 1
            for w in qset:
                s += 2.0 * toks.get(w, 0) / tot + (1.0 if w in ut else 0.0)
            s *= 1 + math.log(1 + view.link_freq[url]) / 4
            if re.search(r"\bonline\b|\bapply\b|\bsign up\b|\bapplication\b|\bjoin\b", q.lower()) and \
                    not url.startswith("https://www.dupaco.com") and url.count("/") <= 3:
                s *= 1.6
            if "#rates" in url or url.endswith("#rates"):
                s *= 1.3 if re.search(r"\brates?\b", q.lower()) else 1.0
            if s > 0 and (best is None or s > best[0]):
                best = (s, url)
        if not best or best[0] < 1.0:
            return None
        url = best[1]
        if "rate" in q.lower() and "#" not in url and (url + "#rates") in view.links:
            url = url + "#rates"
        return Answer(text=f"Use {url}", sources=[Source(url=url)])

    # ------------------------------------------------------------------ text
    @staticmethod
    def _answer_type(ql: str) -> Optional[str]:
        if re.search(r"^\s*how many\b|\bnumber of\b|\bhow many\b", ql):
            return "count"
        if re.search(r"\bwhen\b|\bwhat date\b|\bstill open\b|\bclos(e|ed|ing|ure)\b|\bdeadline\b|\bexpire", ql):
            return "date"
        if re.search(r"\bhow much\b|\bfee\b|\bcost\b|\bcharge\b|\bminimum\b|\bprice\b|\bup to\b", ql):
            return "money"
        if re.search(r"\bphone\b|\bcall\b|\bnumber to\b|\bcontact\b", ql):
            return "phone"
        if re.search(r"\bwhere(?:'s| is| are)\b|\baddress\b|\blocated\b", ql):
            return "where"
        return None

    TYPE_RE = {
        "count": re.compile(r"\b\d[\d,]*\+?\s+(?:[A-Za-z’'-]+\s){0,3}"),
        "date": re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|"
                           r"December|Jan\.|Feb\.|Aug\.|Sept?\.|Oct\.|Nov\.|Dec\.)\s+\d{1,2}(,\s*\d{4})?"),
        "money": re.compile(r"\$\s?\d|\d%|no (?:charge|cost|fee)", re.I),
        "phone": re.compile(r"\b\d{3}[-.)\s]+\d{3}[-.]\d{4}\b"),
        "where": re.compile(r"\b\d{2,6}\s+(?:[NSEW]\.?\s+)?[A-Z][A-Za-z.]*(?:\s+[A-Z][A-Za-z.]*){0,3}\s+(?:St|Street|Ave|Avenue|"
                            r"Rd|Road|Dr|Drive|Ln|Lane|Hwy|Highway|Blvd|Plaza|Ct|Court|Way|Pkwy|Parkway|Circle|Pl)\b\.?"),
    }

    def _typed(self, atype: str, txt: str):
        if atype == "where":
            if not re.search(r"branch|dupaco", txt, re.I):
                return None
            for m in self.TYPE_RE["where"].finditer(txt):
                if not re.search(r"ATM|Shared Branch", txt[max(0, m.start() - 40):m.start()]):
                    return m
            return None
        return self.TYPE_RE[atype].search(txt)

    def _answer_text(self, q: str, view: View):
        ql = q.lower()
        qt = content_tokens(q)
        if not qt:
            return Answer(text=REFUSE_TEXT, confident=False)
        atype = self._answer_type(ql)
        scores = view.bm25.search(qt)
        if not scores:
            return Answer(text=REFUSE_TEXT, confident=False)
        newest = view.taken_on
        qset = set(qt)
        noun = None
        if atype == "count":
            m = re.search(r"how many ([a-z]+)", ql)
            noun = stem(m.group(1)) if m else None
        ranked = []
        for i, s in scores.items():
            heading, txt, pv, present = view.chunks[i]
            prior = KIND_PRIOR[pv.kind] * (1.0 if present else 0.8)
            if pv.pub:
                age = max(0.0, (newest - pv.pub).days / 365.0)
                prior *= 1.0 / (1.0 + (0.25 if atype == "count" else 0.06) * age)
            if atype and atype in self.TYPE_RE:
                if atype == "count" and noun:
                    ok = count_hit(noun, txt)
                else:
                    ok = self._typed(atype, txt)
                prior *= 1.35 if ok else 0.8
            if re.match(r"^(for more information|learn more|visit|click|read more|keep reading|see )", txt, re.I) \
                    and len(txt) < 200:
                prior *= 0.5
            if txt.count(" | ") >= 2 and pv.kind != "pdf":
                prior *= 0.6
            if atype == "money" and len(txt) < 200 and self._typed("money", txt):
                prior *= 1.3
            ranked.append((s * prior, i))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        if atype in ("count", "date", "where"):
            typed = []
            pool = ranked[:300]
            if atype == "count" and noun and noun in view.bm25.post:
                have = {i for _, i in pool}
                pool = pool + [(scores.get(i, 0.0), i) for i, _ in view.bm25.post[noun] if i not in have]
            for sc_, i in pool:
                txt_ = view.chunks[i][1]
                if atype == "count" and noun:
                    ok = count_hit(noun, txt_)
                else:
                    ok = self._typed(atype, txt_)
                if ok and (atype == "count" or sc_ >= 0.3 * ranked[0][0]):
                    typed.append((sc_, i))
            if typed and atype == "count":
                mx = max(x[0] for x in typed)
                typed = [x for x in typed if x[0] >= 0.5 * mx]
                # counts change over time: the newest dated statement about Dupaco wins
                def ckey(x):
                    _, txt_, pv_, pres_ = view.chunks[x[1]]
                    d = pv_.pub or date(1900, 1, 1)
                    return ("dupaco" in txt_.lower(), pres_, d, x[0])
                typed.sort(key=ckey, reverse=True)
            if typed:
                ranked = typed + [r for r in ranked if r not in typed]
        top_s, top_i = ranked[0]
        heading, txt, pv, present = view.chunks[top_i]
        ctoks = set(content_tokens(txt) + content_tokens(heading) + content_tokens(pv.title))
        idf = view.bm25.idf
        cov = sum(idf.get(w, 0) for w in qset if w in ctoks) / (sum(idf.get(w, 0) for w in qset if w in idf) or 1e-9)
        known = [w for w in qset if w in idf]
        rare = max(known, key=lambda w: idf[w]) if known else None
        if cov < 0.34 or (rare and rare not in ctoks and cov < 0.5 and len(known) <= 3):
            return Answer(text=REFUSE_TEXT, confident=False)
        best_sents = self._best_sentences(txt, qset, idf, atype, heading, noun, question_quals(q).get("months"))
        srcs = [Source(url=pv.url, snapshot_id=pv.snapshot_id, quote=best_sents[:200])]
        for s2, i2 in ranked[1:4]:
            h2, t2, pv2, _ = view.chunks[i2]
            if pv2.url != pv.url and s2 >= 0.97 * top_s:
                srcs.append(Source(url=pv2.url, snapshot_id=pv2.snapshot_id))
                break
        return Answer(text=best_sents, sources=srcs, confident=True)

    def _best_sentences(self, txt: str, qset: set, idf: dict, atype=None, heading="", noun=None,
                        months=None) -> str:
        sents = [s.strip(" -*") for s in SENT_SPLIT.split(txt) if s.strip(" -*")]
        if months:
            def terms(x):
                return {row_quals(m.group(0)).get("months") for m in
                        re.finditer(r"\d+\s*-?\s*(?:month|year)", x, re.I)} - {None}
            right = [x for x in sents if months in terms(x)]
            if right:
                sents = [x for x in sents if not terms(x) or months in terms(x)]
        if len(sents) == 1:
            return sents[0][:600]
        if len(sents) <= 1:
            return txt[:600]
        sc = []
        for k, s in enumerate(sents):
            st = set(content_tokens(s))
            v = sum(idf.get(w, 0) for w in qset if w in st)
            if s.endswith("?"):
                v *= 0.3
            if re.match(r"^(for more information|learn more|visit|click|read more|keep reading|see )", s, re.I):
                v *= 0.3
            sc.append((v, k))
        # FAQ pattern: a matching question line is answered by the sentence that follows it
        for k, s in enumerate(sents[:-1]):
            if s.endswith("?") and not sents[k + 1].endswith("?"):
                raw = sc[k][0] / 0.3
                if raw > sc[k + 1][0]:
                    sc[k + 1] = (max(sc[k + 1][0], 0.95 * raw), k + 1)
        maxv = max(v for v, _ in sc) or 1.0
        for j, (v, k) in enumerate(sc):
            s = sents[k]
            typed = False
            if atype == "count" and noun:
                typed = bool(count_hit(noun, s))
            elif atype and atype in self.TYPE_RE:
                typed = bool(self._typed(atype, s))
            if typed and (v > 0 or atype == "count"):
                sc[j] = (v * 1.5 + (1.2 if atype == "count" else 0.8) * maxv, k)
        sc.sort(key=lambda x: (-x[0], x[1]))
        k = sc[0][1]
        picked = {k}
        # second-best sentence when it adds something (keeps answers short but complete)
        if len(sc) > 1 and sc[1][0] > 0.5 * sc[0][0] and abs(sc[1][1] - k) <= 3:
            picked.add(sc[1][1])
        # a definition-style answer usually continues in the next sentence
        if k + 1 < len(sents) and len(sents[k]) < 220 and not sents[k + 1].endswith("?") and len(picked) == 1:
            picked.add(k + 1)
        if k > 0 and len(sents[k]) < 50 and sents[k - 1].endswith(":") is False and len(picked) < 3:
            picked.add(k - 1)
        out = " ".join(sents[i] for i in sorted(picked))
        return out[:700]

    # ------------------------------------------------------------------ conflicts
    def conflicts(self):
        if self._conflicts is not None:
            return self._conflicts
        latest = self.views[-1]
        recs: dict[tuple, dict] = {}
        for v in self.views:
            for t in v.tables:
                if t.page.kind in ("junk", "listing"):
                    continue
                fam = self._family(t)
                if not fam:
                    continue
                col = self._main_col(t)
                if col is None:
                    continue
                tiers = self._tier_keys(t)
                for r, tier in zip(t.rows, tiers):
                    if col >= len(r):
                        continue
                    val = norm_pct(r[col])
                    if val is None or tier is None:
                        continue
                    key = (fam,) + tier
                    rec = recs.setdefault(key, {"label": t.row_label(r), "fam": fam, "tier": tier, "vals": {},
                                                "col": t.header[col]})
                    eff = t.effective or date(1900, 1, 1)
                    auth = TABLE_PRIOR[t.page.kind] * (1.0 if t.page.url in v.present else 0.5)
                    live = v is latest and t.page.url in latest.present
                    cur = rec["vals"].get(val)
                    cand = {"eff": eff, "snap": v.idx, "auth": auth, "url": t.page.url, "live": cur["live"] if cur
                            else False}
                    cand["live"] = cand["live"] or live
                    if cur is None or (eff, v.idx, auth) > (cur["eff"], cur["snap"], cur["auth"]):
                        rec["vals"][val] = cand
                    else:
                        cur["live"] = cur["live"] or live
        out = []
        for key, rec in recs.items():
            vals = rec["vals"]
            if len(vals) < 2:
                continue
            # the right value: freshest effective date from an authoritative, still-live page
            def rank(item):
                v_, c_ = item
                return (c_["live"] and c_["auth"] >= 0.9, c_["eff"], c_["auth"], c_["snap"])
            ordered = sorted(vals.items(), key=rank, reverse=True)
            win_v, win_c = ordered[0]
            live_vals = [v_ for v_, c_ in vals.items() if c_["live"]]
            kind = "stale page still live" if len(live_vals) > 1 else "changed between snapshots"
            losers = ", ".join(f"{fmt_pct(v_)} ({self._short(c_['url'])}, eff. "
                               f"{c_['eff'].isoformat() if c_['eff'].year > 1900 else 'undated'})"
                               for v_, c_ in ordered[1:4])
            reason = (f"{kind}: {fmt_pct(win_v)} comes from the newest effective date on "
                      f"{self._short(win_c['url'])} (eff. {win_c['eff'].isoformat() if win_c['eff'].year > 1900 else 'undated'}); "
                      f"older/stale values: {losers}")
            pri = (len(live_vals) > 1, max(c_["eff"] for c_ in vals.values()))
            out.append((pri, Conflict(topic=self._topic(rec), values=[fmt_pct(v_) for v_ in sorted(vals)],
                                      sources=[Source(url=c_["url"]) for _, c_ in sorted(vals.items())],
                                      resolution=fmt_pct(win_v), reason=reason)))
        out.sort(key=lambda x: x[0], reverse=True)
        # a business table that mirrors the personal one adds nothing new
        personal = {(c.topic.replace("business ", ""), tuple(c.values)) for _, c in out if "business " not in c.topic}
        result = [c for _, c in out if not ("business " in c.topic and
                                            (c.topic.replace("business ", ""), tuple(c.values)) in personal)]
        result.extend(self._text_conflicts(latest))
        self._conflicts = result
        return result

    @staticmethod
    def _short(url: str) -> str:
        return re.sub(r"^https?://(www\.)?", "", url)

    @staticmethod
    def _family(t: Table) -> Optional[str]:
        txt = (t.caption + " " + " ".join(t.row_label(r) for r in t.rows[:3]) + " " + t.page.url).lower()
        biz = "business " if ("business" in txt) else ""
        if re.search(r"model year", txt):
            if re.search(r"atv|snowmobile|jet ski", txt):
                return biz + "atv/snowmobile loan"
            return biz + "auto loan"
        if re.search(r"conventional|home loan|mortgage", txt):
            if "refinance" in txt or "refi" in txt:
                return biz + "mortgage refinance"
            return biz + "mortgage purchase"
        if re.search(r"boat|rv loan", txt):
            return biz + "boat & rv loan"
        if "heloc" in txt or "line of credit" in txt:
            return biz + "heloc"
        if "home equity" in txt:
            return biz + "home equity loan"
        if "home improvement" in txt:
            return biz + "home improvement loan"
        if "personal loan" in txt:
            return biz + "personal loan"
        if re.search(r"visa|credit card", txt):
            if "rewards" in txt:
                return biz + "rewards visa"
            return biz + "platinum visa"
        if re.search(r"\bira\b|roth|traditional", txt):
            return None  # IRA certificates mirror the certificate tables
        if "certificate" in txt or re.search(r"\b\d+ month\b", txt):
            if "special" in txt:
                return biz + "special certificate"
            return biz + "certificate"
        for name in ("share savings", "bright savings", "you name it", "holiday club", "launchpad",
                     "investor's choice", "high-yield", "health savings", "1st rate checking", "free checking",
                     "achieve"):
            if name in txt:
                return biz + name
        return None

    @staticmethod
    def _main_col(t: Table) -> Optional[int]:
        for i in t.rate_cols:
            if re.search(r"apy|annual percentage yield", t.header[i], re.I):
                return i
        for i in t.rate_cols:
            if re.search(r"\bapr\b|annual percentage rate", t.header[i], re.I):
                return i
        return t.rate_cols[0] if t.rate_cols else None

    @staticmethod
    def _tier_keys(t: Table) -> list:
        """Semantic tier key per row, stable across label wording (2023-26 vs 2024-27 model years)."""
        quals = [row_quals(t.row_label(r)) for r in t.rows]
        starts = sorted({q["model"][0] for q in quals if "model" in q}, reverse=True)
        keys = []
        for r, q in zip(t.rows, quals):
            lbl = t.row_label(r).lower()
            jumbo = "jumbo" in lbl
            intro = bool(re.search(r"intro|transfer", lbl))
            k = [q.get("months")]
            if "model" in q:
                k.append(("modeltier", starts.index(q["model"][0])))
            if "amount" in q:
                k.append(("amt", q["amount"][0]))
            if "amount_max" in q:
                k.append(("cap", q["amount_max"]))
            k += [jumbo, intro]
            if k[0] is None and len(k) == 3 and not jumbo and not intro:
                k.append(("label", re.sub(r"[^a-z]", "", lbl.split(",")[0])[:20]))
            keys.append(tuple(k))
        return keys

    @staticmethod
    def _topic(rec) -> str:
        fam, tier, label = rec["fam"], rec["tier"], rec["label"]
        months = tier[0] if tier else None
        if re.search(r"loan|mortgage|heloc|visa", fam):
            col = "APR"
        else:
            col = "APY" if re.search(r"apy|yield", rec["col"], re.I) else "rate"
        term = ""
        if isinstance(months, int):
            term = (f"{months // 12}-year " if fam.startswith(("mortgage", "business mortgage")) and months % 12 == 0
                    else f"{months}-month ")
        return f"{term}{fam} {col} ({label})".strip()

    def _text_conflicts(self, view: View):
        out = []
        # branch counts in "About Dupaco" boilerplate
        counts: dict[int, tuple] = {}
        for url, (pv, present) in view.pages.items():
            for m in re.finditer(r"(\d{2}) branch(?: office)? locations", pv.md):
                n = int(m.group(1))
                d = pv.pub or date(1900, 1, 1)
                if n not in counts or d > counts[n][0]:
                    counts[n] = (d, url)
        if len(counts) > 1:
            win = max(counts.items(), key=lambda kv: kv[1][0])[0]
            out.append(Conflict(topic="Number of Dupaco branch office locations",
                                values=[f"{n} branch office locations" for n in sorted(counts)],
                                sources=[Source(url=counts[n][1]) for n in sorted(counts)],
                                resolution=f"{win} branch office locations",
                                reason="value from the most recently published press release"))
        # Forbes recognitions
        forbes = {}
        for url, (pv, present) in view.pages.items():
            for m in re.finditer(r"(\w+) time Forbes recognized Dupaco", pv.md):
                d = pv.pub or date(1900, 1, 1)
                w = m.group(1).lower()
                if w not in forbes or d > forbes[w][0]:
                    forbes[w] = (d, url)
        if len(forbes) > 1:
            win = max(forbes.items(), key=lambda kv: kv[1][0])[0]
            out.append(Conflict(topic="How many times Forbes recognized Dupaco",
                                values=sorted(forbes), sources=[Source(url=forbes[w][1]) for w in sorted(forbes)],
                                resolution=win, reason="most recent Forbes press release"))
        return out


def build():
    return TimeAwareKnowledge()
