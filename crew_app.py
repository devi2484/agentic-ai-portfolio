import os
import json
import time
import re
import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from urllib.parse import urlparse

# ==========================================
# 1. SETUP
# ==========================================
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY") or st.secrets.get("GROQ_KEY", "")
llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.1-8b-instant", temperature=0.1)

st.set_page_config(page_title="Strategic Intelligence Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ Strategic Intelligence Engine")
st.markdown("**Evidence-Based Decision Support System** · Strict Traceability · Scored Options · Validated Reasoning")
st.divider()

# ==========================================
# 2. TRUST & SCORING
# ==========================================

# PRIMARY SOURCES — concall transcripts, annual reports, regulatory filings
# These carry the highest trust: direct company disclosures to regulators or investors
PRIMARY_SOURCE_DOMAINS = [
    # Indian stock exchanges & regulatory filings
    "bseindia.com", "nseindia.com", "sebi.gov.in",
    # Company investor relations pages (matched via path/URL heuristics below)
    "ir.", "investor.", "investors.",
    # Concall transcript aggregators
    "tickertape.in", "screener.in", "trendlyne.com",
    "stockanalysis.com", "simplywall.st",
    # Regulatory / exchange filings
    "sec.gov", "edgaronline.com", "mca.gov.in",
    # Annual report / earnings release hosts
    "annualreports.com", "iexchange.in",
]

# Paths/keywords in URLs that indicate primary source documents
PRIMARY_SOURCE_URL_PATTERNS = [
    "/annual-report", "/investor-presentation", "/earnings-call",
    "/concall", "/con-call", "/transcript", "/results-presentation",
    "/quarterly-results", "/agm", "/investor-day", "/earnings-release",
    "annualreport", "investorpresentation", "earningscall", "concall",
    "q1results", "q2results", "q3results", "q4results",
    "fy20", "fy21", "fy22", "fy23", "fy24", "fy25",
    "/filing/", "/disclosures/", "/pdf/", "/uploads/announcements/",
]

HIGH_TRUST_DOMAINS = [
    "reuters.com", "bloomberg.com", "cnbc.com", "wsj.com", "ft.com",
    "moneycontrol.com", "economictimes.indiatimes.com", "livemint.com",
    "businessstandard.com", "thehindubusinessline.com", "financialexpress.com",
    "rbi.org.in", "hbr.org", "mckinsey.com", "bain.com", "bcg.com",
    "economist.com", "statista.com", "nyse.com", "nasdaq.com",
    # Concall/AR aggregators treated as high trust
    "tickertape.in", "screener.in", "trendlyne.com", "stockanalysis.com",
]

MEDIUM_TRUST_DOMAINS = [
    "techcrunch.com", "forbes.com", "inc42.com", "entrackr.com",
    "yourstory.com", "themorningcontext.com", "restofworld.org", "fortune.com",
    "nytimes.com", "theguardian.com", "bbc.co.uk", "bbc.com", "cnn.com"
]

LOW_TRUST_DOMAINS = [
    "linkedin.com", "reddit.com", "quora.com", "wikipedia.org",
    "medium.com", "twitter.com", "x.com", "substack.com",
]

# Trust level → numeric score (PRIMARY_SOURCE is highest)
TRUST_SCORE_MAP = {
    "PRIMARY SOURCE": 15,   # concall / annual report / regulatory filing
    "HIGH TRUST":     10,
    "MEDIUM TRUST":    6,
    "LOW TRUST":       2,
}

# ==========================================
# THRESHOLDS — tunable quality gates
# ==========================================
MIN_VERIFIED_FACTS            = 2
MIN_REPORT_CONFIDENCE         = 50
ENTITY_CONFIDENCE_THRESHOLD   = 60
FACT_QUALITY_THRESHOLD        = 55   # NEW: minimum composite fact quality score
OPTION_SCORE_THRESHOLD        = 30   # NEW: minimum option score to be selectable
GENERIC_WORD_THRESHOLD        = 2    # NEW: max generic words before rejection

# ==========================================
# GENERIC LANGUAGE DETECTOR
# NEW: Requirement 10 — reject generic recommendations
# ==========================================
GENERIC_PHRASES = [
    "leverage synergies", "best practices", "holistic approach", "paradigm shift",
    "move the needle", "low-hanging fruit", "boil the ocean", "think outside the box",
    "take it to the next level", "core competencies", "value-add", "proactive",
    "robust solution", "streamline operations", "going forward", "at the end of the day",
    "circle back", "deep dive", "bandwidth", "actionable insights", "digital transformation",
    "invest in capabilities", "strengthen positioning", "explore opportunities",
    "consider expanding", "may wish to", "could potentially", "it is recommended that"
]

REASONING_WORDS_IN_OBSERVATIONS = [
    "because", "therefore", "suggests", "indicates", "implies", "means that",
    "as a result", "due to", "caused by", "which shows", "this proves",
    "consequently", "hence", "thus", "leading to", "resulting in"
]

# ==========================================
# REQ 1 — REJECT CATEGORIES: non-decision-grade content
# ==========================================
REJECT_CONTENT_PATTERNS = [
    # Mission / vision / corporate description markers
    r'\b(our mission|our vision|our purpose|we believe|we strive|we are committed)\b',
    r'\b(company overview|about us|who we are|our story|founded in)\b',
    r'\b(world.?class|industry.?leading|best.?in.?class|leading provider|trusted partner)\b',
    r'\b(dedicated to|passionate about|committed to excellence|customer.?centric)\b',
    # Pure marketing language
    r'\b(innovative solutions?|cutting.?edge|state.?of.?the.?art|next.?generation)\b',
    r'\b(seamless experience|transforming the way|reimagining|revolutionizing)\b',
]

PREFERRED_CONTENT_KEYWORDS = [
    "revenue", "profit", "margin", "ebitda", "earnings", "net income",
    "market share", "growth rate", "capex", "acquisition", "divestiture",
    "launched", "partnership", "regulatory", "compliance", "penalty",
    "crore", "billion", "million", "lakh", "percent", "%", "₹", "$", "€",
    "quarter", "annual", "fiscal", "q1", "q2", "q3", "q4",
]


# ==========================================
# METRIC PRESERVATION — observation must not swap metric identity
# ==========================================
# Maps source metric keywords → the canonical metric name
METRIC_IDENTITY_MAP = {
    "pat": "PAT (Profit After Tax)",
    "profit after tax": "PAT (Profit After Tax)",
    "net profit": "Net Profit",
    "net income": "Net Income",
    "ebitda": "EBITDA",
    "operating profit": "Operating Profit",
    "ebit": "EBIT",
    "gross margin": "Gross Margin",
    "operating margin": "Operating Margin",
    "net margin": "Net Margin",
    "revenue": "Revenue",
    "sales": "Revenue",
    "turnover": "Revenue",
    "market share": "Market Share",
    "capex": "Capital Expenditure",
    "capital expenditure": "Capital Expenditure",
    "free cash flow": "Free Cash Flow",
    "fcf": "Free Cash Flow",
    "eps": "EPS",
    "earnings per share": "EPS",
    "debt": "Debt",
    "leverage": "Leverage",
}

# Groups of metrics that must NOT be interchanged
METRIC_GROUPS = [
    {"pat", "profit after tax", "net profit", "net income"},
    {"ebitda", "operating profit", "ebit"},
    {"gross margin", "operating margin", "net margin"},
    {"revenue", "sales", "turnover"},
    {"market share"},
    {"capex", "capital expenditure"},
    {"free cash flow", "fcf"},
    {"eps", "earnings per share"},
]


def detect_metric_in_text(text: str) -> Optional[str]:
    """Returns the first recognised metric keyword found in text (lowercase)."""
    tl = text.lower()
    # Longer phrases first to avoid partial matches
    for key in sorted(METRIC_IDENTITY_MAP.keys(), key=len, reverse=True):
        if key in tl:
            return key
    return None


def check_metric_preservation(evidence: str, observation: str) -> tuple[bool, str]:
    """
    Returns (violated, reason).
    Violated if the observation references a DIFFERENT metric group than the evidence.
    """
    if not evidence or not observation:
        return False, ""
    ev_metric  = detect_metric_in_text(evidence)
    obs_metric = detect_metric_in_text(observation)
    if not ev_metric or not obs_metric:
        return False, ""
    if ev_metric == obs_metric:
        return False, ""
    # Check if they belong to different groups
    ev_group  = next((g for g in METRIC_GROUPS if ev_metric  in g), None)
    obs_group = next((g for g in METRIC_GROUPS if obs_metric in g), None)
    if ev_group and obs_group and ev_group != obs_group:
        return True, (
            f"Metric substitution: evidence references '{METRIC_IDENTITY_MAP.get(ev_metric, ev_metric)}' "
            f"but observation references '{METRIC_IDENTITY_MAP.get(obs_metric, obs_metric)}'. "
            "Observations must preserve the exact metric from evidence."
        )
    return False, ""


# ==========================================
# COMPANY RELEVANCE VALIDATOR
# Reject facts that describe industry trends without explicit company linkage
# ==========================================
INDUSTRY_TREND_MARKERS = [
    r'\b(the industry|the sector|the market|industry as a whole|sector wide|across the industry)\b',
    r'\b(globally|worldwide|industry players|market participants|analysts expect|experts predict)\b',
    r'\b(the overall market|broader market|industry average|sector average|peer group)\b',
    r'\b(it is expected|it is projected|forecasters|research firms predict)\b',
]

COMPANY_LINKAGE_MARKERS = [
    # Will be populated dynamically using entity canonical name at runtime
]


def check_company_relevance(fact_text: str, canonical_name: str) -> tuple[bool, str]:
    """
    Returns (irrelevant, reason).
    A fact is irrelevant if it describes an industry/market trend without
    explicitly naming or linking to the target company.
    """
    text_lower = fact_text.lower()
    name_lower = canonical_name.lower()

    # Remove common suffixes for matching
    name_core = name_lower
    for suffix in [" limited", " ltd", " inc", " corp", " group", " pvt", " plc"]:
        name_core = name_core.replace(suffix, "")
    name_core = name_core.strip()

    # Check if the fact mentions the company at all
    company_mentioned = name_core in text_lower or name_lower in text_lower

    # Check for industry-trend language
    has_trend_language = any(
        re.search(pattern, text_lower) for pattern in INDUSTRY_TREND_MARKERS
    )

    if has_trend_language and not company_mentioned:
        return True, (
            f"Fact describes an industry/market trend without explicitly linking to "
            f"'{canonical_name}'. Only company-specific facts are admitted."
        )
    return False, ""


# ==========================================
# SEMANTIC DEDUPLICATION
# Reject facts that express the same idea with different wording
# ==========================================
def semantic_overlap_score(text_a: str, text_b: str) -> float:
    """
    Returns word-overlap ratio between two texts (Jaccard similarity on content words).
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
        "in", "of", "to", "for", "and", "or", "but", "its", "their", "this",
        "that", "with", "by", "at", "on", "from", "it", "as", "be", "will",
        "been", "also", "which", "who", "when", "where", "during", "after",
        "about", "into", "than", "more", "over", "not", "all", "new",
    }
    def content_words(t: str) -> set:
        return {w.strip(".,;:()[]\"'") for w in t.lower().split()
                if w.strip(".,;:()[]\"'") and w.strip(".,;:()[]\"'") not in stopwords}

    wa = content_words(text_a)
    wb = content_words(text_b)
    if not wa or not wb:
        return 0.0
    intersection = wa & wb
    union = wa | wb
    return len(intersection) / len(union)


def deduplicate_facts(facts: List) -> tuple[List, List[dict]]:
    """
    Removes semantically duplicate facts.
    Returns (deduplicated_facts, duplicate_log).
    Two facts are duplicates if their Jaccard similarity > 0.55.
    """
    kept: List = []
    dup_log: List[dict] = []

    for candidate in facts:
        duplicate_of = None
        for existing in kept:
            score = semantic_overlap_score(candidate.fact, existing.fact)
            if score > 0.55:
                duplicate_of = existing.fact
                break
        if duplicate_of:
            dup_log.append({
                "rejected_fact": candidate.fact[:120],
                "duplicate_of": duplicate_of[:120],
                "reason": f"Semantic overlap > 55% — same idea expressed differently."
            })
        else:
            kept.append(candidate)

    return kept, dup_log


def is_non_decision_content(fact_text: str) -> tuple[bool, str]:
    """Returns (True, reason) if fact is mission/marketing/non-decision content."""
    import re as _re
    text_lower = fact_text.lower()
    for pattern in REJECT_CONTENT_PATTERNS:
        if _re.search(pattern, text_lower):
            return True, f"Non-decision content pattern detected: '{pattern}'"
    # If fact has none of the preferred keywords and is short, likely generic
    has_preferred = any(kw in text_lower for kw in PREFERRED_CONTENT_KEYWORDS)
    if not has_preferred and len(fact_text.split()) < 12:
        return True, "No decision-relevant keywords (revenue, margin, share, regulatory, etc.)"
    return False, ""


def count_generic_phrases(text: str) -> tuple[int, list]:
    text_lower = text.lower()
    found = [p for p in GENERIC_PHRASES if p in text_lower]
    return len(found), found


def contains_reasoning(observation: str) -> tuple[bool, str]:
    obs_lower = observation.lower()
    found = [w for w in REASONING_WORDS_IN_OBSERVATIONS if w in obs_lower]
    if found:
        return True, f"Contains reasoning language: {', '.join(found)}"
    return False, ""


# ==========================================
# REQ 3 — INFERENCE QUALITY: reject inferences that merely rephrase the observation
# ==========================================
def inference_merely_rephrases(observation: str, inference: str) -> tuple[bool, str]:
    """
    Returns (True, reason) if the inference is just a renamed version of the observation.
    Checks:
    1. Levenshtein-like word overlap > 70%
    2. Inference is shorter or equal to observation (no new meaning added)
    3. Inference adds no explanatory value (no significance words)
    """
    if not observation or not inference:
        return False, ""

    # Strip classification tag from inference for comparison
    inf_clean = inference.split("|")[0].strip().lower()
    obs_clean = observation.lower()

    obs_words = set(obs_clean.split())
    inf_words = set(inf_clean.split())

    # Remove stopwords for overlap calculation
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "has", "have",
                 "had", "in", "of", "to", "for", "and", "or", "but", "its",
                 "their", "this", "that", "with", "by", "at", "on", "from"}
    obs_content = obs_words - stopwords
    inf_content = inf_words - stopwords

    if not obs_content:
        return False, ""

    overlap = len(obs_content & inf_content) / len(obs_content)

    # Significance words signal the inference adds meaning
    significance_words = [
        "signal", "suggest", "pattern", "pressure", "advantage", "risk",
        "opportunity", "challenge", "momentum", "strength", "weakness",
        "competitive", "strategic", "structural", "cyclical", "systemic",
        "demand", "supply", "portfolio", "capacity", "execution",
        "erosion", "expansion", "discipline", "positioning", "exposure"
    ]
    adds_significance = any(sw in inf_clean for sw in significance_words)

    if overlap > 0.70 and not adds_significance:
        return True, (
            f"Inference overlaps {int(overlap*100)}% with observation and adds no explanatory significance. "
            "Inference must explain strategic meaning, not rephrase the fact."
        )
    return False, ""


# ==========================================
# REQ 12 — LAYER DIFFERENTIATION: observation ≠ inference ≠ theme
# ==========================================
def check_layer_differentiation(observation: str, inference: str, theme_name: str = "") -> list[str]:
    """Returns list of differentiation violations."""
    issues = []
    if not observation or not inference:
        return issues

    inf_clean = inference.split("|")[0].strip().lower()
    obs_clean = observation.lower()

    # Obs vs Inference
    rephrased, msg = inference_merely_rephrases(observation, inference)
    if rephrased:
        issues.append(f"Layer violation — Inference is a rephrasing of Observation: {msg}")

    # Inference vs Theme
    if theme_name:
        theme_lower = theme_name.lower()
        inf_words = set(inf_clean.split()) - {"the", "a", "an", "is", "are", "in", "of", "and"}
        theme_words = set(theme_lower.split()) - {"the", "a", "an", "is", "are", "in", "of", "and"}
        if theme_words and inf_words and len(theme_words & inf_words) / max(len(theme_words), 1) > 0.65:
            issues.append(
                f"Layer violation — Theme '{theme_name}' appears to be a renamed Inference, not a broader pattern."
            )

    return issues


def evaluate_trust(url: str, company: str = "") -> str:
    """
    Returns trust label for a URL.
    PRIMARY SOURCE = concall transcripts, annual reports, regulatory filings.
    HIGH TRUST     = established financial press.
    MEDIUM TRUST   = general business press.
    LOW TRUST      = social / user-generated content.
    """
    url_lower = url.lower()
    domain = urlparse(url).netloc.lower().replace("www.", "")

    # Check primary source URL path patterns (concall/AR/filing)
    for pattern in PRIMARY_SOURCE_URL_PATTERNS:
        if pattern in url_lower:
            return "PRIMARY SOURCE"

    # Check primary source domains
    for ps_domain in PRIMARY_SOURCE_DOMAINS:
        if ps_domain in domain:
            return "PRIMARY SOURCE"

    # Check if URL is from the company's own IR/investor page
    if company:
        clean_company = company.lower()
        for stopword in ["the", "group", "inc", "ltd", "llc", "corp", "co", "pvt", "plc", "incorporated", "limited"]:
            clean_company = clean_company.replace(stopword, "")
        clean_company = clean_company.strip().replace(" ", "")
        if clean_company and clean_company in domain.replace("-", ""):
            # Company's own domain — is it an IR page?
            if any(ir in url_lower for ir in ["/investor", "/ir/", "/financials/", "/results/", "/annual"]):
                return "PRIMARY SOURCE"
            return "HIGH TRUST"

    if any(h in domain for h in HIGH_TRUST_DOMAINS):   return "HIGH TRUST"
    if any(m in domain for m in MEDIUM_TRUST_DOMAINS): return "MEDIUM TRUST"
    if any(l in domain for l in LOW_TRUST_DOMAINS):    return "LOW TRUST"
    return "MEDIUM TRUST"


# ==========================================
# 3. DETERMINISTIC SCORING
# ==========================================
def calculate_confidence(trust_label: str, board_relevance: int, strategic_impact: int) -> int:
    trust_score = TRUST_SCORE_MAP.get(trust_label.strip().upper(), 5)
    # PRIMARY SOURCE scores out of 15; normalise to same 0-10 scale for formula
    trust_normalised = min(trust_score, 10)
    raw = (trust_normalised * 0.4) + (board_relevance * 0.3) + (strategic_impact * 0.3)
    return int((raw / 10) * 100)


def calculate_fact_quality_score(fact_text: str, source_trust: str,
                                  board_relevance: int, strategic_impact: int,
                                  date_signal: str) -> tuple[int, dict]:
    """
    Composite fact quality score (0-100) from 5 dimensions:
    - Specificity (has numbers/percentages/named entities): 0-25
    - Source trust: 0-30  (PRIMARY SOURCE gets full 30; others scale down)
    - Board relevance: 0-20
    - Strategic impact: 0-20
    - Recency (has date signal): 0-10
    PRIMARY SOURCE documents (concall/AR/filing) receive a 5-point bonus.
    """
    breakdown = {}

    # Specificity — does the fact contain hard data?
    has_numbers  = bool(re.search(r'\d', fact_text))
    has_percent  = '%' in fact_text
    has_currency = bool(re.search(r'[$₹€£¥]|\b(crore|lakh|billion|million|trillion)\b', fact_text, re.I))
    has_named    = len(fact_text.split()) > 8

    specificity = 0
    if has_numbers:  specificity += 10
    if has_percent:  specificity += 7
    if has_currency: specificity += 5
    if has_named:    specificity += 3
    breakdown["specificity"] = min(specificity, 25)

    # Source trust — PRIMARY SOURCE gets full 30 points
    trust_upper = source_trust.strip().upper()
    if trust_upper == "PRIMARY SOURCE":
        breakdown["source_trust"] = 30
    else:
        trust_raw = TRUST_SCORE_MAP.get(trust_upper, 5)
        breakdown["source_trust"] = int((trust_raw / 10) * 25)

    # Board relevance (1-10 → 0-20)
    breakdown["board_relevance"] = int((board_relevance / 10) * 20)

    # Strategic impact (1-10 → 0-20)
    breakdown["strategic_impact"] = int((strategic_impact / 10) * 20)

    # Recency
    breakdown["recency"] = 10 if date_signal not in ["Undated", "Unknown", ""] else 2

    total = sum(breakdown.values())
    return min(total, 100), breakdown


def calculate_entity_confidence(entity) -> tuple[int, str]:
    score = 100
    reasons = []
    if entity.industry == "Unknown":       score -= 20; reasons.append("industry unknown")
    if entity.sector == "Unknown":         score -= 10; reasons.append("sector unknown")
    if entity.business_model == "Unknown": score -= 15; reasons.append("business model unknown")
    if entity.primary_market == "Unknown": score -= 10; reasons.append("primary market unknown")
    if entity.known_competitors == "Unknown": score -= 10; reasons.append("competitors unknown")
    contamination = entity.contamination_warnings.lower()
    if "failed" in contamination:
        score -= 25; reasons.append("entity resolution failed")
    elif "none" not in contamination and contamination != "":
        score -= 20; reasons.append(f"contamination risk: {entity.contamination_warnings}")
    explanation = f"Entity confidence {score}%"
    if reasons: explanation += f" — Issues: {', '.join(reasons)}"
    return max(0, score), explanation


def calculate_report_confidence(verified_facts: list, total_facts: int) -> int:
    if not verified_facts or total_facts == 0: return 15
    gate_rate = len(verified_facts) / total_facts
    avg_conf  = sum(f.confidence for f in verified_facts) / len(verified_facts)
    return int((gate_rate * 0.4 + avg_conf / 100 * 0.6) * 100)


# ==========================================
# REQ 11 — CONFIDENCE CALIBRATION: deterministic label from evidence count & quality
# ==========================================
def calibrate_confidence_label(verified_facts: list) -> tuple[str, str]:
    """
    Returns (label, explanation) based on evidence quantity and quality.
    Scale:
      LOW       = 1-2 verified facts
      MEDIUM    = 3-4 verified facts
      MEDIUM-HIGH = 5-7 verified facts
      HIGH      = 8+ verified facts with multiple independent sources
    Never returns HIGH from weak evidence (< 2 high-trust sources).
    """
    n = len(verified_facts)
    high_trust_count = sum(1 for f in verified_facts if "HIGH TRUST" in f.source_trust.upper())
    avg_quality = sum(f.fact_quality_score for f in verified_facts) / n if n else 0

    if n >= 8 and high_trust_count >= 2 and avg_quality >= 65:
        label = "HIGH"
        explanation = (
            f"{n} verified facts, {high_trust_count} high-trust independent sources, "
            f"avg quality {avg_quality:.0f}/100 — multiple corroborating sources present."
        )
    elif n >= 5:
        label = "MEDIUM-HIGH"
        explanation = (
            f"{n} verified facts, {high_trust_count} high-trust source(s), "
            f"avg quality {avg_quality:.0f}/100 — solid evidence base."
        )
    elif n >= 3:
        label = "MEDIUM"
        explanation = (
            f"{n} verified facts, {high_trust_count} high-trust source(s), "
            f"avg quality {avg_quality:.0f}/100 — moderate evidence base."
        )
    else:
        label = "LOW"
        explanation = (
            f"Only {n} verified fact(s) passed quality gate — "
            "conclusions are tentative and should not drive irreversible decisions."
        )

    return label, explanation


def get_evidence_sufficiency(verified_facts: list, report_confidence: int) -> tuple[bool, str]:
    if len(verified_facts) < MIN_VERIFIED_FACTS:
        return False, f"Only {len(verified_facts)} fact(s) passed validation. Insufficient evidence."
    if report_confidence < MIN_REPORT_CONFIDENCE:
        return False, f"Report confidence {report_confidence}% is below threshold. Evidence quality is low."
    return True, "Evidence sufficient for reliable conclusions."


# Option scoring — 6-dimension formula (new prompt spec)
# Score = (0.25 × Evidence Support) + (0.20 × Strategic Fit) + (0.25 × Opportunity)
#       + (0.15 × Urgency) - (0.10 × Risk) - (0.05 × Complexity)
# All inputs 1-10; output scaled to 0-100.
def calculate_option_score(evidence_support: int, strategic_fit: int,
                            opportunity: int, urgency: int,
                            risk: int, complexity: int) -> int:
    """
    6-dimension composite option score (0-100).
    Risk and Complexity are cost terms (subtracted).
    Aggressive options can win when opportunity and urgency are high.
    """
    raw = (
        evidence_support * 0.25 +
        strategic_fit    * 0.20 +
        opportunity      * 0.25 +
        urgency          * 0.15 -
        risk             * 0.10 -
        complexity       * 0.05
    )
    # raw range: min ≈ (1*0.85 - 10*0.15) = -0.65  →  max ≈ (10*0.85 - 1*0.15) = 8.35
    # Normalise to 0-100
    raw_min = 1 * 0.85 - 10 * 0.15   # ≈ 0.35 (1s for positives, 10s for costs)
    raw_max = 10 * 0.85 - 1 * 0.15   # ≈ 8.35
    normalised = (raw - raw_min) / (raw_max - raw_min)
    return max(0, min(100, int(normalised * 100)))


# ==========================================
# SPECIFICITY TEST — "50 unrelated companies" rule
# ==========================================
UNIVERSAL_STRATEGY_PATTERNS = [
    r'\b(improve (customer|operational|product|service) (experience|quality|efficiency))\b',
    r'\b(expand (market|global|international|geographic) (presence|reach|footprint))\b',
    r'\b(invest in (technology|talent|innovation|digital|infrastructure))\b',
    r'\b(build (brand|awareness|loyalty|recognition))\b',
    r'\b(reduce (costs?|expenses?|overhead))\b',
    r'\b(increase (revenue|sales|market share|profitability))\b',
    r'\b(develop new (products?|services?|offerings?))\b',
    r'\b(improve (operations?|processes?|systems?))\b',
    r'\b(strengthen (governance|compliance|risk management))\b',
    r'\b(acquire (companies|businesses|startups|competitors))\b',
]


def check_recommendation_specificity(decision: str) -> tuple[bool, str]:
    """
    Returns (failed, reason) if recommendation could apply to 50 unrelated companies.
    A recommendation passes if it mentions company-specific facts, numbers, or named entities.
    """
    decision_lower = decision.lower()
    pattern_hits = [
        p for p in UNIVERSAL_STRATEGY_PATTERNS
        if re.search(p, decision_lower)
    ]
    if not pattern_hits:
        return False, ""

    # Check if it is rescued by company-specific anchors (numbers, named products, etc.)
    has_specifics = bool(re.search(
        r'\d|%|₹|\$|€|crore|billion|million|lakh|q[1-4]|fy\d|fiscal|\b[A-Z][a-z]{3,}\b',
        decision
    ))
    if has_specifics:
        return False, ""  # Anchored to specific data — passes

    return True, (
        f"Specificity test FAILED — recommendation contains {len(pattern_hits)} universal strategy "
        f"pattern(s): {', '.join(p[:40] for p in pattern_hits[:2])}. "
        "This could apply to 50 unrelated companies. Rewrite with company-specific evidence anchors."
    )


# ==========================================
# OPTION DIFFERENTIATION CHECK
# Conservative=Protect, Balanced=Optimize, Aggressive=Create new advantage
# ==========================================
OPTION_STRATEGY_SIGNATURES = {
    "Conservative": {
        "required": ["protect", "defend", "maintain", "retain", "secure", "preserve",
                     "hold", "consolidate", "safeguard", "sustain"],
        "forbidden": ["new market", "new segment", "launch", "acquire", "disrupt",
                      "transform", "new advantage", "aggressive"],
    },
    "Balanced": {
        "required": ["optimis", "optimiz", "improve", "enhance", "efficienc",
                     "strengthen existing", "build on", "leverage existing",
                     "deepen", "expand within"],
        "forbidden": ["protect existing", "defend position", "disrupt", "acquire",
                      "new market", "new category"],
    },
    "Aggressive": {
        "required": ["new", "create", "disrupt", "enter", "acquire", "launch",
                     "advantage", "transform", "pioneer", "first-mover"],
        "forbidden": ["protect", "maintain current", "sustain existing", "hold"],
    },
}


def check_option_differentiation(options) -> list[str]:
    """
    Returns list of violations where options are not materially different.
    Also checks that each option matches its intended strategic posture.
    """
    issues = []
    if not options or len(options) < 3:
        return issues

    descriptions = {opt.option_type: (opt.description or "").lower() for opt in options}

    # Cross-option semantic overlap — pairwise
    option_types = list(descriptions.keys())
    for i in range(len(option_types)):
        for j in range(i + 1, len(option_types)):
            ta, tb = option_types[i], option_types[j]
            overlap = semantic_overlap_score(descriptions[ta], descriptions[tb])
            if overlap > 0.60:
                issues.append(
                    f"Option differentiation FAILED: '{ta}' and '{tb}' share {int(overlap*100)}% "
                    "semantic overlap — they appear to be the same strategy at different intensity levels. "
                    "Conservative must protect, Balanced must optimize, Aggressive must create new advantage."
                )

    # Strategic posture check per option
    for opt_type, sigs in OPTION_STRATEGY_SIGNATURES.items():
        desc = descriptions.get(opt_type, "")
        if not desc:
            continue
        has_required = any(kw in desc for kw in sigs["required"])
        if not has_required:
            issues.append(
                f"Option '{opt_type}': Description does not reflect the required strategic posture. "
                f"Expected language reflecting: {', '.join(sigs['required'][:4])}."
            )

    return issues


# NEW: Requirement 9 — traceability chain validator (+ Req 3, 5, 6, 11, 12)
def validate_traceability_chain(brief, verified_facts: list = None) -> list[str]:
    """
    Returns a list of traceability violations found in the brief.
    Empty list = chain is valid.
    Covers: Req 2 (obs purity), Req 3 (inference quality), Req 4 (classification),
            Req 5 (theme threshold), Req 6 (competitor evidence sentinel),
            Req 9 (chain completeness), Req 10 (generic check), Req 12 (layer diff).
    """
    violations = []

    # Collect theme names for layer differentiation check
    theme_names = [ts.name or "" for ts in brief.strategic_themes_and_signals]

    # ── Evidence → Observation → Inference ─────────────────────────────
    for i, log in enumerate(brief.evidence_and_observation_log):
        tag = f"Log {i+1}"

        if not log.evidence or len(log.evidence.strip()) < 10:
            violations.append(f"{tag}: Evidence is missing or too vague.")
        if not log.observation or len(log.observation.strip()) < 10:
            violations.append(f"{tag}: Observation is missing.")

        # Req 2 — observation purity
        has_reasoning, reasoning_msg = contains_reasoning(log.observation or "")
        if has_reasoning:
            violations.append(f"{tag}: Observation contains reasoning language — {reasoning_msg}")

        # METRIC PRESERVATION — observation must not swap metric identity
        metric_violated, metric_msg = check_metric_preservation(
            log.evidence or "", log.observation or ""
        )
        if metric_violated:
            violations.append(f"{tag}: {metric_msg}")

        if log.inference:
            # Req 4 — inference must have classification tag
            parts = log.inference.split("|")
            classification = parts[-1].strip().upper() if len(parts) > 1 else ""
            if classification not in ["CONFIRMED", "LIKELY", "HYPOTHESIS"]:
                violations.append(
                    f"{tag}: Inference missing classification — must end with "
                    "'| CONFIRMED', '| LIKELY', or '| HYPOTHESIS'."
                )

            # Req 3 — inference must not merely rephrase observation
            rephrased, rephrase_msg = inference_merely_rephrases(log.observation or "", log.inference)
            if rephrased:
                violations.append(f"{tag}: {rephrase_msg}")

            # Req 12 — layer differentiation: inference ≠ observation ≠ any theme
            for tname in theme_names:
                layer_issues = check_layer_differentiation(log.observation or "", log.inference, tname)
                for issue in layer_issues:
                    violations.append(f"{tag}: {issue}")
        else:
            violations.append(f"{tag}: Inference is missing entirely.")

    # ── Themes (Req 5 threshold enforcement) ────────────────────────────
    for i, theme in enumerate(brief.strategic_themes_and_signals):
        trace_count = len(theme.traceability)
        theme_type = (theme.type or "").upper()

        # A STRATEGIC THEME requires ≥ 2 traceability references
        if "THEME" in theme_type and trace_count < 2:
            violations.append(
                f"Theme '{theme.name}': Classified as STRATEGIC THEME but has only "
                f"{trace_count} traceability reference(s). Minimum 2 required — "
                "should be downgraded to EMERGING SIGNAL."
            )
        # An EMERGING SIGNAL with 0 traceability is still a violation
        elif trace_count == 0:
            violations.append(
                f"Theme '{theme.name}': No traceability references at all — "
                "minimum 1 required even for EMERGING SIGNAL."
            )

        # Req 12 — theme name should not be a direct copy of an observation/inference
        # (checked above in the per-log loop; flagging here if theme is just a bare label)
        if theme.name and len(theme.name.split()) <= 2:
            bad_bare_labels = [
                "revenue growth", "profitability", "market share", "growth",
                "innovation", "expansion", "efficiency", "performance"
            ]
            if theme.name.lower().strip() in bad_bare_labels:
                violations.append(
                    f"Theme '{theme.name}': Name is a bare category label, not a strategic pattern. "
                    "Rename to a descriptive theme (e.g. 'Brand-Led Growth', 'Margin Expansion Under Pressure')."
                )

    # ── Competitors (Req 6 — sentinel instead of null) ──────────────────
    for comp in brief.competitive_landscape:
        comp_name = comp.competitor or "Unknown"
        if comp.advantage and not comp.advantage_evidence:
            violations.append(
                f"Competitor '{comp_name}': Advantage claim '{comp.advantage[:60]}' "
                "has no evidence backing. Use INSUFFICIENT_COMPETITIVE_EVIDENCE or cite specific evidence."
            )
        if comp.vulnerability and not comp.vulnerability_evidence:
            violations.append(
                f"Competitor '{comp_name}': Vulnerability claim '{comp.vulnerability[:60]}' "
                "has no evidence backing. Use INSUFFICIENT_COMPETITIVE_EVIDENCE or cite specific evidence."
            )

    # ── Decision traceability (Req 9) ───────────────────────────────────
    decision = brief.recommended_decision or ""
    if decision:
        for required in ["Observation", "Inference", "Theme", "Option"]:
            if required.lower() not in decision.lower():
                violations.append(f"Decision: Does not explicitly reference a {required} — chain is broken.")

        # Req 10 — generic language check
        generic_count, generic_found = count_generic_phrases(decision)
        if generic_count >= GENERIC_WORD_THRESHOLD:
            violations.append(
                f"Decision: Contains {generic_count} generic phrase(s): {', '.join(generic_found)}. "
                "Recommendation must be specific to this entity and evidence."
            )

        # SPECIFICITY TEST — reject if recommendation could apply to 50 unrelated companies
        specificity_fail, specificity_reason = check_recommendation_specificity(decision)
        if specificity_fail:
            violations.append(f"Decision: {specificity_reason}")
    else:
        violations.append("Decision: recommended_decision is empty — no recommendation was generated.")

    # ── Option differentiation check ────────────────────────────────────
    option_diff_issues = check_option_differentiation(brief.evaluated_options)
    violations.extend(option_diff_issues)

    return violations


# ==========================================
# 4. SEARCH — two-phase: primary docs first, then general
# ==========================================

def _ddgs_search(queries: list, max_per_query: int = 3) -> list[dict]:
    """Run a list of DDGS text queries and return raw result dicts."""
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=max_per_query):
                    results.append(r)
    except Exception as e:
        st.warning(f"Search partial failure: {e}")
    return results


def run_primary_source_search(company: str) -> str:
    """
    Phase 1 — targeted search for concall transcripts, annual reports,
    investor presentations, and regulatory filings.
    These are the highest-value evidence sources.
    """
    current_year = datetime.now().year
    prev_year    = current_year - 1

    queries = [
        # Concall transcripts
        f'"{company}" concall transcript {current_year} earnings call',
        f'"{company}" Q4 {prev_year} earnings call transcript management commentary',
        f'"{company}" Q3 {prev_year} concall transcript site:trendlyne.com OR site:tickertape.in OR site:screener.in',
        # Annual reports
        f'"{company}" annual report {prev_year} {current_year} PDF investor',
        f'"{company}" annual report highlights FY{str(prev_year)[-2:]} FY{str(current_year)[-2:]}',
        # Investor presentations
        f'"{company}" investor presentation {current_year} strategy roadmap',
        f'"{company}" investor day {current_year} OR {prev_year} guidance targets',
        # Regulatory / exchange filings
        f'"{company}" BSE NSE filing results announcement {current_year}',
        f'"{company}" SEBI disclosure quarterly results site:bseindia.com OR site:nseindia.com',
        # Management guidance
        f'"{company}" management guidance FY{str(current_year)[-2:]} outlook commentary CEO CFO',
        f'"{company}" CFO interview revenue guidance margin outlook {current_year}',
    ]

    results = []
    for r in _ddgs_search(queries, max_per_query=3):
        url = r.get("href", "")
        trust = evaluate_trust(url, company)
        results.append(
            f"[PRIMARY SOURCE SEARCH]\n"
            f"SOURCE: {url}\nTRUST: {trust}\n"
            f"CONTENT: {r.get('title','')} — {r.get('body','')}\n{'='*50}"
        )
    return "\n".join(results)


def run_general_search(company: str) -> str:
    """
    Phase 2 — general financial and strategic intelligence search.
    Supplements primary docs with news, analysis, and competitive context.
    """
    current_year = datetime.now().year
    queries = [
        f"{company} revenue profit margin earnings performance {current_year}",
        f"{company} market share competitor position metrics {current_year}",
        f"{company} capital allocation acquisitions investments {current_year}",
        f"{company} regulatory challenges compliance penalty {current_year}",
        f"{company} strategic shift new business segment {current_year}",
        f"{company} competitor threat market disruption {current_year}",
    ]

    results = []
    for r in _ddgs_search(queries, max_per_query=2):
        url = r.get("href", "")
        trust = evaluate_trust(url, company)
        results.append(
            f"SOURCE: {url}\nTRUST: {trust}\n"
            f"CONTENT: {r.get('title','')} — {r.get('body','')}\n{'-'*40}"
        )
    return "\n".join(results)


def run_enhanced_search(company: str) -> str:
    """
    Combined two-phase search.
    Primary sources (concall/AR/filings) are searched first and prepended
    so the researcher agent prioritises them.
    """
    st.write("📂 Phase 1 — Searching concall transcripts, annual reports & regulatory filings...")
    primary_context = run_primary_source_search(company)

    st.write("📰 Phase 2 — Searching financial news & competitive intelligence...")
    general_context = run_general_search(company)

    # Primary sources prepended with clear separator so LLM sees them first
    combined = ""
    if primary_context:
        combined += "===== PRIMARY DOCUMENTS (CONCALL / ANNUAL REPORT / FILING) =====\n"
        combined += primary_context + "\n\n"
    if general_context:
        combined += "===== GENERAL FINANCIAL INTELLIGENCE =====\n"
        combined += general_context
    return combined


# ==========================================
# 5. JSON INVOKE
# ==========================================
def invoke_json(prompt: str) -> dict:
    messages = [
        SystemMessage(content=(
            "You are a precise JSON-only responder. "
            "Output ONLY valid JSON. No markdown, no explanation, no code fences, no trailing commas."
        )),
        HumanMessage(content=prompt)
    ]
    resp = llm.invoke(messages)
    text = resp.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip().rstrip("```").strip()
    return json.loads(text)


# ==========================================
# 6. PYDANTIC MODELS
# ==========================================
class EntityProfile(BaseModel):
    canonical_name: str
    industry: str
    sector: str
    business_model: str
    primary_market: str
    known_subsidiaries: str
    known_competitors: str
    contamination_warnings: str


class IntelligenceFact(BaseModel):
    category: str
    fact: str
    source_url: str
    source_trust: str
    source_type: str = "General"   # "Concall", "Annual Report", "Regulatory Filing", "Investor Presentation", "General"
    date_signal: str = "Undated"
    board_relevance: int
    strategic_impact: int


# NEW: Requirement 1 — fact quality score attached to validated facts
class ValidatedFact(BaseModel):
    category: str
    fact: str
    source_url: str
    source_trust: str
    source_type: str = "General"
    date_signal: str
    board_relevance: int
    strategic_impact: int
    confidence: int
    fact_quality_score: int = 0
    quality_breakdown: dict = Field(default_factory=dict)


class StrategicSignal(BaseModel):
    signal_type: str
    signal: str
    urgency: str
    implication: str


class EvidenceLog(BaseModel):
    evidence: Optional[str] = None
    observation: Optional[str] = Field(
        default=None,
        description="MUST only restate evidence. No reasoning, no 'because', no 'therefore'."
    )
    root_cause: Optional[str] = Field(
        default=None,
        description="Must explain the observation or be UNKNOWN. Never restate the observation."
    )
    inference: Optional[str] = Field(
        default=None,
        description="Format: [Inference statement] | [CONFIRMED/LIKELY/HYPOTHESIS]"
    )
    # NEW: observation purity flag
    observation_purity_passed: bool = True
    inference_classified: bool = True


class ThemeSignal(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = Field(
        default=None,
        description="Must be STRATEGIC THEME or EMERGING SIGNAL"
    )
    traceability: List[str] = Field(
        default_factory=list,
        description="Minimum 2 observations OR 3 facts required. No themes without supporting evidence."
    )


class CompetitiveLandscape(BaseModel):
    competitor: Optional[str] = None
    advantage: Optional[str] = None
    advantage_evidence: Optional[str] = Field(
        default=None,
        description="REQUIRED. Must cite a specific fact or source. Cannot be empty if advantage is stated."
    )
    vulnerability: Optional[str] = None
    vulnerability_evidence: Optional[str] = Field(
        default=None,
        description="REQUIRED. Must cite a specific fact or source. Cannot be empty if vulnerability is stated."
    )


# NEW: Requirement 8 — numeric scores on options (1-10 integers, not text)
class EvaluatedOption(BaseModel):
    option_type: Optional[str] = Field(
        default=None,
        description="Must be exactly: Conservative, Balanced, or Aggressive"
    )
    option_strategy: Optional[str] = Field(
        default=None,
        description=(
            "Conservative=Protect existing position, "
            "Balanced=Optimize existing position, "
            "Aggressive=Create new strategic advantage. "
            "Must NOT be same strategy at different intensity."
        )
    )
    description: Optional[str] = None
    traceability_chain: Union[str, List[str], None] = Field(
        default=None,
        description="Theme [X] -> Inference [Y] -> Observation [Z] -> Evidence [W]"
    )
    # 6-dimension scoring (1-10)
    evidence_support_score: int = Field(default=5, description="1-10. How strongly does evidence support this option?")
    strategic_fit_score: int    = Field(default=5, description="1-10. How well does it fit the strategic context?")
    opportunity_score: int      = Field(default=5, description="1-10. How large is the opportunity this option captures?")
    urgency_score: int          = Field(default=5, description="1-10. How time-sensitive is execution? Higher = more urgent.")
    risk_score: int             = Field(default=5, description="1-10. How high is the risk? Higher = riskier.")
    complexity_score: int       = Field(default=5, description="1-10. How complex to execute? Higher = more complex.")
    composite_score: int        = Field(default=0, description="Computed: weighted composite of the 6 scores.")
    generic_test_passed: Optional[str] = Field(
        default=None,
        description="Yes — uniquely applies to this situation. No — could apply to 50 unrelated companies."
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="If generic_test_passed is No, explain why this is rejected."
    )


class DecisionIntelligenceBrief(BaseModel):
    status: str = Field(description="Must be exactly 'SUFFICIENT' or 'INSUFFICIENT_EVIDENCE'")
    reason: Optional[str] = Field(default=None)

    evidence_and_observation_log: List[EvidenceLog] = Field(default_factory=list)
    strategic_themes_and_signals: List[ThemeSignal] = Field(default_factory=list)
    competitive_landscape: List[CompetitiveLandscape] = Field(default_factory=list)
    evaluated_options: List[EvaluatedOption] = Field(default_factory=list)

    recommended_decision: Optional[str] = Field(
        default=None,
        description=(
            "Must explicitly reference: 1 Observation (Obs:), 1 Inference (Inf:), 1 Theme (Theme:), 1 Option (Opt:). "
            "Must NOT contain generic language. Must be traceable to evidence."
        )
    )
    selected_option_type: Optional[str] = Field(
        default=None,
        description="Which option type was selected: Conservative, Balanced, or Aggressive"
    )
    selection_rationale: Optional[str] = Field(
        default=None,
        description="Why this option scored highest relative to the others."
    )
    contradicting_evidence: Optional[str] = Field(default=None)
    confidence_assessment: Optional[str] = Field(default=None)


# ==========================================
# 7. PIPELINE AGENTS
# ==========================================
FACT_CATEGORIES = [
    "Profitability", "Growth", "Competitive Threat",
    "Competitive Advantage", "Capital Allocation", "Strategic Shift"
]


def run_entity_resolution(company: str, raw_context: str) -> EntityProfile:
    prompt = f"""You are an Entity Resolution Specialist. Identify exactly which company the search results describe.
Return a JSON object:
{{
  "canonical_name": "official registered name",
  "industry": "specific industry",
  "sector": "sector",
  "business_model": "how it makes money",
  "primary_market": "main geography",
  "known_subsidiaries": "subsidiaries or Unknown",
  "known_competitors": "competitors or Unknown",
  "contamination_warnings": "None detected OR describe results about a different entity"
}}
Company queried: {company}
Search context: {raw_context[:2000]}"""
    try:
        data = invoke_json(prompt)
        return EntityProfile(**data)
    except Exception:
        return EntityProfile(
            canonical_name=company, industry="Unknown", sector="Unknown",
            business_model="Unknown", primary_market="Unknown",
            known_subsidiaries="Unknown", known_competitors="Unknown",
            contamination_warnings="Entity resolution failed — verify facts manually"
        )


def run_researcher(company: str, entity: EntityProfile, raw_context: str) -> List[IntelligenceFact]:
    current_year = datetime.now().year
    prompt = f"""You are a Fact Extraction System. Extract highly specific, verifiable data for {entity.canonical_name}.
Extract 4-6 deep, descriptive factual markers.

CRITICAL RULES:
1. Every fact MUST contain at least one of: a number, percentage, currency figure, named entity, or specific date.
2. Vague facts like "the company is growing" or "they are expanding" will be rejected. Do not include them.
3. Map source URLs and trust labels precisely.
4. Score board_relevance and strategic_impact between 1-10. Core operational updates, revenue numbers, and market share changes MUST score 8-10.

Return JSON:
{{
  "facts": [
    {{
      "category": "one of: {', '.join(FACT_CATEGORIES)}",
      "fact": "specific verifiable fact with numbers, percentages, or milestones — never vague",
      "source_url": "exact absolute URL",
      "source_trust": "HIGH TRUST / MEDIUM TRUST / LOW TRUST",
      "date_signal": "Q3 2024 or 2025 or specific month. If undated, write Undated.",
      "board_relevance": 9,
      "strategic_impact": 9
    }}
  ]
}}
Raw Context:
{raw_context}"""
    try:
        data = invoke_json(prompt)
        facts = []
        for f in data.get("facts", []):
            try:
                if isinstance(f.get("board_relevance"), str):
                    f["board_relevance"] = int(''.join(filter(str.isdigit, f["board_relevance"])) or 9)
                if isinstance(f.get("strategic_impact"), str):
                    f["strategic_impact"] = int(''.join(filter(str.isdigit, f["strategic_impact"])) or 9)
                facts.append(IntelligenceFact(**f))
            except Exception:
                continue
        return facts
    except Exception:
        return []


# NEW: Requirement 1 & 2 — fact quality scoring + rejection of low-value facts
def run_hard_gate_validation(
    facts: List[IntelligenceFact],
    canonical_name: str = ""
) -> tuple[List[ValidatedFact], List[dict]]:
    """
    Returns (verified_facts, rejection_log).
    rejection_log contains every rejected fact with its rejection reason.
    Applies: non-decision content, company relevance, board/impact scores,
             source trust, confidence, recency, fact quality score.
    Semantic deduplication is applied AFTER this gate (separate call).
    """
    verified  = []
    rejected  = []

    for f in facts:
        reasons = []

        # NEW Gate 0: reject mission/marketing/non-decision content (Req 1)
        is_non_decision, non_decision_reason = is_non_decision_content(f.fact)
        if is_non_decision:
            reasons.append(f"Non-decision-grade content — {non_decision_reason}")

        # NEW Gate 0b: reject facts not about the company (company relevance)
        if canonical_name:
            is_irrelevant, relevance_reason = check_company_relevance(f.fact, canonical_name)
            if is_irrelevant:
                reasons.append(f"Company relevance — {relevance_reason}")

        # Gate 1: board relevance + strategic impact
        if f.board_relevance < 8 or f.strategic_impact < 8:
            reasons.append(f"Low scores — board_relevance={f.board_relevance}, strategic_impact={f.strategic_impact} (min 8 each)")

        # Gate 2: reject low trust sources
        # PRIMARY SOURCE always passes — it's a direct company disclosure
        if "LOW TRUST" in f.source_trust.upper():
            reasons.append("Source is LOW TRUST")

        # Gate 3: confidence threshold
        confidence = calculate_confidence(f.source_trust, f.board_relevance, f.strategic_impact)
        # PRIMARY SOURCE gets a relaxed threshold (60 vs 70) — management guidance is valuable
        # even when board_relevance or strategic_impact was scored conservatively
        conf_threshold = 60 if "PRIMARY SOURCE" in f.source_trust.upper() else 70
        if confidence < conf_threshold:
            reasons.append(f"Confidence {confidence}% < {conf_threshold}% threshold")

        # Gate 4: undated fact from non-high-trust source
        # PRIMARY SOURCE and HIGH TRUST are exempt — concall/AR dates are often implicit
        if f.date_signal == "Undated" and f.source_trust.upper() not in ["PRIMARY SOURCE", "HIGH TRUST"]:
            reasons.append("Undated fact from non-HIGH TRUST source")

        # NEW Gate 5: fact quality score
        fqs, fqs_breakdown = calculate_fact_quality_score(
            f.fact, f.source_trust, f.board_relevance, f.strategic_impact, f.date_signal
        )
        if fqs < FACT_QUALITY_THRESHOLD:
            reasons.append(f"Fact quality score {fqs}/100 < {FACT_QUALITY_THRESHOLD} threshold (specificity/recency too low)")

        if reasons:
            rejected.append({"fact": f.fact[:120], "reasons": reasons, "fact_quality_score": fqs})
            continue

        verified.append(ValidatedFact(
            category=f.category,
            fact=f.fact,
            source_url=f.source_url,
            source_trust=f.source_trust.upper(),
            date_signal=f.date_signal,
            board_relevance=f.board_relevance,
            strategic_impact=f.strategic_impact,
            confidence=confidence,
            fact_quality_score=fqs,
            quality_breakdown=fqs_breakdown,
        ))

    return verified, rejected


def run_signal_detector(company: str, verified_facts: List[ValidatedFact]) -> List[StrategicSignal]:
    if not verified_facts: return []
    fact_text = "\n".join([f"[{f.category}] FACT: {f.fact}" for f in verified_facts])
    prompt = f"""You are a Strategic Signal Detector.
Return a JSON object:
{{
  "signals": [
    {{
      "signal_type": "Emerging Threat, Strategic Inflection, Moat Erosion, etc.",
      "signal": "what is specifically changing — must reference a validated fact",
      "urgency": "IMMEDIATE, 90-DAY, 6-MONTH, or WATCH",
      "implication": "which specific decision or resource allocation is affected"
    }}
  ]
}}
Validated Facts:
{fact_text}"""
    try:
        data = invoke_json(prompt)
        signals = []
        for s in data.get("signals", []):
            try: signals.append(StrategicSignal(**s))
            except Exception: continue
        return signals
    except Exception:
        return []


# NEW: Requirement 8 — score options numerically before selecting
def score_options_deterministically(options: List[EvaluatedOption]) -> List[EvaluatedOption]:
    """Compute composite_score (6-dim formula) for every option and sort descending."""
    scored = []
    for opt in options:
        score = calculate_option_score(
            opt.evidence_support_score,
            opt.strategic_fit_score,
            opt.opportunity_score,
            opt.urgency_score,
            opt.risk_score,
            opt.complexity_score,
        )
        opt.composite_score = score
        scored.append(opt)
    scored.sort(key=lambda x: x.composite_score, reverse=True)
    return scored


def run_expert_reasoner(
    company: str, entity: EntityProfile, verified_facts: List[ValidatedFact],
    signals: List[StrategicSignal], evidence_sufficient: bool, sufficiency_message: str
) -> Optional[DecisionIntelligenceBrief]:

    fact_text   = "\n".join([f"- [{f.category}] {f.fact} (Trust: {f.source_trust}, Quality: {f.fact_quality_score}/100)"
                              for f in verified_facts]) if verified_facts else "INSUFFICIENT EVIDENCE."
    signal_text = "\n".join([f"- [{s.urgency}] {s.signal}" for s in signals]) or "No validated signals."

    prompt = f"""# SYSTEM INSTRUCTIONS: EVIDENCE-BASED REASONING ENGINE

## ROLE
You are a strict Evidence-Based Reasoning Engine. Maximize accuracy, traceability, and decision reliability.
Output ONLY what is directly supported by the evidence provided. Never speculate beyond evidence.

## REQUIRED TRACEABILITY CHAIN
[Evidence] -> [Observation] -> [Root Cause] -> [Inference] -> [Theme] -> [Options] -> [Decision]
Every item in the output must trace back to this chain.

---

## GATE 0 — DATA SUFFICIENCY
If evidence_sufficient is False: set status to INSUFFICIENT_EVIDENCE, populate reason, leave all arrays empty.

## GATE 1 — OBSERVATION PURITY (CRITICAL)
Observations MUST only restate what the evidence shows. They MUST NOT contain:
- Any of these words: because, therefore, suggests, indicates, implies, means that, as a result, due to, caused by, which shows, consequently, hence, thus, leading to, resulting in
- Any causal reasoning
- Any forward-looking language
Violation = observation is rejected. Root cause handles the WHY.

METRIC PRESERVATION (MANDATORY):
Never substitute one metric for another in an observation.
If evidence says PAT grew → observation says PAT grew. NOT "revenue grew" or "profits improved".
If evidence says market share declined → observation says market share declined. NOT "revenue declined".
The exact metric identity from evidence MUST be preserved in the observation.

## GATE 2 — ROOT CAUSE RULE
Root cause must explain WHY the observation occurred. If evidence does not support a cause, write UNKNOWN.
Never copy the observation into the root cause field.

## GATE 3 — INFERENCE QUALITY (MANDATORY)
Inferences MUST explain the strategic significance of the observation — NOT rephrase it.
REJECTED inference: Observation says "Revenue increased 12%." — Inference says "Revenue growth." ← REJECTED
REQUIRED inference: Observation says "Revenue increased 12%." — Inference says "Demand and portfolio performance appear strong. | LIKELY"
Rules:
- Every inference MUST end with: | CONFIRMED, | LIKELY, or | HYPOTHESIS
  • CONFIRMED: directly stated in evidence with high-trust source
  • LIKELY: strongly implied by evidence from medium-or-higher trust source
  • HYPOTHESIS: logical but not directly evidenced
- If the inference merely renames or shortens the observation: REGENERATE.
- Inferences must contain at least one significance word: signal, pressure, advantage, risk, opportunity, challenge, momentum, competitive, structural, erosion, expansion, discipline, exposure, demand, capacity.

## GATE 4 — THEME EVIDENCE REQUIREMENT & CLASSIFICATION
A theme requires MINIMUM 2 observations OR 3 facts from verified evidence → classify as STRATEGIC THEME.
If only 1 observation supports it → classify as EMERGING SIGNAL (not STRATEGIC THEME).
Observation restatements cannot become themes.
GOOD theme names: "Brand-Led Growth", "Margin Expansion Under Pressure", "Competitive Erosion", "Capital Discipline".
BAD theme names (bare labels — REJECTED): "Revenue Growth", "Profitability", "Market Share", "Growth", "Expansion".
A theme must describe a pattern or trajectory, not a category.

## GATE 5 — COMPETITOR EVIDENCE RULE
Every competitor advantage and vulnerability claim MUST include a specific evidence citation.
If competitive evidence is unavailable, write exactly: INSUFFICIENT_COMPETITIVE_EVIDENCE
Do NOT invent competitors. Do NOT use prior knowledge. Do NOT leave evidence fields null if a claim is made.

## GATE 6 — OPTION GENERATION & SCORING (6 DIMENSIONS, 1-10 EACH)
Generate exactly 3 options: Conservative, Balanced, Aggressive.
CRITICAL — MATERIAL DIFFERENTIATION REQUIRED:
- Conservative = Protect existing position (defend moats, lock in existing revenue, reduce exposure)
- Balanced     = Optimize existing position (improve efficiency, deepen existing advantages, reduce cost)
- Aggressive   = Create new strategic advantage (enter new markets, acquire, disrupt, build new capability)
These must be MATERIALLY different strategies — NOT the same strategy at different intensity levels.

Score each dimension 1-10:
- evidence_support_score: How strongly does verified evidence support this option?
- strategic_fit_score: How well does it fit the entity's strategic context?
- opportunity_score: How large is the opportunity this option captures?
- urgency_score: How time-sensitive is execution? (higher = more urgent)
- risk_score: How high is execution risk? (higher = riskier)
- complexity_score: How complex is execution? (higher = more complex)

Scoring formula: (0.25 × Evidence) + (0.20 × Strategic Fit) + (0.25 × Opportunity) + (0.15 × Urgency) - (0.10 × Risk) - (0.05 × Complexity)
Aggressive options CAN win when opportunity and urgency are high — the formula allows this.
Do NOT select a recommendation until all three options are scored. Pick the highest composite scorer that passes the generic test.

## GATE 7 — GENERIC RECOMMENDATION REJECTION & SPECIFICITY TEST
The recommendation MUST be rejected and rewritten if it contains ANY of:
leverage synergies, best practices, holistic approach, consider expanding, may wish to,
explore opportunities, strengthen positioning, invest in capabilities, could potentially, it is recommended that

SPECIFICITY TEST (MANDATORY): Ask yourself — could this recommendation apply to 50 unrelated companies?
If YES → REJECT. Rewrite anchoring to specific evidence: a named metric, a percentage, a named product/market, a specific period.
A valid recommendation must contain at least one company-specific anchor (number, %, named entity, specific date/period).

## GATE 8 — DECISION TRACEABILITY (MANDATORY FORMAT)
The recommended_decision field MUST use this exact format:
"Based on Obs: [observation], Inf: [inference], Theme: [theme name], Opt: [option type]: [specific action]"

## GATE 9 — LAYER DIFFERENTIATION (CRITICAL)
Every layer must add new value. If three layers describe the same thing with different words: REGENERATE.
- Observation = what happened (pure restatement of evidence)
- Inference = what it means (strategic significance, forward implication)
- Theme = what broader pattern it represents across multiple observations
Example of VIOLATION: Obs: "Revenue up 12%." / Inf: "Revenue increased." / Theme: "Revenue Growth." ← ALL REJECTED
Example of VALID: Obs: "Revenue up 12% YoY." / Inf: "Broad-based demand and pricing power appear intact. | LIKELY" / Theme: "Portfolio-Driven Revenue Resilience"

## GATE 10 — CONFIDENCE CALIBRATION
Set confidence_assessment using this deterministic scale (do not inflate):
- LOW: 1-2 verified facts
- MEDIUM: 3-4 verified facts
- MEDIUM-HIGH: 5-7 verified facts
- HIGH: 8+ verified facts with multiple independent high-trust sources
Never write HIGH confidence unless 8+ facts from 2+ high-trust independent sources are present.
Format: "Confidence: [LABEL] — [N] verified facts, [X] high-trust sources. [One sentence on reliability]."

---
Evidence Sufficiency: {'SUFFICIENT' if evidence_sufficient else 'INSUFFICIENT_EVIDENCE'} ({sufficiency_message})

Verified Facts (with quality scores):
{fact_text}

Strategic Signals:
{signal_text}

Entity Context: {entity.canonical_name} | {entity.industry} | {entity.primary_market}

OUTPUT STRICT JSON:
{{
  "status": "SUFFICIENT or INSUFFICIENT_EVIDENCE",
  "reason": "...",
  "evidence_and_observation_log": [
    {{
      "evidence": "direct quote or paraphrase of the verified fact",
      "observation": "what happened — NO reasoning words",
      "root_cause": "why it happened — or UNKNOWN",
      "inference": "strategic meaning — must explain significance, not rephrase | CONFIRMED/LIKELY/HYPOTHESIS",
      "observation_purity_passed": true,
      "inference_classified": true
    }}
  ],
  "strategic_themes_and_signals": [
    {{
      "name": "Descriptive pattern name — NOT a bare label",
      "type": "STRATEGIC THEME (if 2+ obs) or EMERGING SIGNAL (if 1 obs)",
      "traceability": ["Observation 1 reference", "Observation 2 reference"]
    }}
  ],
  "competitive_landscape": [
    {{
      "competitor": "name — only if found in evidence",
      "advantage": "specific advantage or null",
      "advantage_evidence": "specific evidence citation — REQUIRED if advantage stated — write INSUFFICIENT_COMPETITIVE_EVIDENCE if unavailable",
      "vulnerability": "specific vulnerability or null",
      "vulnerability_evidence": "specific evidence citation — REQUIRED if vulnerability stated — write INSUFFICIENT_COMPETITIVE_EVIDENCE if unavailable"
    }}
  ],
  "evaluated_options": [
    {{
      "option_type": "Conservative",
      "option_strategy": "Protect existing position — [specific what to protect]",
      "description": "specific action unique to this entity and situation — anchored to evidence",
      "traceability_chain": "Theme [X] -> Inference [Y] -> Observation [Z]",
      "evidence_support_score": 7,
      "strategic_fit_score": 8,
      "opportunity_score": 4,
      "urgency_score": 5,
      "risk_score": 3,
      "complexity_score": 4,
      "composite_score": 0,
      "generic_test_passed": "Yes",
      "rejection_reason": null
    }},
    {{
      "option_type": "Balanced",
      "option_strategy": "Optimize existing position — [specific what to optimize]",
      "description": "...",
      "traceability_chain": "...",
      "evidence_support_score": 6,
      "strategic_fit_score": 7,
      "opportunity_score": 6,
      "urgency_score": 6,
      "risk_score": 5,
      "complexity_score": 5,
      "composite_score": 0,
      "generic_test_passed": "Yes",
      "rejection_reason": null
    }},
    {{
      "option_type": "Aggressive",
      "option_strategy": "Create new strategic advantage — [specific new advantage to create]",
      "description": "...",
      "traceability_chain": "...",
      "evidence_support_score": 5,
      "strategic_fit_score": 6,
      "opportunity_score": 9,
      "urgency_score": 8,
      "risk_score": 8,
      "complexity_score": 7,
      "composite_score": 0,
      "generic_test_passed": "Yes",
      "rejection_reason": null
    }}
  ],
  "recommended_decision": "Based on Obs: [...], Inf: [...], Theme: [...], Opt: [...]: [specific action]",
  "selected_option_type": "Conservative/Balanced/Aggressive",
  "selection_rationale": "Why this option scored highest and why the other two were rejected.",
  "contradicting_evidence": "any evidence that contradicts the recommendation or None",
  "confidence_assessment": "Confidence: LOW/MEDIUM/MEDIUM-HIGH/HIGH — N verified facts, X high-trust sources. [One sentence on reliability]."
}}"""

    try:
        data = invoke_json(prompt)
        brief = DecisionIntelligenceBrief(**data)

        # Post-process: compute composite scores deterministically
        brief.evaluated_options = score_options_deterministically(brief.evaluated_options)

        return brief
    except Exception as e:
        st.error(f"Reasoning Engine error: {e}")
        return None


# ==========================================
# 8. STREAMLIT UI
# ==========================================
company = st.text_input("Target Company / Entity:", placeholder="e.g. Zomato, Reliance, Tesla...")

if st.button("Run Evidence-Based Reasoning", type="primary"):
    if not company:
        st.error("Please enter an entity name.")
    else:
        with st.status(f"Executing Decision Pipeline for {company}...", expanded=True) as status:

            st.write("📡 Searching for evidence...")
            raw_context = run_enhanced_search(company)
            if not raw_context:
                st.error("Search returned no data.")
                st.stop()
            time.sleep(1)

            st.write("🔍 Resolving entity...")
            entity = run_entity_resolution(company, raw_context)
            entity_conf, entity_conf_msg = calculate_entity_confidence(entity)
            if entity_conf < ENTITY_CONFIDENCE_THRESHOLD:
                st.warning(f"⚠️ Low Entity Confidence — {entity_conf_msg}")
            time.sleep(1)

            st.write("📊 Extracting facts...")
            raw_facts = run_researcher(company, entity, raw_context[:15000])

            st.write("🔒 Running quality gate — scoring and rejecting low-value facts...")
            verified_facts, rejected_facts = run_hard_gate_validation(raw_facts, entity.canonical_name)

            # Semantic deduplication — remove facts expressing the same idea
            st.write("🔁 Running semantic deduplication...")
            verified_facts, dup_log = deduplicate_facts(verified_facts)

            report_confidence_prelim = calculate_report_confidence(verified_facts, len(raw_facts))
            evidence_sufficient, sufficiency_message = get_evidence_sufficiency(verified_facts, report_confidence_prelim)
            if not evidence_sufficient:
                st.warning(f"⚠️ Evidence Warning: {sufficiency_message}")

            st.write("🔭 Detecting strategic signals...")
            signals = run_signal_detector(company, verified_facts)
            time.sleep(1)

            st.write("⚖️ Running reasoning engine with traceability enforcement...")
            final_brief = run_expert_reasoner(
                company, entity, verified_facts, signals,
                evidence_sufficient, sufficiency_message
            )

            status.update(label="Analysis Complete", state="complete")

        if not final_brief:
            st.error("Reasoning Engine failed to produce valid JSON. Try again.")
            st.stop()

        # ==========================================
        # DISPLAY LAYER
        # ==========================================
        st.divider()
        st.header(f"Decision Validation Brief — {entity.canonical_name.upper()}")
        st.caption(f"**Entity:** {entity.industry} | {entity.sector} | {entity.primary_market}")

        # ── Data sufficiency gate display ─────────────────────
        if final_brief.status == "INSUFFICIENT_EVIDENCE":
            st.error("🛑 DATA SUFFICIENCY GATE FAILED")
            st.warning(f"**Reason:** {final_brief.reason or 'Insufficient evidence.'}")
            st.info("Reliable conclusions cannot be generated. Analysis aborted.")
            st.stop()
        else:
            st.success(f"✅ DATA SUFFICIENCY GATE PASSED — {final_brief.reason or 'Evidence meets threshold.'}")

        # ── NEW: Fact Quality Report ─────────────────────────
        with st.expander(f"📊 Fact Quality Gate Report — {len(verified_facts)} passed / {len(rejected_facts)} rejected / {len(dup_log)} deduplicated", expanded=False):
            col_pass, col_fail = st.columns(2)

            with col_pass:
                st.markdown("**✅ Verified Facts**")
                for vf in verified_facts:
                    with st.container(border=True):
                        st.markdown(f"**[{vf.category}]** {vf.fact[:120]}...")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Quality", f"{vf.fact_quality_score}/100")
                        m2.metric("Confidence", f"{vf.confidence}%")
                        m3.metric("Trust", vf.source_trust.replace(" TRUST", ""))
                        with st.expander("Quality breakdown"):
                            for k, v in vf.quality_breakdown.items():
                                st.write(f"• {k}: {v}")

            with col_fail:
                st.markdown("**❌ Rejected Facts**")
                if not rejected_facts:
                    st.info("No facts rejected.")
                for rf in rejected_facts:
                    with st.container(border=True):
                        st.markdown(f"`{rf['fact']}`")
                        st.caption(f"Quality score: {rf['fact_quality_score']}/100")
                        for r in rf["reasons"]:
                            st.error(f"• {r}")

            if dup_log:
                st.markdown("**🔁 Semantic Duplicates Removed**")
                for dl in dup_log:
                    with st.container(border=True):
                        st.warning(f"**Removed:** `{dl['rejected_fact']}`")
                        st.caption(f"Duplicate of: `{dl['duplicate_of']}`")
                        st.caption(dl["reason"])

        # ── NEW: Traceability Validation Report ──────────────
        violations = validate_traceability_chain(final_brief, verified_facts)
        if violations:
            with st.expander(f"⚠️ Traceability Violations ({len(violations)} found)", expanded=True):
                for v in violations:
                    st.warning(f"• {v}")
        else:
            st.success("✅ Traceability Chain Validated — no violations detected.")

        # 1. EVIDENCE, OBSERVATION & INFERENCE LOG
        st.markdown("### 1. Evidence → Observation → Inference Log")
        for i, log in enumerate(final_brief.evidence_and_observation_log):
            with st.container(border=True):
                st.markdown(f"**Evidence:** `{log.evidence or 'N/A'}`")

                # Show observation purity status
                has_reasoning, reasoning_msg = contains_reasoning(log.observation or "")
                if has_reasoning:
                    st.error(f"❌ **Observation (PURITY FAILED):** {log.observation or 'N/A'}\n\n_{reasoning_msg}_")
                else:
                    st.info(f"✅ **Observation (pure):** {log.observation or 'N/A'}")

                # Metric preservation check
                metric_violated, metric_msg = check_metric_preservation(log.evidence or "", log.observation or "")
                if metric_violated:
                    st.error(f"❌ **Metric Substitution:** {metric_msg}")

                c1, c2 = st.columns(2)
                with c1:
                    st.warning(f"**Root Cause:** {log.root_cause or 'UNKNOWN'}")
                with c2:
                    # Show inference classification
                    inf = log.inference or ""
                    classification = inf.split("|")[-1].strip().upper() if "|" in inf else "UNCLASSIFIED"
                    badge_color = "green" if classification == "CONFIRMED" else "orange" if classification == "LIKELY" else "red" if classification == "HYPOTHESIS" else "gray"
                    st.markdown(f"**Inference:** {inf}")
                    st.markdown(f"**Classification:** :{badge_color}[{classification}]")

                    # Req 3 — inference quality check
                    rephrased, rephrase_msg = inference_merely_rephrases(log.observation or "", inf)
                    if rephrased:
                        st.error(f"❌ **Inference Quality Failed:** {rephrase_msg}")
                    else:
                        st.success("✅ Inference adds explanatory value beyond observation.")

                    # Req 12 — layer differentiation check
                    for tname in [ts.name or "" for ts in final_brief.strategic_themes_and_signals]:
                        layer_issues = check_layer_differentiation(log.observation or "", inf, tname)
                        for issue in layer_issues:
                            st.warning(f"⚠️ {issue}")

        # 2. STRATEGIC THEMES & SIGNALS
        st.markdown("### 2. Strategic Themes & Signals")
        c1, c2 = st.columns(2)
        for i, ts in enumerate(final_brief.strategic_themes_and_signals):
            col = c1 if i % 2 == 0 else c2
            with col.container(border=True):
                st.subheader(ts.name or "Unnamed Theme")
                type_val   = ts.type or "UNKNOWN"
                trace_count = len(ts.traceability)

                # Req 5 — enforce STRATEGIC THEME vs EMERGING SIGNAL threshold
                if "THEME" in type_val.upper() and trace_count < 2:
                    st.error(
                        f"⚠️ Threshold Violation: Classified as STRATEGIC THEME "
                        f"but only {trace_count} traceability reference(s). "
                        "Must be downgraded to EMERGING SIGNAL."
                    )
                    type_val = "EMERGING SIGNAL ⚠️ (reclassified — insufficient evidence)"
                    type_color = "red"
                elif "THEME" in type_val.upper():
                    type_color = "green"
                else:
                    type_color = "orange"

                st.markdown(f"**Type:** :{type_color}[{type_val}]")
                if trace_count < 2:
                    st.error(f"⚠️ Only {trace_count} traceability reference(s) — minimum 2 required for STRATEGIC THEME")
                else:
                    st.success(f"✅ {trace_count} traceability references")
                st.markdown("**Traceability:**")
                for trace in ts.traceability:
                    st.markdown(f"- {trace}")

        # 3. COMPETITIVE LANDSCAPE
        st.markdown("### 3. Competitive Landscape (Evidence-Backed Only)")
        for comp in final_brief.competitive_landscape:
            with st.container(border=True):
                st.markdown(f"**Competitor:** {comp.competitor or 'N/A'}")
                c_adv, c_vuln = st.columns(2)
                with c_adv:
                    st.success(f"**Advantage:** {comp.advantage or 'None explicitly supported'}")
                    adv_evidence = comp.advantage_evidence or ""
                    if comp.advantage and "INSUFFICIENT_COMPETITIVE_EVIDENCE" in adv_evidence.upper():
                        st.warning("⚠️ INSUFFICIENT_COMPETITIVE_EVIDENCE — advantage stated but no evidence available.")
                    elif comp.advantage and not adv_evidence:
                        st.error("❌ No evidence backing for this advantage claim")
                    elif adv_evidence:
                        st.caption(f"**Evidence:** {adv_evidence}")
                with c_vuln:
                    st.error(f"**Vulnerability:** {comp.vulnerability or 'None explicitly supported'}")
                    vuln_evidence = comp.vulnerability_evidence or ""
                    if comp.vulnerability and "INSUFFICIENT_COMPETITIVE_EVIDENCE" in vuln_evidence.upper():
                        st.warning("⚠️ INSUFFICIENT_COMPETITIVE_EVIDENCE — vulnerability stated but no evidence available.")
                    elif comp.vulnerability and not vuln_evidence:
                        st.error("❌ No evidence backing for this vulnerability claim")
                    elif vuln_evidence:
                        st.caption(f"**Evidence:** {vuln_evidence}")

        # 4. EVALUATED OPTIONS WITH NUMERIC SCORES
        st.markdown("### 4. Evaluated Options (Scored & Ranked)")
        st.caption("Options ranked by composite score (descending). Only options passing the generic test are selectable.")

        for rank, opt in enumerate(final_brief.evaluated_options):
            opt_type   = opt.option_type or "Unknown"
            color      = "blue" if "Conservative" in opt_type else "orange" if "Balanced" in opt_type else "red"
            is_selected = opt_type == final_brief.selected_option_type
            rank_label  = "🏆 SELECTED" if is_selected else f"#{rank+1}"

            with st.container(border=True):
                header_col, score_col = st.columns([3, 1])
                with header_col:
                    st.markdown(f"**{rank_label} — :{color}[{opt_type}]**")
                    if opt.option_strategy:
                        st.caption(f"🎯 Strategy posture: _{opt.option_strategy}_")
                    st.markdown(f"{opt.description or 'N/A'}")
                with score_col:
                    st.metric("Composite Score", f"{opt.composite_score}/100")

                # 6-dimension scores
                sc1, sc2, sc3 = st.columns(3)
                sc4, sc5, sc6 = st.columns(3)
                sc1.metric("Evidence Support", f"{opt.evidence_support_score}/10")
                sc2.metric("Strategic Fit", f"{opt.strategic_fit_score}/10")
                sc3.metric("Opportunity ↑", f"{opt.opportunity_score}/10")
                sc4.metric("Urgency ↑", f"{opt.urgency_score}/10")
                sc5.metric("Risk (↑=worse)", f"{opt.risk_score}/10")
                sc6.metric("Complexity (↑=harder)", f"{opt.complexity_score}/10")

                chain = opt.traceability_chain
                if isinstance(chain, list):
                    chain = " -> ".join(chain)
                st.info(f"**Traceability:** {chain or 'N/A'}")

                if opt.generic_test_passed == "Yes":
                    st.success("✅ Passed Specificity Test — specific to this entity and situation.")
                elif opt.generic_test_passed == "No":
                    st.error(f"❌ Failed Specificity Test — {opt.rejection_reason or 'Too generic.'}")

        # 5. FINAL DECISION
        st.markdown("### 5. Final Decision & Integrity Check")
        with st.container(border=True):
            st.subheader("Recommended Decision")
            decision = final_brief.recommended_decision or ""

            if decision:
                generic_count, generic_found = count_generic_phrases(decision)
                if generic_count >= GENERIC_WORD_THRESHOLD:
                    st.error(f"❌ Decision contains generic language: {', '.join(generic_found)}")
                else:
                    st.success(decision)

                if final_brief.selection_rationale:
                    st.info(f"**Selection rationale:** {final_brief.selection_rationale}")

                st.caption(
                    "*Integrity: recommendation must explicitly reference "
                    "Obs: (Observation), Inf: (Inference), Theme: (Theme name), Opt: (Option type)*"
                )
                # Check all 4 required tags
                for tag in ["Obs:", "Inf:", "Theme:", "Opt:"]:
                    present = tag.lower() in decision.lower()
                    icon    = "✅" if present else "❌"
                    st.caption(f"{icon} {tag} reference {'found' if present else 'MISSING'}")
            else:
                st.write("No recommendation generated.")

            st.divider()
            st.markdown("**Contradicting Evidence:**")
            contradicting = final_brief.contradicting_evidence or "None explicitly noted."
            if "none" in contradicting.lower():
                st.info(contradicting)
            else:
                st.warning(contradicting)

            st.markdown("**Confidence Assessment:**")
            st.markdown(f"`{final_brief.confidence_assessment or 'N/A'}`")

            # Req 11 — deterministic confidence label (overrides LLM-generated confidence)
            conf_label, conf_explanation = calibrate_confidence_label(verified_facts)
            conf_color = {"HIGH": "green", "MEDIUM-HIGH": "blue", "MEDIUM": "orange", "LOW": "red"}.get(conf_label, "gray")
            st.markdown(f"**Calibrated Evidence Confidence:** :{conf_color}[{conf_label}]")
            st.caption(f"_{conf_explanation}_")

        # Export — now includes quality scores and rejection log
        st.divider()
        export = {
            "entity_profile": entity.model_dump(),
            "fact_quality_report": {
                "verified_facts": [vf.model_dump() for vf in verified_facts],
                "rejected_facts": rejected_facts,
                "deduplicated_facts": dup_log,
                "report_confidence": report_confidence_prelim,
            },
            "reasoning_brief": final_brief.model_dump(),
            "traceability_violations": violations,
        }
        st.download_button(
            "⬇️ Download Full Decision Package (JSON)",
            data=json.dumps(export, indent=2, ensure_ascii=False),
            file_name=f"{company.replace(' ', '_')}_decision_package.json",
            mime="application/json"
        )