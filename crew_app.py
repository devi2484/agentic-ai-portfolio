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
st.markdown("**Institutional Evidence-Based Decision Support System** · Filing & Concall Traceability · Scored Options")
st.divider()

# ==========================================
# 2. TRUST & SCORING
# ==========================================
HIGH_TRUST_DOMAINS = [
    "reuters.com", "bloomberg.com", "cnbc.com", "wsj.com", "ft.com", "sec.gov",
    "moneycontrol.com", "economictimes.indiatimes.com", "livemint.com",
    "businessstandard.com", "thehindubusinessline.com", "financialexpress.com",
    "bseindia.com", "nseindia.com", "sebi.gov.in", "rbi.org.in",
    "hbr.org", "mckinsey.com", "bain.com", "bcg.com", "economist.com",
    "statista.com", "nyse.com", "nasdaq.com"
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

TRUST_SCORE_MAP = {"HIGH TRUST": 10, "MEDIUM TRUST": 6, "LOW TRUST": 2}

# ==========================================
# THRESHOLDS — tunable quality gates
# ==========================================
MIN_VERIFIED_FACTS            = 2
MIN_REPORT_CONFIDENCE         = 50
ENTITY_CONFIDENCE_THRESHOLD   = 60
FACT_QUALITY_THRESHOLD        = 55   
OPTION_SCORE_THRESHOLD        = 30   
GENERIC_WORD_THRESHOLD        = 2    

# ==========================================
# GENERIC LANGUAGE DETECTOR
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

REJECT_CONTENT_PATTERNS = [
    r'\b(our mission|our vision|our purpose|we believe|we strive|we are committed)\b',
    r'\b(company overview|about us|who we are|our story|founded in)\b',
    r'\b(world.?class|industry.?leading|best.?in.?class|leading provider|trusted partner)\b',
    r'\b(dedicated to|passionate about|committed to excellence|customer.?centric)\b',
    r'\b(innovative solutions?|cutting.?edge|state.?of.?the.?art|next.?generation)\b',
    r'\b(seamless experience|transforming the way|reimagining|revolutionizing)\b',
]

PREFERRED_CONTENT_KEYWORDS = [
    "revenue", "profit", "margin", "ebitda", "earnings", "net income",
    "market share", "growth rate", "capex", "acquisition", "divestiture",
    "launched", "partnership", "regulatory", "compliance", "penalty",
    "crore", "billion", "million", "lakh", "percent", "%", "₹", "$", "€",
    "quarter", "annual", "fiscal", "q1", "q2", "q3", "q4", "guidance", "concall", "transcript"
]

# ==========================================
# METRIC PRESERVATION
# ==========================================
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
    tl = text.lower()
    for key in sorted(METRIC_IDENTITY_MAP.keys(), key=len, reverse=True):
        if key in tl:
            return key
    return None

def check_metric_preservation(evidence: str, observation: str) -> tuple[bool, str]:
    if not evidence or not observation:
        return False, ""
    ev_metric  = detect_metric_in_text(evidence)
    obs_metric = detect_metric_in_text(observation)
    if not ev_metric or not obs_metric:
        return False, ""
    if ev_metric == obs_metric:
        return False, ""
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
# ==========================================
INDUSTRY_TREND_MARKERS = [
    r'\b(the industry|the sector|the market|industry as a whole|sector wide|across the industry)\b',
    r'\b(globally|worldwide|industry players|market participants|analysts expect|experts predict)\b',
    r'\b(the overall market|broader market|industry average|sector average|peer group)\b',
    r'\b(it is expected|it is projected|forecasters|research firms predict)\b',
]

def check_company_relevance(fact_text: str, canonical_name: str) -> tuple[bool, str]:
    text_lower = fact_text.lower()
    name_lower = canonical_name.lower()
    name_core = name_lower
    for suffix in [" limited", " ltd", " inc", " corp", " group", " pvt", " plc"]:
        name_core = name_core.replace(suffix, "")
    name_core = name_core.strip()

    company_mentioned = name_core in text_lower or name_lower in text_lower
    has_trend_language = any(re.search(pattern, text_lower) for pattern in INDUSTRY_TREND_MARKERS)

    if has_trend_language and not company_mentioned:
        return True, (
            f"Fact describes an industry/market trend without explicitly linking to "
            f"'{canonical_name}'. Only company-specific facts are admitted."
        )
    return False, ""

# ==========================================
# SEMANTIC DEDUPLICATION
# ==========================================
def semantic_overlap_score(text_a: str, text_b: str) -> float:
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
    import re as _re
    text_lower = fact_text.lower()
    for pattern in REJECT_CONTENT_PATTERNS:
        if _re.search(pattern, text_lower):
            return True, f"Non-decision content pattern detected: '{pattern}'"
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
# INFERENCE QUALITY VALIDATOR
# ==========================================
def inference_merely_rephrases(observation: str, inference: str) -> tuple[bool, str]:
    if not observation or not inference:
        return False, ""

    inf_clean = inference.split("|")[0].strip().lower()
    obs_clean = observation.lower()

    obs_words = set(obs_clean.split())
    inf_words = set(inf_clean.split())

    stopwords = {"the", "a", "an", "is", "are", "was", "were", "has", "have",
                 "had", "in", "of", "to", "for", "and", "or", "but", "its",
                 "their", "this", "that", "with", "by", "at", "on", "from"}
    obs_content = obs_words - stopwords
    inf_content = inf_words - stopwords

    if not obs_content:
        return False, ""

    overlap = len(obs_content & inf_content) / len(obs_content)

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
# LAYER DIFFERENTIATION
# ==========================================
def check_layer_differentiation(observation: str, inference: str, theme_name: str = "") -> list[str]:
    issues = []
    if not observation or not inference:
        return issues

    inf_clean = inference.split("|")[0].strip().lower()

    rephrased, msg = inference_merely_rephrases(observation, inference)
    if rephrased:
        issues.append(f"Layer violation — Inference is a rephrasing of Observation: {msg}")

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
    domain = urlparse(url).netloc.lower().replace("www.", "")
    if company:
        clean_company = company.lower()
        for stopword in ["the", "group", "inc", "ltd", "llc", "corp", "co", "pvt", "plc", "incorporated", "limited"]:
            clean_company = clean_company.replace(stopword, "")
        clean_company = clean_company.strip().replace(" ", "")
        if clean_company and clean_company in domain.replace("-", ""):
            return "HIGH TRUST"
    if any(h in domain for h in HIGH_TRUST_DOMAINS):   return "HIGH TRUST"
    if any(m in domain for m in MEDIUM_TRUST_DOMAINS): return "MEDIUM TRUST"
    if any(l in domain for l in LOW_TRUST_DOMAINS):    return "LOW TRUST"
    return "MEDIUM TRUST"

# ==========================================
# DETERMINISTIC BACKEND SCORING
# ==========================================
def calculate_confidence(trust_label: str, board_relevance: int, strategic_impact: int) -> int:
    trust_score = TRUST_SCORE_MAP.get(trust_label.strip().upper(), 5)
    raw = (trust_score * 0.4) + (board_relevance * 0.3) + (strategic_impact * 0.3)
    return int((raw / 10) * 100)

def calculate_fact_quality_score(fact_text: str, source_trust: str,
                                  board_relevance: int, strategic_impact: int,
                                  date_signal: str) -> tuple[int, dict]:
    breakdown = {}

    has_numbers = bool(re.search(r'\d', fact_text))
    has_percent = '%' in fact_text
    has_currency = bool(re.search(r'[$₹€£¥]|\b(crore|lakh|billion|million|trillion)\b', fact_text, re.I))
    has_named    = len(fact_text.split()) > 8  

    specificity = 0
    if has_numbers:  specificity += 10
    if has_percent:  specificity += 7
    if has_currency: specificity += 5
    if has_named:    specificity += 3
    breakdown["specificity"] = min(specificity, 25)

    trust_raw = TRUST_SCORE_MAP.get(source_trust.strip().upper(), 5)
    breakdown["source_trust"] = int((trust_raw / 10) * 25)
    breakdown["board_relevance"] = int((board_relevance / 10) * 20)
    breakdown["strategic_impact"] = int((strategic_impact / 10) * 20)
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

def calibrate_confidence_label(verified_facts: list) -> tuple[str, str]:
    n = len(verified_facts)
    high_trust_count = sum(1 for f in verified_facts if "HIGH TRUST" in f.source_trust.upper())
    avg_quality = sum(f.fact_quality_score for f in verified_facts) / n if n else 0

    if n >= 8 and high_trust_count >= 2 and avg_quality >= 65:
        label = "HIGH"
        explanation = f"{n} verified data points from {high_trust_count} institutional repositories."
    elif n >= 5:
        label = "MEDIUM-HIGH"
        explanation = f"{n} corporate filing disclosures structured successfully."
    elif n >= 3:
        label = "MEDIUM"
        explanation = f"{n} baseline disclosures tracked."
    else:
        label = "LOW"
        explanation = "Tentative data structure."

    return label, explanation

def get_evidence_sufficiency(verified_facts: list, report_confidence: int) -> tuple[bool, str]:
    if len(verified_facts) < MIN_VERIFIED_FACTS:
        return False, f"Only {len(verified_facts)} filing facts isolated. Insufficient dataset."
    if report_confidence < MIN_REPORT_CONFIDENCE:
        return False, f"Source dataset composition failed confidence constraints ({report_confidence}%)."
    return True, "Filing evidence parameters fully met."

def calculate_option_score(evidence_support: int, strategic_fit: int,
                            opportunity: int, urgency: int,
                            risk: int, complexity: int) -> int:
    raw = (
        evidence_support * 0.25 +
        strategic_fit    * 0.20 +
        opportunity      * 0.25 +
        urgency          * 0.15 -
        risk             * 0.10 -
        complexity       * 0.05
    )
    raw_min = 1 * 0.85 - 10 * 0.15   
    raw_max = 10 * 0.85 - 1 * 0.15   
    normalised = (raw - raw_min) / (raw_max - raw_min)
    return max(0, min(100, int(normalised * 100)))

# ==========================================
# SPECIFICITY TEST
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
    decision_lower = decision.lower()
    pattern_hits = [p for p in UNIVERSAL_STRATEGY_PATTERNS if re.search(p, decision_lower)]
    if not pattern_hits:
        return False, ""

    has_specifics = bool(re.search(
        r'\d|%|₹|\$|€|crore|billion|million|lakh|q[1-4]|fy\d|fiscal|\b[A-Z][a-z]{3,}\b',
        decision
    ))
    if has_specifics:
        return False, ""  

    return True, (
        f"Specificity test FAILED — recommendation contains {len(pattern_hits)} universal strategy "
        f"pattern(s). Rewrite anchoring to clear metric constraints from the corporate filings."
    )

# ==========================================
# STRATEGIC POSTURE OPTION TEST
# ==========================================
OPTION_STRATEGY_SIGNATURES = {
    "Conservative": {
        "required": ["protect", "defend", "maintain", "retain", "secure", "preserve", "hold", "consolidate", "safeguard", "sustain"],
        "forbidden": ["new market", "new segment", "launch", "acquire", "disrupt", "transform", "new advantage", "aggressive"],
    },
    "Balanced": {
        "required": ["optimis", "optimiz", "improve", "enhance", "efficienc", "strengthen existing", "build on", "leverage existing", "deepen", "expand within"],
        "forbidden": ["protect existing", "defend position", "disrupt", "acquire", "new market", "new category"],
    },
    "Aggressive": {
        "required": ["new", "create", "disrupt", "enter", "acquire", "launch", "advantage", "transform", "pioneer", "first-mover"],
        "forbidden": ["protect", "maintain current", "sustain existing", "hold"],
    },
}

def check_option_differentiation(options) -> list[str]:
    issues = []
    if not options or len(options) < 3:
        return issues

    descriptions = {opt.option_type: (opt.description or "").lower() for opt in options}
    option_types = list(descriptions.keys())
    for i in range(len(option_types)):
        for j in range(i + 1, len(option_types)):
            ta, tb = option_types[i], option_types[j]
            overlap = semantic_overlap_score(descriptions[ta], descriptions[tb])
            if overlap > 0.60:
                issues.append(
                    f"Option differentiation FAILED: '{ta}' and '{tb}' share {int(overlap*100)}% structural overlap."
                )

    for opt_type, sigs in OPTION_STRATEGY_SIGNATURES.items():
        desc = descriptions.get(opt_type, "")
        if not desc:
            continue
        has_required = any(kw in desc for kw in sigs["required"])
        if not has_required:
            issues.append(f"Option '{opt_type}': Metric profile missing intended posture keywords ({', '.join(sigs['required'][:3])}).")

    return issues

# ==========================================
# TRACEABILITY CHAIN LOGIC
# ==========================================
def validate_traceability_chain(brief, verified_facts: list = None) -> list[str]:
    violations = []
    theme_names = [ts.name or "" for ts in brief.strategic_themes_and_signals]

    for i, log in enumerate(brief.evidence_and_observation_log):
        tag = f"Log {i+1}"
        if not log.evidence or len(log.evidence.strip()) < 10:
            violations.append(f"{tag}: Evidence context missing.")
        if not log.observation or len(log.observation.strip()) < 10:
            violations.append(f"{tag}: Observation string missing.")

        has_reasoning, reasoning_msg = contains_reasoning(log.observation or "")
        if has_reasoning:
            violations.append(f"{tag}: Observation contaminated with causal language — {reasoning_msg}")

        metric_violated, metric_msg = check_metric_preservation(log.evidence or "", log.observation or "")
        if metric_violated:
            violations.append(f"{tag}: {metric_msg}")

        if log.inference:
            parts = log.inference.split("|")
            classification = parts[-1].strip().upper() if len(parts) > 1 else ""
            if classification not in ["CONFIRMED", "LIKELY", "HYPOTHESIS"]:
                violations.append(f"{tag}: Inference lacks probability class vector (| CONFIRMED, etc.).")

            rephrased, rephrase_msg = inference_merely_rephrases(log.observation or "", log.inference)
            if rephrased:
                violations.append(f"{tag}: {rephrase_msg}")

            for tname in theme_names:
                layer_issues = check_layer_differentiation(log.observation or "", log.inference, tname)
                for issue in layer_issues:
                    violations.append(f"{tag}: {issue}")
        else:
            violations.append(f"{tag}: Inference link broken.")

    for i, theme in enumerate(brief.strategic_themes_and_signals):
        trace_count = len(theme.traceability)
        theme_type = (theme.type or "").upper()

        if "THEME" in theme_type and trace_count < 2:
            violations.append(f"Theme '{theme.name}': Strategic criteria requires ≥ 2 references. Downgrade target.")
        elif trace_count == 0:
            violations.append(f"Theme '{theme.name}': Orphaned theme block.")

        if theme.name and len(theme.name.split()) <= 2:
            if theme.name.lower().strip() in ["revenue growth", "profitability", "market share", "growth", "innovation", "expansion"]:
                violations.append(f"Theme '{theme.name}': Bare categorical bucket. Needs descriptive trajectory naming.")

    for comp in brief.competitive_landscape:
        comp_name = comp.competitor or "Unknown"
        if comp.advantage and not comp.advantage_evidence:
            violations.append(f"Competitor '{comp_name}': Advantage metadata lacks explicit corporate documentation.")
        if comp.vulnerability and not comp.vulnerability_evidence:
            violations.append(f"Competitor '{comp_name}': Vulnerability metadata lacks explicit corporate documentation.")

    decision = brief.recommended_decision or ""
    if decision:
        for required in ["Observation", "Inference", "Theme", "Option"]:
            if required.lower() not in decision.lower():
                violations.append(f"Decision structural break: Missing '{required}' trace anchor.")

        generic_count, generic_found = count_generic_phrases(decision)
        if generic_count >= GENERIC_WORD_THRESHOLD:
            violations.append(f"Decision: Fails semantic filtering. Generic terminology flagged: {', '.join(generic_found)}.")

        specificity_fail, specificity_reason = check_recommendation_specificity(decision)
        if specificity_fail:
            violations.append(f"Decision: {specificity_reason}")
    else:
        violations.append("Decision target block empty.")

    violations.extend(check_option_differentiation(brief.evaluated_options))
    return violations

# ==========================================
# 4. ENHANCED DEEP FILINGS & CONCALL SEARCH
# ==========================================
def run_enhanced_search(company: str) -> str:
    current_year = datetime.now().year
    queries = [
        f"{company} \"annual report\" OR \"financial statements\" OR \"MD&A\" performance {current_year}",
        f"{company} \"earnings call transcript\" OR \"concall\" transcript {current_year}",
        f"{company} \"investor presentation\" OR \"analyst meet\" strategy metrics {current_year}",
        f"{company} \"annual report\" capex capital allocation leverage",
        f"{company} \"concall\" margin pressure cost headwinds guidance",
        f"{company} regulatory filings \"Form 10-K\" OR \"SEBI\" filings disclosures compliance"
    ]
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=2):
                    url = r.get("href", "")
                    results.append(
                        f"SOURCE: {url}\nTRUST: {evaluate_trust(url, company)}\n"
                        f"CONTENT: {r.get('title','')} — {r.get('body','')}\n{'-'*40}"
                    )
    except Exception as e:
        st.error(f"Filing ingestion execution failure: {e}")
    return "\n".join(results)

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
    date_signal: str = "Undated"
    board_relevance: int
    strategic_impact: int

class ValidatedFact(BaseModel):
    category: str
    fact: str
    source_url: str
    source_trust: str
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
    observation: Optional[str] = Field(default=None)
    root_cause: Optional[str] = Field(default=None)
    inference: Optional[str] = Field(default=None)
    observation_purity_passed: bool = True
    inference_classified: bool = True

class ThemeSignal(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = Field(default=None)
    traceability: List[str] = Field(default_factory=list)

class CompetitiveLandscape(BaseModel):
    competitor: Optional[str] = None
    advantage: Optional[str] = None
    advantage_evidence: Optional[str] = Field(default=None)
    vulnerability: Optional[str] = None
    vulnerability_evidence: Optional[str] = Field(default=None)

class EvaluatedOption(BaseModel):
    option_type: Optional[str] = Field(default=None)
    option_strategy: Optional[str] = Field(default=None)
    description: Optional[str] = None
    traceability_chain: Union[str, List[str], None] = Field(default=None)
    evidence_support_score: int = Field(default=5)
    strategic_fit_score: int    = Field(default=5)
    opportunity_score: int      = Field(default=5)
    urgency_score: int          = Field(default=5)
    risk_score: int             = Field(default=5)
    complexity_score: int       = Field(default=5)
    composite_score: int        = Field(default=0)
    generic_test_passed: Optional[str] = Field(default=None)
    rejection_reason: Optional[str] = Field(default=None)

class DecisionIntelligenceBrief(BaseModel):
    status: str 
    reason: Optional[str] = Field(default=None)
    evidence_and_observation_log: List[EvidenceLog] = Field(default_factory=list)
    strategic_themes_and_signals: List[ThemeSignal] = Field(default_factory=list)
    competitive_landscape: List[CompetitiveLandscape] = Field(default_factory=list)
    evaluated_options: List[EvaluatedOption] = Field(default_factory=list)
    recommended_decision: Optional[str] = Field(default=None)
    selected_option_type: Optional[str] = Field(default=None)
    selection_rationale: Optional[str] = Field(default=None)
    contradicting_evidence: Optional[str] = Field(default=None)
    confidence_assessment: Optional[str] = Field(default=None)

# ==========================================
# 7. PIPELINE AGENTS (FILING FOCUSED)
# ==========================================
FACT_CATEGORIES = [
    "Profitability", "Growth", "Competitive Threat",
    "Competitive Advantage", "Capital Allocation", "Strategic Shift"
]

def run_entity_resolution(company: str, raw_context: str) -> EntityProfile:
    prompt = f"""You are an Entity Resolution Specialist parsing statutory corporate disclosures and call transcripts. Identify exactly which company the text describes.
Return a JSON object:
{{
  "canonical_name": "official registered name",
  "industry": "specific industry",
  "sector": "sector",
  "business_model": "how it makes money",
  "primary_market": "main geography",
  "known_subsidiaries": "subsidiaries or Unknown",
  "known_competitors": "competitors or Unknown",
  "contamination_warnings": "None detected OR describe structural mismatch"
}}
Company queried: {company}
Filings context: {raw_context[:2000]}"""
    try:
        data = invoke_json(prompt)
        return EntityProfile(**data)
    except Exception:
        return EntityProfile(
            canonical_name=company, industry="Unknown", sector="Unknown",
            business_model="Unknown", primary_market="Unknown",
            known_subsidiaries="Unknown", known_competitors="Unknown",
            contamination_warnings="Resolution fault — step validation manually."
        )

def run_researcher(company: str, entity: EntityProfile, raw_context: str) -> List[IntelligenceFact]:
    prompt = f"""You are an Institutional Investment Analyst. Extract highly specific disclosures for {entity.canonical_name} explicitly tracking management narrative, numbers, and guidance from annual reports, financial notes, and investor transcripts.
Extract 4-6 deep, non-obvious factual metrics.

CRITICAL DISCLOSURE RULES:
1. Every fact MUST link directly to corporate reporting syntax: forward-looking target numbers, margin compression basis points, capEx execution milestones, or market compliance penalties.
2. Filter out generic boilerplate marketing statements. Prioritize hard numbers, percentages, currency metrics, or official timeline milestones.
3. Map source URLs precisely.
4. Score board_relevance and strategic_impact (1-10). Baseline accounting metrics, institutional management adjustments, or product segment margins must score 8-10.

Return JSON:
{{
  "facts": [
    {{
      "category": "one of: {', '.join(FACT_CATEGORIES)}",
      "fact": "exact filing fact detailing specific metrics or management commentary transcript quotes — avoid general summary",
      "source_url": "exact absolute URL",
      "source_trust": "HIGH TRUST / MEDIUM TRUST / LOW TRUST",
      "date_signal": "Specific fiscal quarter or year. If completely absent, use Undated.",
      "board_relevance": 9,
      "strategic_impact": 9
    }}
  ]
}}
Raw Filings Context:
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

def run_hard_gate_validation(facts: List[IntelligenceFact], canonical_name: str = "") -> tuple[List[ValidatedFact], List[dict]]:
    verified  = []
    rejected  = []

    for f in facts:
        reasons = []

        is_non_decision, non_decision_reason = is_non_decision_content(f.fact)
        if is_non_decision:
            reasons.append(f"Non-decision-grade content — {non_decision_reason}")

        if canonical_name:
            is_irrelevant, relevance_reason = check_company_relevance(f.fact, canonical_name)
            if is_irrelevant:
                reasons.append(f"Company relevance filter — {relevance_reason}")

        if f.board_relevance < 8 or f.strategic_impact < 8:
            reasons.append(f"Insufficient granularity threshold profile (min 8 requirement)")

        if "LOW TRUST" in f.source_trust.upper():
            reasons.append("Source domain fails trust requirements")

        confidence = calculate_confidence(f.source_trust, f.board_relevance, f.strategic_impact)
        if confidence < 70:
            reasons.append(f"Composite filtering execution threshold violation ({confidence}% < 70%)")

        if f.date_signal == "Undated" and "HIGH TRUST" not in f.source_trust.upper():
            reasons.append("Undated unverified repository data trace")

        fqs, fqs_breakdown = calculate_fact_quality_score(f.fact, f.source_trust, f.board_relevance, f.strategic_impact, f.date_signal)
        if fqs < FACT_QUALITY_THRESHOLD:
            reasons.append(f"Filing extraction quality index deficit ({fqs}/100)")

        if reasons:
            rejected.append({"fact": f.fact[:120], "reasons": reasons, "fact_quality_score": fqs})
            continue

        verified.append(ValidatedFact(
            category=f.category, fact=f.fact, source_url=f.source_url, source_trust=f.source_trust.upper(),
            date_signal=f.date_signal, board_relevance=f.board_relevance, strategic_impact=f.strategic_impact,
            confidence=confidence, fact_quality_score=fqs, quality_breakdown=fqs_breakdown,
        ))

    return verified, rejected

def run_signal_detector(company: str, verified_facts: List[ValidatedFact]) -> List[StrategicSignal]:
    if not verified_facts: return []
    fact_text = "\n".join([f"[{f.category}] METRIC: {f.fact}" for f in verified_facts])
    prompt = f"""You are a Strategic Signal Detector processing corporate data.
Return a JSON object:
{{
  "signals": [
    {{
      "signal_type": "Moat Erosion, Regulatory Friction, Margin Trajectory, etc.",
      "signal": "isolated management shift or performance gap matching extracted metrics",
      "urgency": "IMMEDIATE, 90-DAY, 6-MONTH, or WATCH",
      "implication": "downstream capability or capital reallocation impact statement"
    }}
  ]
}}
Verified Disclosures:
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

def score_options_deterministically(options: List[EvaluatedOption]) -> List[EvaluatedOption]:
    scored = []
    for opt in options:
        score = calculate_option_score(
            opt.evidence_support_score, opt.strategic_fit_score, opt.opportunity_score,
            opt.urgency_score, opt.risk_score, opt.complexity_score,
        )
        opt.composite_score = score
        scored.append(opt)
    scored.sort(key=lambda x: x.composite_score, reverse=True)
    return scored

def run_expert_reasoner(
    company: str, entity: EntityProfile, verified_facts: List[ValidatedFact],
    signals: List[StrategicSignal], evidence_sufficient: bool, sufficiency_message: str
) -> Optional[DecisionIntelligenceBrief]:

    fact_text   = "\n".join([f"- [{f.category}] {f.fact} (Quality: {f.fact_quality_score}/100)" for f in verified_facts]) if verified_facts else "DATASET DEFICIT."
    signal_text = "\n".join([f"- [{s.urgency}] {s.signal}" for s in signals]) or "No trends isolated."

    prompt = f"""# SYSTEM INSTRUCTIONS: STRATEGIC REASONING MATRIX

## REQUIRED TRACEABILITY PATH
[Filing Evidence] -> [Observation] -> [Root Cause] -> [Inference] -> [Theme] -> [Options] -> [Decision]

---
## GATE 1 — OBSERVATION PURITY
Observations MUST only restate filing metrics without causal words (because, therefore, etc.) or forward-looking projections.
Exact metric identity MUST be preserved (e.g., if filing cites PAT, use PAT; never switch to general profit labels).

## GATE 2 — ROOT CAUSE MANAGEMENT
Isolate institutional triggers. If evidence does not specify an operational cause, mark UNKNOWN.

## GATE 3 — INFERENCE SPECIFICITY
Explain structural significance using industry language. Append uncertainty parameters (| CONFIRMED, | LIKELY, or | HYPOTHESIS). Must use strategic markers (pressure, moat, execution risk).

## GATE 4 — THEME REQUIREMENT
Strategic Themes require min 2 observations tracking a continuous operational path. Otherwise label as EMERGING SIGNAL.

## GATE 5 — COMPETITIVE INTEGRITY
Isolate competitor claims only if supported by reporting text. Do not generate parameters outside context.

## GATE 6 — STRATEGIC POSTURES
Generate 3 distinct trajectories:
- Conservative = Protect existing base lines, maintain margin profile
- Balanced     = Optimize cost structure, improve asset turns
- Aggressive   = Deploy capital into new advantage channels, structural change

## GATE 7 — DISCLOSURE SPECIFICITY TEST
Recommending action paths missing explicit anchors (percentages, exact fiscal quarters, product lines) triggers system rejection.

## GATE 8 — DECISION LAYERING FORMAT
Format: "Based on Obs: [...], Inf: [...], Theme: [...], Opt: [...]: [uniquely anchored execution step]"
---
Filing Dataset Parameters: {'SUFFICIENT' if evidence_sufficient else 'INSUFFICIENT_EVIDENCE'} ({sufficiency_message})

Verified Filing Records:
{fact_text}

Strategic Trends:
{signal_text}

Entity: {entity.canonical_name} | {entity.industry} | {entity.primary_market}

OUTPUT STRICT JSON MATCHING THE REGISTERED SCHEMATICS PROFILED IN THE CONFIGURATIONS.
"""
    try:
        data = invoke_json(prompt)
        brief = DecisionIntelligenceBrief(**data)
        brief.evaluated_options = score_options_deterministically(brief.evaluated_options)
        return brief
    except Exception as e:
        st.error(f"Execution error on reasoning node: {e}")
        return None

# ==========================================
# 8. STREAMLIT USER INTERFACE
# ==========================================
company = st.text_input("Target Corporate Entity:", placeholder="e.g. Zomato, Reliance, Tata Motors...")

if st.button("Run Evidence-Based Ingestion Pipeline", type="primary"):
    if not company:
        st.error("Target parameter required.")
    else:
        with st.status(f"Executing Filing Analytics Pipeline for {company}...", expanded=True) as status:

            st.write("📡 Ingesting Annual Reports & Concall Transcripts...")
            raw_context = run_enhanced_search(company)
            if not raw_context:
                st.error("Search engine surface error. No filing data extracted.")
                st.stop()
            time.sleep(1)

            st.write("🔍 Resolving legal corporate entity entity...")
            entity = run_entity_resolution(company, raw_context)
            entity_conf, entity_conf_msg = calculate_entity_confidence(entity)
            if entity_conf < ENTITY_CONFIDENCE_THRESHOLD:
                st.warning("⚠️ Filing data resolution variance detected.")
            time.sleep(1)

            st.write("📊 Isolating disclosure items and audit parameters...")
            raw_facts = run_researcher(company, entity, raw_context[:15000])

            st.write("🔒 Processing institutional data quality checks...")
            verified_facts, rejected_facts = run_hard_gate_validation(raw_facts, entity.canonical_name)

            st.write("🔁 Executing semantic data deduplication engine...")
            verified_facts, dup_log = deduplicate_facts(verified_facts)

            report_confidence_prelim = calculate_report_confidence(verified_facts, len(raw_facts))
            evidence_sufficient, sufficiency_message = get_evidence_sufficiency(verified_facts, report_confidence_prelim)
            if not evidence_sufficient:
                st.warning(f"⚠️ Dataset alert: {sufficiency_message}")

            st.write("🔭 Mapping trajectory signals...")
            signals = run_signal_detector(company, verified_facts)
            time.sleep(1)

            st.write("⚖️ Initiating traceability matrix mapping...")
            final_brief = run_expert_reasoner(
                company, entity, verified_facts, signals,
                evidence_sufficient, sufficiency_message
            )

            status.update(label="Regulatory Analysis Compiled", state="complete")

        if not final_brief:
            st.error("JSON formatting error during inference execution cycle.")
            st.stop()

        # ==========================================
        # DISPLAY LAYER (CLEANED OF CONFIDENCE MARKS)
        # ==========================================
        st.divider()
        st.header(f"Corporate Intelligence Brief — {entity.canonical_name.upper()}")
        st.caption(f"**Sector Classification:** {entity.industry} | {entity.sector} | {entity.primary_market}")

        if final_brief.status == "INSUFFICIENT_EVIDENCE":
            st.error("🛑 DISCLOSURE GATE REJECTED")
            st.warning(f"**Fault parameter:** {final_brief.reason or 'Insufficient corporate filing density.'}")
            st.stop()
        else:
            st.success("✅ DISCLOSURE SUFFICIENCY PROFILE VERIFIED")

        # 1. FACT QUALITY REPORT (CONFIDENCE METRIC REMOVED)
        with st.expander(f"📊 Filing Validation Ledger — {len(verified_facts)} passed / {len(rejected_facts)} rejected", expanded=False):
            col_pass, col_fail = st.columns(2)

            with col_pass:
                st.markdown("**✅ Audited & Transcribed Evidence**")
                for vf in verified_facts:
                    with st.container(border=True):
                        st.markdown(f"**[{vf.category}]** {vf.fact[:140]}...")
                        m1, m2 = st.columns(2)
                        m1.metric("Filing Specificity Score", f"{vf.fact_quality_score}/100")
                        m2.markdown(f"**Repository Trust:** `{vf.source_trust.replace(' TRUST', '')}`")

            with col_fail:
                st.markdown("**❌ Rejected Input Noise**")
                if not rejected_facts:
                    st.info("No input data discarded.")
                for rf in rejected_facts:
                    with st.container(border=True):
                        st.markdown(f"`{rf['fact']}`")
                        for r in rf["reasons"]:
                            st.error(f"• {r}")

        # 2. TRACEABILITY CHAIN LEDGER
        violations = validate_traceability_chain(final_brief, verified_facts)
        if violations:
            with st.expander(f"⚠️ Structural Log Mismatch Flaggings ({len(violations)})", expanded=True):
                for v in violations:
                    st.warning(f"• {v}")

        st.markdown("### 1. Ingestion → Observation → Inference Chain")
        for i, log in enumerate(final_brief.evidence_and_observation_log):
            with st.container(border=True):
                st.markdown(f"**Filing/Transcript Ground-Truth:** `{log.evidence or 'N/A'}`")

                has_reasoning, reasoning_msg = contains_reasoning(log.observation or "")
                if has_reasoning:
                    st.error(f"❌ **Observation Statement (Contaminated):** {log.observation or 'N/A'}")
                else:
                    st.info(f"✅ **Isolated Metric Observation:** {log.observation or 'N/A'}")

                metric_violated, metric_msg = check_metric_preservation(log.evidence or "", log.observation or "")
                if metric_violated:
                    st.error(f"❌ {metric_msg}")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Management Action Cause:** `{log.root_cause or 'UNKNOWN'}`")
                with c2:
                    inf = log.inference or ""
                    classification = inf.split("|")[-1].strip().upper() if "|" in inf else "UNCLASSIFIED"
                    badge_color = "green" if classification == "CONFIRMED" else "orange" if classification == "LIKELY" else "red" if classification == "HYPOTHESIS" else "gray"
                    st.markdown(f"**Structural Inference:** {inf}")

        # 3. STRATEGIC MANAGEMENT THEMES
        st.markdown("### 2. Strategic Narrative Patterns & Guidance")
        c1, c2 = st.columns(2)
        for i, ts in enumerate(final_brief.strategic_themes_and_signals):
            col = c1 if i % 2 == 0 else c2
            with col.container(border=True):
                st.subheader(ts.name or "Unmapped Pattern")
                type_val   = ts.type or "UNKNOWN"
                trace_count = len(ts.traceability)

                if "THEME" in type_val.upper() and trace_count < 2:
                    type_val = "EMERGING TRACK"
                    type_color = "orange"
                elif "THEME" in type_val.upper():
                    type_color = "green"
                else:
                    type_color = "blue"

                st.markdown(f"**Structural Trajectory Classification:** :{type_color}[{type_val}]")
                st.markdown("**Trace Links:**")
                for trace in ts.traceability:
                    st.markdown(f"- {trace}")

        # 4. COMPETITIVE PROFILE
        st.markdown("### 3. Transcript-Backed Competitive Positions")
        for comp in final_brief.competitive_landscape:
            with st.container(border=True):
                st.markdown(f"**Competitor Node:** {comp.competitor or 'N/A'}")
                c_adv, c_vuln = st.columns(2)
                with c_adv:
                    st.success(f"**Advantage Claim:** {comp.advantage or 'No explicit documentation found.'}")
                    if comp.advantage_evidence:
                        st.caption(f"**Source Context Reference:** {comp.advantage_evidence}")
                with c_vuln:
                    st.error(f"**Vulnerability Matrix Position:** {comp.vulnerability or 'No explicit documentation found.'}")
                    if comp.vulnerability_evidence:
                        st.caption(f"**Source Context Reference:** {comp.vulnerability_evidence}")

        # 5. STRATEGIC EXECUTION OPTIONS
        st.markdown("### 4. Strategy Framework Assessment (Scored Frameworks)")
        for rank, opt in enumerate(final_brief.evaluated_options):
            opt_type   = opt.option_type or "Unknown"
            color      = "blue" if "Conservative" in opt_type else "orange" if "Balanced" in opt_type else "red"
            is_selected = opt_type == final_brief.selected_option_type
            rank_label  = "🏆 ALLOCATED POSITION PATH" if is_selected else f"Rank #{rank+1}"

            with st.container(border=True):
                header_col, score_col = st.columns([3, 1])
                with header_col:
                    st.markdown(f"**{rank_label} — :{color}[{opt_type}]**")
                    if opt.option_strategy:
                        st.caption(f"🎯 Posture Blueprint: _{opt.option_strategy}_")
                    st.markdown(f"{opt.description or 'N/A'}")
                with score_col:
                    st.metric("Composite Processing Score", f"{opt.composite_score}/100")

                sc1, sc2, sc3 = st.columns(3)
                sc4, sc5, sc6 = st.columns(3)
                sc1.metric("Disclosed Evidence Alignment", f"{opt.evidence_support_score}/10")
                sc2.metric("Strategic Context Fit", f"{opt.strategic_fit_score}/10")
                sc3.metric("Opportunity Capture index", f"{opt.opportunity_score}/10")
                sc4.metric("Execution Urgency Window", f"{opt.urgency_score}/10")
                sc5.metric("Structural Friction Risk", f"{opt.risk_score}/10")
                sc6.metric("Operational Complexity Load", f"{opt.complexity_score}/10")

                st.info(f"**Path Anchor Trace:** {opt.traceability_chain or 'N/A'}")

        # 6. EXECUTIVE OUTCOME BLOCK
        st.markdown("### 5. Final Recommended Execution Path")
        with st.container(border=True):
            st.subheader("Target Allocation Framework")
            decision = final_brief.recommended_decision or ""

            if decision:
                generic_count, generic_found = count_generic_phrases(decision)
                if generic_count >= GENERIC_WORD_THRESHOLD:
                    st.error("❌ Action path formulation relies on non-specific management jargon.")
                else:
                    st.success(decision)

                if final_brief.selection_rationale:
                    st.info(f"**Allocation logic documentation:** {final_brief.selection_rationale}")
            else:
                st.write("System failed to generate an explicit execution path.")

            st.divider()
            st.markdown("**Filing Contradiction Log:**")
            contradicting = final_brief.contradicting_evidence or "No reporting contradictions traced."
            st.info(contradicting)

        # Export (Keeps confidence indicators in the raw data packet for audit purposes)
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
            "⬇️ Export Full Decision Package Data Ledger (JSON)",
            data=json.dumps(export, indent=2, ensure_ascii=False),
            file_name=f"{company.replace(' ', '_')}_filing_ledger.json",
            mime="application/json"
        )