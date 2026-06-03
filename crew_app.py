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
# 1. SETUP & COGNITIVE MODEL ROUTING
# ==========================================
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY") or os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_KEY", "")

# Dual-Model Routing Strategy
llm_8b = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.1-8b-instant", temperature=0.1)
llm_70b = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.3-70b-versatile", temperature=0.1)

st.set_page_config(page_title="Strategic Intelligence Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ Strategic Intelligence Engine")
st.markdown("**Evidence-Based Decision Support System** · Bulletproof Parsing Architecture · Native JSON Engine Mode")
st.divider()

# ==========================================
# 2. TRUST & SCORING CONFIGURATIONS
# ==========================================

PRIMARY_SOURCE_DOMAINS = [
    "bseindia.com", "nseindia.com", "sebi.gov.in",
    "ir.", "investor.", "investors.",
    "tickertape.in", "screener.in", "trendlyne.com",
    "stockanalysis.com", "simplywall.st",
    "sec.gov", "edgaronline.com", "mca.gov.in",
    "annualreports.com", "iexchange.in",
]

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

TRUST_SCORE_MAP = {
    "PRIMARY SOURCE": 15,
    "HIGH TRUST":     10,
    "MEDIUM TRUST":    6,
    "LOW TRUST":       2,
}

# Tunable quality gates
MIN_VERIFIED_FACTS            = 2
MIN_REPORT_CONFIDENCE         = 50
ENTITY_CONFIDENCE_THRESHOLD   = 60
FACT_QUALITY_THRESHOLD        = 55
OPTION_SCORE_THRESHOLD        = 30
GENERIC_WORD_THRESHOLD        = 2

# Anti-Template / Generic Language Guards
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
    "quarter", "annual", "fiscal", "q1", "q2", "q3", "q4",
]

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

# ==========================================
# 3. DETERMINISTIC ENGINE UTILITIES
# ==========================================

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
            "Observations must preserve the exact metric identity from evidence."
        )
    return False, ""

INDUSTRY_TREND_MARKERS = [
    r'\b(the industry|the sector|the market|industry as a whole|sector wide|across the industry)\b',
    r'\b(globally|worldwide|industry players|market participants|analysts expect|experts predict)\b',
    r'\b(the overall market|broader market|industry average|sector average|peer group)\b',
    r'\b(it is expected|it is projected|forecasters|research firms predict)\b',
]

def check_company_relevance(fact_text: str, canonical_name: str, competitors_str: str = "") -> tuple[bool, str]:
    text_lower = fact_text.lower()
    name_lower = canonical_name.lower()
    name_core = name_lower
    for suffix in [" limited", " ltd", " inc", " corp", " group", " pvt", " plc"]:
        name_core = name_core.replace(suffix, "")
    name_core = name_core.strip()

    company_mentioned = name_core in text_lower or name_lower in text_lower
    
    competitor_mentioned = False
    if competitors_str and competitors_str.lower() != "unknown":
        for comp in competitors_str.split(","):
            comp_clean = comp.strip().lower()
            for suffix in [" limited", " ltd", " inc", " corp", " group", " pvt", " plc"]:
                comp_clean = comp_clean.replace(suffix, "")
            comp_clean = comp_clean.strip()
            if comp_clean and comp_clean in text_lower:
                competitor_mentioned = True
                break

    has_trend_language = any(re.search(pattern, text_lower) for pattern in INDUSTRY_TREND_MARKERS)

    if has_trend_language and not (company_mentioned or competitor_mentioned):
        return True, (
            f"Fact describes an industry trend without explicitly linking to "
            f"'{canonical_name}' or its rivals ({competitors_str}). Only relevant operational data is admitted."
        )
    return False, ""

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
    return len(wa & wb) / len(wa | wb)

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
                "reason": "Semantic overlap > 55% — same operational fact expressed differently."
            })
        else:
            kept.append(candidate)
    return kept, dup_log

def is_non_decision_content(fact_text: str) -> tuple[bool, str]:
    text_lower = fact_text.lower()
    for pattern in REJECT_CONTENT_PATTERNS:
        if re.search(pattern, text_lower):
            return True, f"Non-decision marketing pattern detected: '{pattern}'"
    has_preferred = any(kw in text_lower for kw in PREFERRED_CONTENT_KEYWORDS)
    if not has_preferred and len(fact_text.split()) < 12:
        return True, "No decision-relevant metrics or regulatory anchors detected."
    return False, ""

def count_generic_phrases(text: str) -> tuple[int, list]:
    text_lower = text.lower()
    found = [p for p in GENERIC_PHRASES if p in text_lower]
    return len(found), found

def contains_reasoning(observation: str) -> tuple[bool, str]:
    obs_lower = observation.lower()
    found = [w for w in REASONING_WORDS_IN_OBSERVATIONS if w in obs_lower]
    if found:
        return True, f"Contains restricted explanatory language: {', '.join(found)}"
    return False, ""

def inference_merely_rephrases(observation: str, inference: str) -> tuple[bool, str]:
    if not observation or not inference:
        return False, ""
    inf_clean = inference.split("|")[0].strip().lower()
    obs_clean = observation.lower()
    obs_words = set(obs_clean.split())
    inf_words = set(inf_clean.split())
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "has", "have", "had", "in", "of", "to", "for"}
    obs_content = obs_words - stopwords
    
    if not obs_content:
        return False, ""
    overlap = len(obs_content & (inf_words - stopwords)) / len(obs_content)

    significance_words = [
        "signal", "suggest", "pattern", "pressure", "advantage", "risk",
        "opportunity", "challenge", "momentum", "strength", "weakness",
        "competitive", "strategic", "structural", "cyclical", "systemic",
        "demand", "supply", "portfolio", "capacity", "execution", "erosion"
    ]
    adds_significance = any(sw in inf_clean for sw in significance_words)

    if overlap > 0.70 and not adds_significance:
        return True, f"Inference mirrors observation words too closely ({int(overlap*100)}%) without strategic evaluation."
    return False, ""

def check_layer_differentiation(observation: str, inference: str, theme_name: str = "") -> list[str]:
    issues = []
    if not observation or not inference:
        return issues
    rephrased, msg = inference_merely_rephrases(observation, inference)
    if rephrased:
        issues.append(f"Layer violation — Inference is a rephrasing of Observation: {msg}")
    if theme_name:
        theme_lower = theme_name.lower()
        inf_words = set(inference.split("|")[0].lower().split()) - {"the", "a", "an", "is", "of", "and"}
        theme_words = set(theme_lower.split()) - {"the", "a", "an", "is", "of", "and"}
        if theme_words and inf_words and len(theme_words & inf_words) / max(len(theme_words), 1) > 0.65:
            issues.append(f"Layer violation — Theme '{theme_name}' duplicates the specific inference layer.")
    return issues

def evaluate_trust(url: str, company: str = "") -> str:
    url_lower = url.lower()
    domain = urlparse(url).netloc.lower().replace("www.", "")
    for pattern in PRIMARY_SOURCE_URL_PATTERNS:
        if pattern in url_lower:
            return "PRIMARY SOURCE"
    for ps_domain in PRIMARY_SOURCE_DOMAINS:
        if ps_domain in domain:
            return "PRIMARY SOURCE"
    if company:
        clean_company = company.lower()
        for stopword in ["the", "group", "inc", "ltd", "llc", "corp", "co", "pvt", "plc"]:
            clean_company = clean_company.replace(stopword, "")
        clean_company = clean_company.strip().replace(" ", "")
        if clean_company and clean_company in domain.replace("-", ""):
            if any(ir in url_lower for ir in ["/investor", "/ir/", "/financials/", "/results/"]):
                return "PRIMARY SOURCE"
            return "HIGH TRUST"
    if any(h in domain for h in HIGH_TRUST_DOMAINS):   return "HIGH TRUST"
    if any(m in domain for m in MEDIUM_TRUST_DOMAINS): return "MEDIUM TRUST"
    if any(l in domain for l in LOW_TRUST_DOMAINS):    return "LOW TRUST"
    return "MEDIUM TRUST"

def calculate_confidence(trust_label: str, board_relevance: int, strategic_impact: int) -> int:
    trust_score = TRUST_SCORE_MAP.get(trust_label.strip().upper(), 5)
    trust_normalised = min(trust_score, 10)
    raw = (trust_normalised * 0.4) + (board_relevance * 0.3) + (strategic_impact * 0.3)
    return int((raw / 10) * 100)

def calculate_fact_quality_score(fact_text: str, source_trust: str, board_relevance: int, strategic_impact: int, date_signal: str) -> tuple[int, dict]:
    breakdown = {}
    has_numbers  = bool(re.search(r'\d', fact_text))
    has_percent  = '%' in fact_text
    has_currency = bool(re.search(r'[$₹€£¥]|\b(crore|lakh|billion|million)\b', fact_text, re.I))
    has_named    = len(fact_text.split()) > 8

    specificity = 0
    if has_numbers:  specificity += 10
    if has_percent:  specificity += 7
    if has_currency: specificity += 5
    if has_named:    specificity += 3
    breakdown["specificity"] = min(specificity, 25)

    trust_upper = source_trust.strip().upper()
    if trust_upper == "PRIMARY SOURCE":
        breakdown["source_trust"] = 30
    else:
        trust_raw = TRUST_SCORE_MAP.get(trust_upper, 5)
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
        score -= 25; reasons.append("resolution failed")
    elif "none" not in contamination and contamination != "":
        score -= 20; reasons.append(f"contamination risk: {entity.contamination_warnings}")
    return max(0, score), f"Entity confidence {score}%" + (f" — Issues: {', '.join(reasons)}" if reasons else "")

def calculate_report_confidence(verified_facts: list, total_facts: int) -> int:
    if not verified_facts or total_facts == 0: return 15
    gate_rate = len(verified_facts) / total_facts
    avg_conf  = sum(f.confidence for f in verified_facts) / len(verified_facts)
    return int((gate_rate * 0.4 + avg_conf / 100 * 0.6) * 100)

def calibrate_confidence_label(verified_facts: list) -> tuple[str, str]:
    n = len(verified_facts)
    high_trust_count = sum(1 for f in verified_facts if "HIGH TRUST" in f.source_trust.upper() or "PRIMARY" in f.source_trust.upper())
    avg_quality = sum(f.fact_quality_score for f in verified_facts) / n if n else 0

    if n >= 4 and high_trust_count >= 1 and avg_quality >= 55:
        return "HIGH", f"{n} verified facts, {high_trust_count} high-trust anchors. Stable database context."
    elif n >= 2:
        return "MEDIUM", f"{n} verified facts. Context satisfies requirements."
    else:
        return "LOW", f"Only {n} factual anchors survived filtering. Metrics require manual validation."

def get_evidence_sufficiency(verified_facts: list, report_confidence: int) -> tuple[bool, str]:
    if len(verified_facts) < MIN_VERIFIED_FACTS:
        return False, f"Only {len(verified_facts)} fact(s) passed validation. Insufficient context base."
    if report_confidence < MIN_REPORT_CONFIDENCE:
        return False, f"Composite dataset confidence ({report_confidence}%) falls below required decision grade."
    return True, "Context base verified as sufficient."

def calculate_option_score(evidence_support: int, strategic_fit: int, opportunity: int, urgency: int, risk: int, complexity: int) -> int:
    raw = (evidence_support * 0.25 + strategic_fit * 0.20 + opportunity * 0.25 + urgency * 0.15 - risk * 0.10 - complexity * 0.05)
    raw_min = 1 * 0.85 - 10 * 0.15
    raw_max = 10 * 0.85 - 1 * 0.15
    return max(0, min(100, int(((raw - raw_min) / (raw_max - raw_min)) * 100)))

UNIVERSAL_STRATEGY_PATTERNS = [
    r'\b(improve (customer|operational|product|service) (experience|quality|efficiency))\b',
    r'\b(expand (market|global|international|geographic) (presence|reach|footprint))\b',
    r'\b(invest in (technology|talent|innovation|digital|infrastructure))\b',
    r'\b(build (brand|awareness|loyalty|recognition))\b',
    r'\b(reduce (costs?|expenses?|overhead))\b',
    r'\b(increase (revenue|sales|market share|profitability))\b',
    r'\b(develop new (products?|services?|offerings?))\b',
]

def check_recommendation_specificity(decision: str) -> tuple[bool, str]:
    decision_lower = decision.lower()
    hits = [p for p in UNIVERSAL_STRATEGY_PATTERNS if re.search(p, decision_lower)]
    if not hits:
        return False, ""
    has_specifics = bool(re.search(r'\d|%|₹|\$|€|crore|billion|million|lakh|q[1-4]|fy\d|\b[A-Z][a-z]{3,}\b', decision))
    if has_specifics:
        return False, ""
    return True, f"Fails specificity check — matches general patterns: {', '.join(hits[:1])}. Require concrete data anchors."

OPTION_STRATEGY_SIGNATURES = {
    "Conservative": {
        "required": ["protect", "defend", "maintain", "retain", "secure", "preserve", "hold", "safeguard", "contain", "hedge"],
        "forbidden": ["new market", "disrupt", "transform", "pioneer", "acquire", "launch marketwide"],
    },
    "Balanced": {
        "required": ["optimis", "optimiz", "improve", "enhance", "efficienc", "strengthen existing", "leverage current", "deepen"],
        "forbidden": ["defend position only", "disrupt industry", "pioneer completely"],
    },
    "Aggressive": {
        "required": ["new segment", "create", "disrupt", "enter new", "acquire", "launch", "pivot", "pioneer"],
        "forbidden": ["protect and hold", "maintain current status", "defend baseline"],
    },
}

def check_option_differentiation(options) -> list[str]:
    issues = []
    if not options or len(options) < 3:
        return issues
    descriptions = {opt.option_type: (opt.description or "").lower() for opt in options}
    op_types = list(descriptions.keys())

    for i in range(len(op_types)):
        for j in range(i + 1, len(op_types)):
            ta, tb = op_types[i], op_types[j]
            overlap = semantic_overlap_score(descriptions[ta], descriptions[tb])
            if overlap > 0.50:
                issues.append(f"Option differentiation failed: '{ta}' and '{tb}' have {int(overlap*100)}% structural overlap. Framework must present mutually exclusive strategies.")

    for opt_type, sigs in OPTION_STRATEGY_SIGNATURES.items():
        desc = descriptions.get(opt_type, "")
        if not desc: continue
        if not any(kw in desc for kw in sigs["required"]):
            issues.append(f"Option '{opt_type}' description doesn't fit required strategic posture ({', '.join(sigs['required'][:3])}).")
    return issues

def validate_traceability_chain(brief, verified_facts: list = None) -> list[str]:
    violations = []
    theme_names = [ts.name or "" for ts in brief.strategic_themes_and_signals]

    for i, log in enumerate(brief.evidence_and_observation_log):
        tag = f"Chain Entry {i+1}"
        if not log.evidence or len(log.evidence.strip()) < 10:
            violations.append(f"{tag}: Source evidence anchor missing.")
        if not log.observation or len(log.observation.strip()) < 10:
            violations.append(f"{tag}: Observational statement missing.")

        has_reasoning, r_msg = contains_reasoning(log.observation or "")
        if has_reasoning:
            violations.append(f"{tag}: Observation violates purity guidelines — {r_msg}")

        m_violated, m_msg = check_metric_preservation(log.evidence or "", log.observation or "")
        if m_violated:
            violations.append(f"{tag}: {m_msg}")

        if log.inference:
            parts = log.inference.split("|")
            classification = parts[-1].strip().upper() if len(parts) > 1 else ""
            if classification not in ["CONFIRMED", "LIKELY", "HYPOTHESIS"]:
                violations.append(f"{tag}: Inference lacks probability classification tag (e.g. '| LIKELY').")
            rephrased, rephrase_msg = inference_merely_rephrases(log.observation or "", log.inference)
            if rephrased:
                violations.append(f"{tag}: {rephrase_msg}")
            for tname in theme_names:
                for issue in check_layer_differentiation(log.observation or "", log.inference, tname):
                    violations.append(f"{tag}: {issue}")
        else:
            violations.append(f"{tag}: Strategic inference layer missing.")

    for theme in brief.strategic_themes_and_signals:
        t_count = len(theme.traceability)
        t_type = (theme.type or "").upper()
        if "THEME" in t_type and t_count < 2:
            violations.append(f"Theme '{theme.name}': Classified as STRATEGIC THEME but backed by only {t_count} link. Downgrade to EMERGING SIGNAL.")
        elif t_count == 0:
            violations.append(f"Theme '{theme.name}': Complete disconnection from upstream observations.")
        
        if theme.name and theme.name.lower().strip() in ["revenue growth", "profitability", "market share", "growth", "expansion", "portfolio-driven revenue resilience"]:
            violations.append(f"Theme '{theme.name}': Permitted customized pattern names only. Universal templates rejected.")

    for comp in brief.competitive_landscape:
        c_name = comp.competitor or "Unknown"
        if comp.advantage and "INSUFFICIENT" in comp.advantage:
            pass 
        elif comp.advantage and not comp.advantage_evidence:
            violations.append(f"Competitor '{c_name}': Stated advantage lacks clear fact-link source tracking.")

    decision = brief.recommended_decision or ""
    if decision:
        for tag in ["Obs:", "Inf:", "Theme:", "Opt:"]:
            if tag.lower() not in decision.lower():
                violations.append(f"Decision: Missing required analytical link identifier '{tag}'. Chain broken.")
        g_count, g_found = count_generic_phrases(decision)
        if g_count >= GENERIC_WORD_THRESHOLD:
            violations.append(f"Decision: Violates corporate-fluff gate with phrases: {', '.join(g_found)}")
        spec_fail, spec_reason = check_recommendation_specificity(decision)
        if spec_fail:
            violations.append(f"Decision: {spec_reason}")
    else:
        violations.append("Decision: Pipeline output field empty.")

    violations.extend(check_option_differentiation(brief.evaluated_options))
    return violations

# ==========================================
# 4. EXECUTOR CORE (DEFENSIVE JSON HANDLING)
# ==========================================

def invoke_json(prompt: str, model_type: str = "8b") -> dict:
    messages = [
        SystemMessage(content=(
            "You are a strict, precise JSON-only responder. "
            "Output ONLY a raw, unformatted valid JSON object. Do NOT wrap output inside markdown block codes or triple backticks. Start directly with '{' and end with '}'."
        )),
        HumanMessage(content=prompt)
    ]
    selected_llm = llm_70b if model_type == "70b" else llm_8b
    
    # Force native JSON validation structure via Groq hardware configuration parameters
    try:
        json_capable_llm = selected_llm.bind(response_format={"type": "json_object"})
        resp = json_capable_llm.invoke(messages)
    except Exception:
        resp = selected_llm.invoke(messages)
        
    text = resp.content.strip()
    
    # Fallback Barrier: Greedy regex extract to intercept conversational text leak strings
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = text
        
    return json.loads(json_str)

def _ddgs_search(queries: list, max_per_query: int = 3) -> list[dict]:
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=max_per_query):
                    results.append(r)
    except Exception as e:
        st.warning(f"Search index connection warning: {e}")
    return results

def run_primary_source_search(company: str) -> str:
    current_year = datetime.now().year
    prev_year    = current_year - 1
    queries = [
        f'"{company}" concall transcript {current_year} earnings call financial metrics',
        f'"{company}" Q4 {prev_year} earnings transcript investor reporting metrics',
        f'"{company}" annual report PDF investor financial performance FY{str(prev_year)[-2:]}',
        f'"{company}" investor presentation strategy roadmap targets {current_year}'
    ]
    results = []
    for r in _ddgs_search(queries, max_per_query=3):
        url = r.get("href", "")
        trust = evaluate_trust(url, company)
        results.append(f"[PRIMARY SOURCE MATCH]\nURL: {url}\nTRUST: {trust}\nDATA: {r.get('title','')} — {r.get('body','')}\n{'='*50}")
    return "\n".join(results)

def run_general_search(company: str) -> str:
    current_year = datetime.now().year
    queries = [
        f"{company} performance revenue profit margin expansion metrics {current_year}",
        f"{company} regulatory updates compliance market position {current_year}"
    ]
    results = []
    for r in _ddgs_search(queries, max_per_query=2):
        url = r.get("href", "")
        trust = evaluate_trust(url, company)
        results.append(f"URL: {url}\nTRUST: {trust}\nDATA: {r.get('title','')} — {r.get('body','')}\n{'-'*40}")
    return "\n".join(results)

def run_competitor_deep_search(company: str, competitors_str: str) -> str:
    current_year = datetime.now().year
    results = []
    if not competitors_str or competitors_str.lower() == "unknown":
        return ""
    
    rivals = [r.strip() for r in competitors_str.split(",")[:3]]
    queries = []
    for rival in rivals:
        queries.extend([
            f'"{company}" vs "{rival}" market share revenue scale benchmarks {current_year}',
            f'"{company}" and "{rival}" comparative operational margins compression'
        ])
        
    for r in _ddgs_search(queries, max_per_query=2):
        url = r.get("href", "")
        trust = evaluate_trust(url, company)
        results.append(f"[COMPETITOR TARGET BENCHMARK]\nURL: {url}\nTRUST: {trust}\nDATA: {r.get('title','')} — {r.get('body','')}\n{'='*50}")
    return "\n".join(results)

def run_enhanced_search(company: str) -> str:
    p_ctx = run_primary_source_search(company)
    g_ctx = run_general_search(company)
    combined = ""
    if p_ctx:
        combined += "===== SECURED STRATEGIC CORPORATE DATA DISCLOSURES =====\n" + p_ctx + "\n\n"
    if g_ctx:
        combined += "===== RELEVANT FINANCIAL PRESS & BENCHMARKS =====\n" + g_ctx
    return combined

# ==========================================
# 5. DATA STRUCTURE SCHEMAS (PYDANTIC)
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
    source_type: str = "General"
    date_signal: str = "Undated"
    board_relevance: int
    strategic_impact: int

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
    observation: Optional[str] = None
    root_cause: Optional[str] = None
    inference: Optional[str] = None
    observation_purity_passed: bool = True
    inference_classified: bool = True

class ThemeSignal(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    traceability: List[str] = Field(default_factory=list)

class CompetitiveLandscape(BaseModel):
    competitor: Optional[str] = None
    advantage: Optional[str] = None
    advantage_evidence: Optional[str] = None
    vulnerability: Optional[str] = None
    vulnerability_evidence: Optional[str] = None

class EvaluatedOption(BaseModel):
    option_type: Optional[str] = None
    option_strategy: Optional[str] = None
    description: Optional[str] = None
    traceability_chain: Union[str, List[str], None] = None
    evidence_support_score: int = 5
    strategic_fit_score: int = 5
    opportunity_score: int = 5
    urgency_score: int = 5
    risk_score: int = 5
    complexity_score: int = 5
    composite_score: int = 0
    generic_test_passed: Optional[str] = None
    rejection_reason: Optional[str] = None

class DecisionIntelligenceBrief(BaseModel):
    status: str
    reason: Optional[str] = None
    evidence_and_observation_log: List[EvidenceLog] = Field(default_factory=list)
    strategic_themes_and_signals: List[ThemeSignal] = Field(default_factory=list)
    competitive_landscape: List[CompetitiveLandscape] = Field(default_factory=list)
    evaluated_options: List[EvaluatedOption] = Field(default_factory=list)
    recommended_decision: Optional[str] = None
    selected_option_type: Optional[str] = None
    selection_rationale: Optional[str] = None
    contradicting_evidence: Optional[str] = None
    confidence_assessment: Optional[str] = None

# ==========================================
# 6. PIPELINE ORCHESTRATION AGENTS
# ==========================================

FACT_CATEGORIES = ["Profitability", "Growth", "Competitive Threat", "Competitive Advantage", "Capital Allocation", "Strategic Shift"]

def run_entity_resolution(company: str, raw_context: str) -> EntityProfile:
    prompt = f"""You are an Entity Resolution Specialist. Parse this text block and pinpoint the specific corporate entity targeted.
Return a valid JSON object matching this schema layout:
{{
  "canonical_name": "Official verified registry corporate name",
  "industry": "Specific industry segment",
  "sector": "Sector category",
  "business_model": "Primary mechanism for cash generation based on text data",
  "primary_market": "Core geographic focus region",
  "known_subsidiaries": "Comma separated list of clear subsidiaries, or Unknown",
  "known_competitors": "List major market operating rivals clearly mentioned or linked in text (e.g., Nike, Puma for Adidas)",
  "contamination_warnings": "None detected OR detail if source material describes a mismatched entity"
}}
Company queried: {company}
Context window snippet: {raw_context[:2000]}"""
    try:
        return EntityProfile(**invoke_json(prompt, model_type="8b"))
    except Exception:
        return EntityProfile(
            canonical_name=company, industry="Unknown", sector="Unknown", business_model="Unknown",
            primary_market="Unknown", known_subsidiaries="Unknown", known_competitors="Unknown",
            contamination_warnings="Entity structure resolution failed. Evaluate fact extraction manually."
        )

def run_researcher(company: str, entity: EntityProfile, raw_context: str) -> List[IntelligenceFact]:
    prompt = f"""You are a High-Precision Extraction Engine running on a advanced LLM framework. Collect granular metrics for {entity.canonical_name}.
Extract 3-5 high-fidelity strategic operational milestones or comparative data records.

CRITICAL DISCIPLINE:
1. Every entry MUST embed exact raw numerical tokens (percentages, currencies, quarter codes, margins, unit shipments).
2. Proactively extract cross-company performance profiles or comparative financial data involving known industry rivals: ({entity.known_competitors}). 

Return JSON format matching this schema structure:
{{
  "facts": [
    {{
      "category": "Must match one of: {', '.join(FACT_CATEGORIES)}",
      "fact": "Granular verifiable data point mapping target parameters or target vs rival performance metrics.",
      "source_url": "Absolute source reference URL",
      "source_trust": "PRIMARY SOURCE / HIGH TRUST / MEDIUM TRUST",
      "date_signal": "Specific tracking period (e.g., Q4 2025). Use 'Undated' if unverified.",
      "board_relevance": 9,
      "strategic_impact": 9
    }}
  ]
}}
Full Available Context Block:
{raw_context}"""
    try:
        data = invoke_json(prompt, model_type="70b")
        facts = []
        for f in data.get("facts", []):
            try:
                if isinstance(f.get("board_relevance"), str):
                    f["board_relevance"] = int(''.join(filter(str.isdigit, f["board_relevance"])) or 9)
                if isinstance(f.get("strategic_impact"), str):
                    f["strategic_impact"] = int(''.join(filter(str.isdigit, f["strategic_impact"])) or 9)
                facts.append(IntelligenceFact(**f))
            except Exception: continue
        return facts
    except Exception: return []

def run_hard_gate_validation(facts: List[IntelligenceFact], canonical_name: str, competitors_str: str) -> tuple[List[ValidatedFact], List[dict]]:
    verified, rejected = [], []
    for f in facts:
        reasons = []
        is_non_decision, nd_reason = is_non_decision_content(f.fact)
        if is_non_decision:
            reasons.append(f"Fails decision grade test: {nd_reason}")
        
        is_irrelevant, rel_reason = check_company_relevance(f.fact, canonical_name, competitors_str)
        if is_irrelevant:
            reasons.append(f"Entity boundary violation: {rel_reason}")
            
        if f.board_relevance < 6 or f.strategic_impact < 6:
            reasons.append(f"Insufficient intensity metrics (Relevance: {f.board_relevance}, Impact: {f.strategic_impact})")
        if "LOW TRUST" in f.source_trust.upper():
            reasons.append("Source channel flagged as LOW TRUST")
        
        confidence = calculate_confidence(f.source_trust, f.board_relevance, f.strategic_impact)
        if confidence < 50:
            reasons.append(f"Calculated data confidence ({confidence}%) fails safety bounds.")

        fqs, fqs_breakdown = calculate_fact_quality_score(f.fact, f.source_trust, f.board_relevance, f.strategic_impact, f.date_signal)
        if fqs < 45:
            reasons.append(f"Composite factual precision score ({fqs}/100) below required threshold.")

        if reasons:
            rejected.append({"fact": f.fact[:120], "reasons": reasons, "fact_quality_score": fqs})
            continue

        verified.append(ValidatedFact(
            category=f.category, fact=f.fact, source_url=f.source_url, source_trust=f.source_trust.upper(),
            date_signal=f.date_signal, board_relevance=f.board_relevance, strategic_impact=f.strategic_impact,
            confidence=confidence, fact_quality_score=fqs, quality_breakdown=fqs_breakdown
        ))
    return verified, rejected

def run_signal_detector(company: str, verified_facts: List[ValidatedFact]) -> List[StrategicSignal]:
    if not verified_facts: return []
    fact_text = "\n".join([f"[{f.category}] FACT: {f.fact}" for f in verified_facts])
    prompt = f"""You are a Strategic Signal Detector operating on Llama 70B framework. Extrapolate underlying structural trends.
Return valid JSON structure:
{{
  "signals": [
    {{
      "signal_type": "Moat Erosion, Competitor Outperformance, Structural Shift, or Regulatory Pressure",
      "signal": "Clear systemic market variance directly anchoring onto verified fact data",
      "urgency": "IMMEDIATE / 90-DAY / 6-MONTH / WATCH",
      "implication": "Specific strategic resource re-allocation triggered by this insight"
    }}
  ]
}}
Verified Dataset Input:
{fact_text}"""
    try:
        data = invoke_json(prompt, model_type="70b")
        signals = []
        for s in data.get("signals", []):
            try: signals.append(StrategicSignal(**s))
            except Exception: continue
        return signals
    except Exception: return []

def score_options_deterministically(options: List[EvaluatedOption]) -> List[EvaluatedOption]:
    scored = []
    for opt in options:
        opt.composite_score = calculate_option_score(
            opt.evidence_support_score, opt.strategic_fit_score, opt.opportunity_score,
            opt.urgency_score, opt.risk_score, opt.complexity_score
        )
        scored.append(opt)
    scored.sort(key=lambda x: x.composite_score, reverse=True)
    return scored

def run_expert_reasoner(
    company: str, entity: EntityProfile, verified_facts: List[ValidatedFact],
    signals: List[StrategicSignal], evidence_sufficient: bool, sufficiency_message: str
) -> Optional[DecisionIntelligenceBrief]:

    fact_text   = "\n".join([f"- [{f.category}] {f.fact} (Trust: {f.source_trust}, FQS: {f.fact_quality_score}/100)" for f in verified_facts]) if verified_facts else "INSUFFICIENT CONTEXT DATA."
    signal_text = "\n".join([f"- [{s.urgency}] {s.signal}" for s in signals]) or "No trends mapped."

    prompt = f"""# SYSTEM INSTRUCTIONS: FRONTIER EVIDENCE-BASED REASONING ENGINE (70B PARADIGM)

## CORE MANDATE
You are a peerless Strategic Reasoning Engine. Your analytical objective is to execute deep diagnostic inference. 
Avoid generic boilerplate summaries or filler text. Every output vector must chain explicitly through the following loop:
[Evidence Link] -> [Pure Observation] -> [Deductive Root Cause] -> [Strategic Inference Layer] -> [Customized Theme] -> [Postured Strategy Matrix] -> [Anchored Decision String]

---

## INTELLECTUAL QUALITY GATES

### GATE 1 — STRUCTURAL OBSERVATION PURITY
Observations MUST strictly state the naked data recorded in the text. They are strictly prohibited from embedding explanatory modifiers or logic links.
- FORBIDDEN TOKENS: because, therefore, suggests, indicates, implies, means that, as a result, due to, caused by, which shows, hence, thus.
- METRIC PRESERVATION: Retain identical metric definitions. If evidence tracking records a variance in EBITDA, the observation cannot swap it for 'operating efficiency'.

### GATE 2 — REASONED CAUSAL DIAGNOSTICS
You must actively analyze broader market trends, corporate restructuring events, and financial pressures present within the text to synthesize and deduce the most likely operational or economic driver ('Root Cause') behind an observation. Never copy the observation text inside the cause field.

### GATE 3 — STRATEGIC INFERENCE DECONSTRUCTION
Inferences must capture the downstream long-term viability impact. 
- All entries must explicitly terminate with a tracking probability flag: | CONFIRMED, | LIKELY, or | HYPOTHESIS.
- Must include at least one diagnostic token: signal, pressure, advantage, risk, opportunity, challenge, momentum, structural, erosion, expansion, exposure, capacity.

### GATE 4 — CUSTOMIZED THEME PATTERNING (STATEGIC UNIQUE TRAJECTORIES)
Do not populate generic templates like "Portfolio-Driven Revenue Resilience". You must frame unique, data-tailored corporate pattern names tied to the precise events (e.g., 'Margin Recovery Post-Yeezy Restructuring', 'Pricing Pressure Overwhelming Volume Penetration Dominance').
- A Strategic Theme requires MINIMUM 2 supporting references. Single link patterns must be classified as EMERGING SIGNAL.
- REJECT BARE LABELS: "Revenue Growth", "Profitability", "Market Share".

### GATE 5 — RELATIONAL COMPETITIVE POSITIONING
Actively map available factual indicators against the specified rivals list ({entity.known_competitors}). If the facts contain specific performance numbers, revenue shifts, or structural advantages/vulnerabilities regarding competitors like Nike or Puma, state them clearly. Deduce competitive advantages or structural positioning deficits based *only* on the text data. If no context allows a logical deduction for a competitor, populate with 'INSUFFICIENT_COMPETITIVE_EVIDENCE'. Do not leave fields blank or null.

### GATE 6 — STRATEGIC SCORE ALLOCATION
Score each option metric strictly as an integer between 1 and 10. Do NOT use string structures or fraction trailing values like "8/10" or "High". 

### GATE 7 — CRITICAL DECISION STRING SYNTAX
The 'recommended_decision' MUST be a single, highly integrated string using this exact structural template:
"Based on Obs: [summary of key pure fact], Inf: [summary of key strategic implication | probability], Theme: [exact tailored theme name], Opt: [Conservative/Balanced/Aggressive]: [highly specific action embedding a metric or geographic target]."
- CRITICAL: Ensure all literal indicator tags ('Obs:', 'Inf:', 'Theme:', 'Opt:') are explicitly spelled out.

---
Data Status: {'SUFFICIENT' if evidence_sufficient else 'INSUFFICIENT_EVIDENCE'} ({sufficiency_message})
Verified Input Context Base:
{fact_text}
Strategic Vectors:
{signal_text}
Target Profile: {entity.canonical_name} | Sector: {entity.sector} | Rival Ring: {entity.known_competitors}

OUTPUT FORMAT — STRICT RAW VALID JSON STRUCTURE ONLY matching the following schema precisely:
{{
  "status": "SUFFICIENT",
  "reason": "Clear confirmation overview statement.",
  "evidence_and_observation_log": [
    {{
      "evidence": "Direct copy or faithful close paraphrase of a verified fact tracking metric",
      "observation": "Naked factual restatement with zero logic tokens or explanatory terms",
      "root_cause": "Synthesized commercial or internal operational driver explaining why the observation happened based on text logic",
      "inference": "Strategic downstream risk or leverage evaluation statement | LIKELY"
    }}
  ],
  "strategic_themes_and_signals": [
    {{
      "name": "Custom data-driven pattern statement matching specific corporate events",
      "type": "STRATEGIC THEME",
      "traceability": ["Observation reference string snippet"]
    }}
  ],
  "competitive_landscape": [
    {{
      "competitor": "Name of rival parsed from profile context",
      "advantage": "Contextually reasoned market edge derived from facts, or INSUFFICIENT_COMPETITIVE_EVIDENCE",
      "advantage_evidence": "Explicit fact link statement, or INSUFFICIENT_COMPETITIVE_EVIDENCE",
      "vulnerability": "Operational deficit or performance delta derived from facts, or INSUFFICIENT_COMPETITIVE_EVIDENCE",
      "vulnerability_evidence": "Explicit fact link statement, or INSUFFICIENT_COMPETITIVE_EVIDENCE"
    }}
  ],
  "evaluated_options": [
    {{
      "option_type": "Conservative",
      "option_strategy": "Protect existing position — [Specific vector to secure]",
      "description": "Mutually exclusive action step anchored onto quantitative targets, completely distinct from balanced or aggressive paths",
      "traceability_chain": "Theme [X] -> Inference [Y] -> Observation [Z]",
      "evidence_support_score": 8,
      "strategic_fit_score": 7,
      "opportunity_score": 4,
      "urgency_score": 5,
      "risk_score": 3,
      "complexity_score": 4,
      "generic_test_passed": "Yes",
      "rejection_reason": null
    }},
    {{
      "option_type": "Balanced",
      "option_strategy": "Optimize existing position — [Specific flow or system to refine]",
      "description": "Action vector targeting performance tuning or incremental operational margin expansion",
      "traceability_chain": "...",
      "evidence_support_score": 7,
      "strategic_fit_score": 8,
      "opportunity_score": 6,
      "urgency_score": 6,
      "risk_score": 4,
      "complexity_score": 5,
      "generic_test_passed": "Yes",
      "rejection_reason": null
    }},
    {{
      "option_type": "Aggressive",
      "option_strategy": "Create new strategic advantage — [Specific platform or market pivot]",
      "description": "Capital heavy development or market entry targeting competitive paradigm expansion",
      "traceability_chain": "...",
      "evidence_support_score": 6,
      "strategic_fit_score": 6,
      "opportunity_score": 9,
      "urgency_score": 8,
      "risk_score": 7,
      "complexity_score": 7,
      "generic_test_passed": "Yes",
      "rejection_reason": null
    }}
  ],
  "recommended_decision": "Based on Obs: [summary of fact data], Inf: [structural meaning | probability label], Theme: [tailored pattern name], Opt: [Selected Type]: [targeted empirical resource allocation execution plan]",
  "selected_option_type": "Conservative/Balanced/Aggressive",
  "selection_rationale": "Comparative trade-off breakdown showing why the mathematical score allocation prioritized this strategic trajectory over other vectors.",
  "contradicting_evidence": "Empirical disclosures conflicting with the vector, or 'None explicitly noted.'",
  "confidence_assessment": "Confidence: HIGH/MEDIUM-HIGH/MEDIUM/LOW — N verified facts, X high-trust sources. Dedicated dataset stability summary sentence."
}}"""
    try:
        data = invoke_json(prompt, model_type="70b")
        
        # Programmatic Coercion Layer: Defensively fix string-to-integer slips before Pydantic parsing
        if not isinstance(data, dict):
            raise ValueError("Returned JSON payload root structure is mismatched.")
            
        if "status" not in data:
            data["status"] = "SUFFICIENT"
            
        if "evidence_and_observation_log" in data and isinstance(data["evidence_and_observation_log"], list):
            for log in data["evidence_and_observation_log"]:
                if not isinstance(log, dict): continue
                log["observation_purity_passed"] = True
                log["inference_classified"] = True
        else:
            data["evidence_and_observation_log"] = []
            
        if "evaluated_options" in data and isinstance(data["evaluated_options"], list):
            for opt in data["evaluated_options"]:
                if not isinstance(opt, dict): continue
                for score_field in ["evidence_support_score", "strategic_fit_score", "opportunity_score", "urgency_score", "risk_score", "complexity_score"]:
                    val = opt.get(score_field, 5)
                    if isinstance(val, str):
                        digits = ''.join(filter(str.isdigit, val.split('/')[0]))
                        opt[score_field] = int(digits) if digits else 5
                    elif isinstance(val, (int, float)):
                        opt[score_field] = int(val)
                    else:
                        opt[score_field] = 5
        else:
            data["evaluated_options"] = []
            
        brief = DecisionIntelligenceBrief(**data)
        brief.evaluated_options = score_options_deterministically(brief.evaluated_options)
        return brief
    except Exception as e:
        st.error(f"Defensive Filter Execution Notice — Programmatic parsing validation: {e}")
        return None

# ==========================================
# 7. USER INTERFACE AND SCREEN GENERATION
# ==========================================
company = st.text_input("Target Company / Entity Profile Name:", placeholder="e.g. Zomato, Reliance Industries, Tesla, Adidas...")

if st.button("Run System Verification Pipeline", type="primary"):
    if not company:
        st.error("Target identification vector required.")
    else:
        with st.status(f"Executing Multi-Agent Strategic Intelligence Pipeline for {company}...", expanded=True) as status:
            st.write("📡 Stage 1: Harvesting target corporate files and raw regulatory records...")
            raw_context = run_enhanced_search(company)
            if not raw_context:
                st.error("Search indices failed to secure raw target context.")
                st.stop()
            time.sleep(0.5)

            st.write("🔍 Stage 2: Resolving legal identity and identifying core rivals...")
            entity = run_entity_resolution(company, raw_context)
            entity_conf, entity_conf_msg = calculate_entity_confidence(entity)
            if entity_conf < ENTITY_CONFIDENCE_THRESHOLD:
                st.warning(f"⚠️ {entity_conf_msg}")
            time.sleep(0.5)

            st.write(f"🎯 Stage 3: Launching deep competitor benchmark search queries for: {entity.known_competitors}...")
            competitor_context = run_competitor_deep_search(entity.canonical_name, entity.known_competitors)
            
            full_data_lake = raw_context + "\n\n===== COMPETITIVE CROSS-COMPANY BENCHMARKS =====\n" + competitor_context

            st.write("📊 Stage 4: Harvesting operational metrics, financial reports, and rival benchmarks...")
            raw_facts = run_researcher(company, entity, full_data_lake[:25000])

            st.write("🔒 Stage 5: Injecting records into validation gate & factual precision scorer...")
            verified_facts, rejected_facts = run_hard_gate_validation(raw_facts, entity.canonical_name, entity.known_competitors)

            st.write("🔁 Stage 6: Executing Jaccard semantic deduplication filter...")
            verified_facts, dup_log = deduplicate_facts(verified_facts)

            report_confidence_prelim = calculate_report_confidence(verified_facts, len(raw_facts))
            evidence_sufficient, sufficiency_message = get_evidence_sufficiency(verified_facts, report_confidence_prelim)
            if not evidence_sufficient:
                st.warning(f"⚠️ Data Sufficiency Warning: {sufficiency_message}")

            st.write("🔭 Stage 7: Extracting market shifts and macro strategic signals...")
            signals = run_signal_detector(company, verified_facts)
            time.sleep(0.5)

            st.write("⚖️ Stage 8: Engaging Llama 70B Strategic Reasoning Engine with traceability compliance...")
            final_brief = run_expert_reasoner(company, entity, verified_facts, signals, evidence_sufficient, sufficiency_message)
            status.update(label="Analytical Pipeline Execution Complete", state="complete")

        if not final_brief:
            st.error("Reasoning Core payload output verification anomaly. Re-engage pipeline structure.")
            st.stop()

        # Display Layer
        st.divider()
        st.header(f"Decision Validation Brief — {entity.canonical_name.upper()}")
        st.caption(f"**Sector Classification:** {entity.sector} | **Industry:** {entity.industry} | **Core Geography:** {entity.primary_market}")

        if final_brief.status == "INSUFFICIENT_EVIDENCE":
            st.error("🛑 PIPELINE ABORTED: DATA SUFFICIENCY GATE INTERCEPTED EXECUTOR")
            st.warning(f"**Reason Flagged:** {final_brief.reason}")
            st.stop()
        else:
            st.success(f"✅ DATA SUFFICIENCY GATE CONFIRMED: {final_brief.reason or 'Dataset metrics satisfy criteria.'}")

        # Fact Quality Explander Display
        with st.expander(f"📊 Factual Precision Summary ({len(verified_facts)} passed / {len(rejected_facts)} filtered / {len(dup_log)} deduplicated)", expanded=False):
            col_pass, col_fail = st.columns(2)
            with col_pass:
                st.markdown("**✅ Admitted High-Fidelity Data Logs**")
                for vf in verified_facts:
                    with st.container(border=True):
                        st.markdown(f"**[{vf.category}]** {vf.fact}")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Precision Score", f"{vf.fact_quality_score}/100")
                        m2.metric("Confidence", f"{vf.confidence}%")
                        m3.metric("Source Tier", vf.source_trust.replace(" TRUST", ""))

            with col_fail:
                st.markdown("**❌ Intercepted / Low-Fidelity Records**")
                if not rejected_facts:
                    st.info("No records rejected by filtering parameters.")
                for rf in rejected_facts:
                    with st.container(border=True):
                        st.markdown(f"`{rf['fact']}`")
                        st.caption(f"FQS Grade: {rf['fact_quality_score']}/100")
                        for r in rf["reasons"]:
                            st.error(f"• {r}")

            if dup_log:
                st.markdown("**🔁 Redundant Overlapping Metrics Dropped**")
                for dl in dup_log:
                    with st.container(border=True):
                        st.warning(f"**Dropped:** `{dl['rejected_fact']}`")
                        st.caption(f"Identified duplicate of: `{dl['duplicate_of']}`")

        # Traceability Explander Display
        violations = validate_traceability_chain(final_brief, verified_facts)
        if violations:
            with st.expander(f"⚠️ Traceability Exceptions Mapped ({len(violations)} anomalies)", expanded=True):
                for v in violations:
                    st.warning(f"• {v}")
        else:
            st.success("✅ Traceability Chain Integrity: No structural decoupling anomalies detected.")

        # Main Traceability Log UI Block
        st.markdown("### 1. Unified Diagnostic Track Log")
        for i, log in enumerate(final_brief.evidence_and_observation_log):
            with st.container(border=True):
                st.markdown(f"**Upstream Source Grounding:** `{log.evidence or 'N/A'}`")
                
                has_reasoning, reasoning_msg = contains_reasoning(log.observation or "")
                if has_reasoning:
                    st.error(f"❌ **Observational Layer (PURITY EXCEPTION):** {log.observation}\n\n_{reasoning_msg}_")
                else:
                    st.info(f"✅ **Observational Layer (Pure Fact):** {log.observation}")

                m_violated, m_msg = check_metric_preservation(log.evidence or "", log.observation or "")
                if m_violated:
                    st.error(f"❌ **Metric Substitution Detected:** {m_msg}")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"🧠 **Deductive Root Cause:** `{log.root_cause or 'UNKNOWN'}`")
                with c2:
                    inf = log.inference or ""
                    classification = inf.split("|")[-1].strip().upper() if "|" in inf else "UNCLASSIFIED"
                    b_color = "green" if classification == "CONFIRMED" else "orange" if classification == "LIKELY" else "red" if classification == "HYPOTHESIS" else "gray"
                    st.markdown(f"🎯 **Strategic Inference:** {inf}")
                    st.markdown(f"Probability Alignment: :{b_color}[{classification}]")

        # Themes Display
        st.markdown("### 2. Tailored Strategic Matrix Mappings")
        c1, c2 = st.columns(2)
        for idx, ts in enumerate(final_brief.strategic_themes_and_signals):
            target_col = c1 if idx % 2 == 0 else c2
            with target_col.container(border=True):
                st.subheader(ts.name or "Context Pattern Signal")
                type_val = ts.type or "UNKNOWN"
                t_count = len(ts.traceability)

                if "THEME" in type_val.upper() and t_count < 2:
                    st.error("⚠️ Downgrade Enforced: Insufficient internal link density for theme status.")
                    type_val = "EMERGING SIGNAL (Enforced)"
                    type_color = "red"
                else:
                    type_color = "green" if "THEME" in type_val.upper() else "orange"

                st.markdown(f"**Taxonomy Category:** :{type_color}[{type_val}]")
                st.markdown("**Upstream Mapped Tracking Anchors:**")
                for trace in ts.traceability:
                    st.markdown(f"- {trace}")

        # Competitive Landscape Display
        st.markdown("### 3. Structural Competitor Intelligence Matrix")
        if not final_brief.competitive_landscape:
            st.info("No comparative operating metrics processed.")
        for comp in final_brief.competitive_landscape:
            with st.container(border=True):
                st.markdown(f"**Rival Operator:** **{comp.competitor or 'N/A'}**")
                c_adv, c_vuln = st.columns(2)
                with c_adv:
                    st.success(f"📈 **Reasoned Advantage:** {comp.advantage or 'No relational edge isolated.'}")
                    if comp.advantage_evidence:
                        st.caption(f"**Grounding Track:** {comp.advantage_evidence}")
                with c_vuln:
                    st.error(f"📉 **Reasoned Vulnerability:** {comp.vulnerability or 'No specific operating risk isolated.'}")
                    if comp.vulnerability_evidence:
                        st.caption(f"**Grounding Track:** {comp.vulnerability_evidence}")

        # Options Matrix Display
        st.markdown("### 4. Ranked Strategic Response Paths")
        for rank, opt in enumerate(final_brief.evaluated_options):
            opt_type = opt.option_type or "Unknown"
            color = "blue" if "Conservative" in opt_type else "orange" if "Balanced" in opt_type else "red"
            is_selected = opt_type == final_brief.selected_option_type
            rank_header = f"🏆 TOP SCORING POSITION — {opt_type.upper()}" if is_selected else f"Strategic Option Position #{rank+1} — {opt_type.upper()}"

            with st.container(border=True):
                h_col, s_col = st.columns([3, 1])
                with h_col:
                    st.markdown(f"### :{color}[{rank_header}]")
                    st.caption(f"**Postured Vector:** {opt.option_strategy}")
                    st.markdown(f"**Action Architecture:** {opt.description}")
                with s_col:
                    st.metric("Deterministic Composite Score", f"{opt.composite_score}/100")

                sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
                sc1.metric("Evidence Support", f"{opt.evidence_support_score}/10")
                sc2.metric("Strategic Fit", f"{opt.strategic_fit_score}/10")
                sc3.metric("Opportunity Delta", f"{opt.opportunity_score}/10")
                sc4.metric("Urgency Vector", f"{opt.urgency_score}/10")
                sc5.metric("Risk Cost", f"{opt.risk_score}/10")
                sc6.metric("Complexity Drag", f"{opt.complexity_score}/10")

                st.info(f"**Traceability Resolution String:** {opt.traceability_chain or 'N/A'}")
                if opt.generic_test_passed == "Yes":
                    st.success("✅ Passed Context Specificity Filtering Gate.")
                else:
                    st.error(f"❌ Failed Specificity Filtering Gate: {opt.rejection_reason}")

        # Final Summary Block
        st.markdown("### 5. Executed Authorization & Integrity Validation")
        with st.container(border=True):
            st.subheader("Final System Recommended Decision String")
            decision = final_brief.recommended_decision or ""

            g_count, g_found = count_generic_phrases(decision)
            if g_count >= GENERIC_WORD_THRESHOLD:
                st.error(f"❌ Integrated string contains restricted generic fluff: {', '.join(g_found)}")
            else:
                st.success(decision)

            if final_brief.selection_rationale:
                st.info(f"**Trade-Off Rationale Analysis:** {final_brief.selection_rationale}")

            st.divider()
            c_tag1, c_tag2, c_tag3, c_tag4 = st.columns(4)
            c_tag1.markdown(f"{'✅' if 'obs:' in decision.lower() else '❌'} `Obs:` Tracking Token")
            c_tag2.markdown(f"{'✅' if 'inf:' in decision.lower() else '❌'} `Inf:` Tracking Token")
            c_tag3.markdown(f"{'✅' if 'theme:' in decision.lower() else '❌'} `Theme:` Tracking Token")
            c_tag4.markdown(f"{'✅' if 'opt:' in decision.lower() else '❌'} `Opt:` Tracking Token")

            st.markdown("**Identified Contradicting Disclosures / Conflicting Evidence:**")
            st.warning(final_brief.contradicting_evidence or "No corporate discrepancies logged.")

            st.markdown("**LLM Engine Self-Assessment:**")
            st.markdown(f"`{final_brief.confidence_assessment or 'N/A'}`")

            c_label, c_exp = calibrate_confidence_label(verified_facts)
            lbl_color = {"HIGH": "green", "MEDIUM-HIGH": "blue", "MEDIUM": "orange", "LOW": "red"}.get(c_label, "gray")
            st.markdown(f"**Calibrated System Core Data Confidence:** :{lbl_color}[{c_label}]")
            st.caption(f"_{c_exp}_")

        # Package Compilation and Download Link
        st.divider()
        export_package = {
            "entity_profile": entity.model_dump(),
            "fact_precision_audit": {
                "verified_facts": [vf.model_dump() for vf in verified_facts],
                "rejected_facts": rejected_facts,
                "deduplicated_logs": dup_log,
                "dataset_confidence": report_confidence_prelim,
            },
            "reasoning_brief_package": final_brief.model_dump(),
            "pipeline_violations_log": violations,
        }
        st.download_button(
            "⬇️ Download Certified Decision Briefing Package (JSON)",
            data=json.dumps(export_package, indent=2, ensure_ascii=False),
            file_name=f"decision_brief_{company.replace(' ', '_').lower()}.json",
            mime="application/json"
        )