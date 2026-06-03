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
# 3. DETERMINISTIC SCORING
# ==========================================
def calculate_confidence(trust_label: str, board_relevance: int, strategic_impact: int) -> int:
    trust_score = TRUST_SCORE_MAP.get(trust_label.strip().upper(), 5)
    raw = (trust_score * 0.4) + (board_relevance * 0.3) + (strategic_impact * 0.3)
    return int((raw / 10) * 100)


# NEW: Requirement 1 — fact quality scoring before reasoning
def calculate_fact_quality_score(fact_text: str, source_trust: str,
                                  board_relevance: int, strategic_impact: int,
                                  date_signal: str) -> tuple[int, dict]:
    """
    Composite fact quality score (0-100) from 5 dimensions:
    - Specificity (has numbers/percentages/named entities): 0-25
    - Source trust: 0-25
    - Board relevance: 0-20
    - Strategic impact: 0-20
    - Recency (has date signal): 0-10
    """
    breakdown = {}

    # Specificity — does the fact contain hard data?
    has_numbers = bool(re.search(r'\d', fact_text))
    has_percent = '%' in fact_text
    has_currency = bool(re.search(r'[$₹€£¥]|\b(crore|lakh|billion|million|trillion)\b', fact_text, re.I))
    has_named    = len(fact_text.split()) > 8  # proxy for specificity

    specificity = 0
    if has_numbers:  specificity += 10
    if has_percent:  specificity += 7
    if has_currency: specificity += 5
    if has_named:    specificity += 3
    breakdown["specificity"] = min(specificity, 25)

    # Source trust
    trust_raw = TRUST_SCORE_MAP.get(source_trust.strip().upper(), 5)
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


def get_evidence_sufficiency(verified_facts: list, report_confidence: int) -> tuple[bool, str]:
    if len(verified_facts) < MIN_VERIFIED_FACTS:
        return False, f"Only {len(verified_facts)} fact(s) passed validation. Insufficient evidence."
    if report_confidence < MIN_REPORT_CONFIDENCE:
        return False, f"Report confidence {report_confidence}% is below threshold. Evidence quality is low."
    return True, "Evidence sufficient for reliable conclusions."


# NEW: Requirement 8 — numeric option scoring before decision selection
def calculate_option_score(evidence_support: int, risk: int,
                            complexity: int, strategic_fit: int) -> int:
    """
    Composite option score (0-100).
    risk is inverted (lower risk = higher score).
    complexity is inverted (lower complexity = higher score).
    """
    risk_inverted       = 10 - risk
    complexity_inverted = 10 - complexity
    raw = (
        evidence_support * 0.35 +
        risk_inverted    * 0.25 +
        complexity_inverted * 0.15 +
        strategic_fit    * 0.25
    )
    return int((raw / 10) * 100)


# NEW: Requirement 9 — traceability chain validator
def validate_traceability_chain(brief) -> list[str]:
    """
    Returns a list of traceability violations found in the brief.
    Empty list = chain is valid.
    """
    violations = []

    # Check evidence → observation links exist
    for i, log in enumerate(brief.evidence_and_observation_log):
        if not log.evidence or len(log.evidence.strip()) < 10:
            violations.append(f"Log {i+1}: Evidence is missing or too vague.")
        if not log.observation or len(log.observation.strip()) < 10:
            violations.append(f"Log {i+1}: Observation is missing.")
        has_reasoning, reasoning_msg = contains_reasoning(log.observation or "")
        if has_reasoning:
            violations.append(f"Log {i+1}: Observation contains reasoning language — {reasoning_msg}")
        if log.inference:
            classification = log.inference.split("|")[-1].strip().upper()
            if classification not in ["CONFIRMED", "LIKELY", "HYPOTHESIS"]:
                violations.append(f"Log {i+1}: Inference missing classification (must end with | CONFIRMED/LIKELY/HYPOTHESIS).")

    # Check themes have ≥ 2 traceability references
    for i, theme in enumerate(brief.strategic_themes_and_signals):
        if len(theme.traceability) < 2:
            violations.append(f"Theme '{theme.name}': Fewer than 2 traceability references (has {len(theme.traceability)}).")

    # Check competitor claims have evidence
    for comp in brief.competitive_landscape:
        if comp.advantage and not comp.advantage_evidence:
            violations.append(f"Competitor '{comp.competitor}': Advantage claim has no evidence backing.")
        if comp.vulnerability and not comp.vulnerability_evidence:
            violations.append(f"Competitor '{comp.competitor}': Vulnerability claim has no evidence backing.")

    # Check recommended decision references required elements
    decision = brief.recommended_decision or ""
    for required in ["Observation", "Inference", "Theme", "Option"]:
        if required.lower() not in decision.lower():
            violations.append(f"Decision: Does not explicitly reference a {required}.")

    # Check generic language in decision
    generic_count, generic_found = count_generic_phrases(decision)
    if generic_count >= GENERIC_WORD_THRESHOLD:
        violations.append(f"Decision: Contains {generic_count} generic phrase(s): {', '.join(generic_found)}. Recommendation may be too generic.")

    return violations


# ==========================================
# 4. SEARCH
# ==========================================
def run_enhanced_search(company: str) -> str:
    current_year = datetime.now().year
    queries = [
        f"{company} corporate profile industry sector business model footprint",
        f"{company} revenue profit margin earnings performance {current_year}",
        f"{company} market share competitor position metrics {current_year}",
        f"{company} capital allocation infrastructure investments {current_year}",
        f"{company} regulatory challenges compliance sustainability goals {current_year}",
        f"{company} strategic execution transformation roadmap {current_year}",
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
        st.error(f"Search error: {e}")
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


# NEW: Requirement 1 — fact quality score attached to validated facts
class ValidatedFact(BaseModel):
    category: str
    fact: str
    source_url: str
    source_trust: str
    date_signal: str
    board_relevance: int
    strategic_impact: int
    confidence: int
    fact_quality_score: int = 0            # NEW
    quality_breakdown: dict = Field(default_factory=dict)  # NEW


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
    description: Optional[str] = None
    traceability_chain: Union[str, List[str], None] = Field(
        default=None,
        description="Theme [X] -> Inference [Y] -> Observation [Z] -> Evidence [W]"
    )
    # Numeric scores 1-10
    evidence_support_score: int = Field(default=5, description="1-10. How strongly does evidence support this option?")
    risk_score: int            = Field(default=5, description="1-10. How high is the risk? Higher = riskier.")
    complexity_score: int      = Field(default=5, description="1-10. How complex to execute? Higher = more complex.")
    strategic_fit_score: int   = Field(default=5, description="1-10. How well does it fit the strategic context?")
    composite_score: int       = Field(default=0, description="Computed: weighted composite of the 4 scores.")
    generic_test_passed: Optional[str] = Field(
        default=None,
        description="Yes — uniquely applies to this situation with specific evidence. No — could apply to any company."
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
def run_hard_gate_validation(facts: List[IntelligenceFact]) -> tuple[List[ValidatedFact], List[dict]]:
    """
    Returns (verified_facts, rejection_log).
    rejection_log contains every rejected fact with its rejection reason.
    """
    verified  = []
    rejected  = []

    for f in facts:
        reasons = []

        # Gate 1: board relevance + strategic impact
        if f.board_relevance < 8 or f.strategic_impact < 8:
            reasons.append(f"Low scores — board_relevance={f.board_relevance}, strategic_impact={f.strategic_impact} (min 8 each)")

        # Gate 2: reject low trust sources
        if "LOW TRUST" in f.source_trust.upper():
            reasons.append("Source is LOW TRUST")

        # Gate 3: confidence threshold
        confidence = calculate_confidence(f.source_trust, f.board_relevance, f.strategic_impact)
        if confidence < 70:
            reasons.append(f"Confidence {confidence}% < 70% threshold")

        # Gate 4: undated fact from non-high-trust source
        if f.date_signal == "Undated" and "HIGH TRUST" not in f.source_trust.upper():
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
    """Compute composite_score for every option and sort descending."""
    scored = []
    for opt in options:
        score = calculate_option_score(
            opt.evidence_support_score,
            opt.risk_score,
            opt.complexity_score,
            opt.strategic_fit_score
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

## GATE 2 — ROOT CAUSE RULE
Root cause must explain WHY the observation occurred. If evidence does not support a cause, write UNKNOWN.
Never copy the observation into the root cause field.

## GATE 3 — INFERENCE CLASSIFICATION (MANDATORY)
Every inference MUST end with: | CONFIRMED, | LIKELY, or | HYPOTHESIS
- CONFIRMED: directly stated in evidence with high-trust source
- LIKELY: strongly implied by evidence from medium-or-higher trust source
- HYPOTHESIS: logical but not directly evidenced

## GATE 4 — THEME EVIDENCE REQUIREMENT
A theme requires MINIMUM 2 observations OR 3 facts from verified evidence.
Never state a theme as a theme unless it appears in multiple evidence items.
Observation restatements cannot become themes.

## GATE 5 — COMPETITOR EVIDENCE RULE
Every competitor advantage and vulnerability claim MUST include a specific evidence citation.
Do not state competitor claims without evidence. Write "Insufficient evidence" rather than speculate.

## GATE 6 — OPTION SCORING (NUMERIC, 1-10)
Generate exactly 3 options: Conservative, Balanced, Aggressive.
Score each dimension 1-10:
- evidence_support_score: How strongly does verified evidence support this option?
- risk_score: How high is execution risk? (higher = riskier)
- complexity_score: How complex is execution? (higher = more complex)
- strategic_fit_score: How well does it fit the entity's strategic context?
Do NOT select a recommendation until after scoring all three options.
Only the highest composite-scoring option that passes the generic test becomes the recommendation.

## GATE 7 — GENERIC RECOMMENDATION REJECTION
The recommendation MUST be rejected and rewritten if it contains ANY of:
leverage synergies, best practices, holistic approach, consider expanding, may wish to,
explore opportunities, strengthen positioning, invest in capabilities, could potentially, it is recommended that
The recommendation must be specific to THIS entity, THIS evidence, THIS moment.

## GATE 8 — DECISION TRACEABILITY (MANDATORY FORMAT)
The recommended_decision field MUST use this exact format:
"Based on Obs: [observation], Inf: [inference], Theme: [theme name], Opt: [option type]: [specific action]"

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
      "inference": "strategic meaning | CONFIRMED/LIKELY/HYPOTHESIS",
      "observation_purity_passed": true,
      "inference_classified": true
    }}
  ],
  "strategic_themes_and_signals": [
    {{
      "name": "theme name",
      "type": "STRATEGIC THEME or EMERGING SIGNAL",
      "traceability": ["Observation 1 reference", "Observation 2 reference"]
    }}
  ],
  "competitive_landscape": [
    {{
      "competitor": "name",
      "advantage": "specific advantage or null",
      "advantage_evidence": "specific evidence citation — required if advantage stated",
      "vulnerability": "specific vulnerability or null",
      "vulnerability_evidence": "specific evidence citation — required if vulnerability stated"
    }}
  ],
  "evaluated_options": [
    {{
      "option_type": "Conservative",
      "description": "specific action unique to this entity and situation",
      "traceability_chain": "Theme [X] -> Inference [Y] -> Observation [Z]",
      "evidence_support_score": 7,
      "risk_score": 3,
      "complexity_score": 4,
      "strategic_fit_score": 8,
      "composite_score": 0,
      "generic_test_passed": "Yes",
      "rejection_reason": null
    }},
    {{
      "option_type": "Balanced",
      "description": "...",
      "traceability_chain": "...",
      "evidence_support_score": 6,
      "risk_score": 5,
      "complexity_score": 5,
      "strategic_fit_score": 7,
      "composite_score": 0,
      "generic_test_passed": "Yes",
      "rejection_reason": null
    }},
    {{
      "option_type": "Aggressive",
      "description": "...",
      "traceability_chain": "...",
      "evidence_support_score": 5,
      "risk_score": 8,
      "complexity_score": 7,
      "strategic_fit_score": 6,
      "composite_score": 0,
      "generic_test_passed": "Yes",
      "rejection_reason": null
    }}
  ],
  "recommended_decision": "Based on Obs: [...], Inf: [...], Theme: [...], Opt: [...]: [specific action]",
  "selected_option_type": "Conservative/Balanced/Aggressive",
  "selection_rationale": "Why this option scored highest...",
  "contradicting_evidence": "any evidence that contradicts the recommendation or None",
  "confidence_assessment": "Overall confidence X% — based on Y verified facts, Z high-trust sources"
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
            verified_facts, rejected_facts = run_hard_gate_validation(raw_facts)

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
        with st.expander(f"📊 Fact Quality Gate Report — {len(verified_facts)} passed / {len(rejected_facts)} rejected", expanded=False):
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

        # ── NEW: Traceability Validation Report ──────────────
        violations = validate_traceability_chain(final_brief)
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

        # 2. STRATEGIC THEMES & SIGNALS
        st.markdown("### 2. Strategic Themes & Signals")
        c1, c2 = st.columns(2)
        for i, ts in enumerate(final_brief.strategic_themes_and_signals):
            col = c1 if i % 2 == 0 else c2
            with col.container(border=True):
                st.subheader(ts.name or "Unnamed Theme")
                type_val   = ts.type or "UNKNOWN"
                type_color = "green" if "THEME" in type_val else "orange"
                st.markdown(f"**Type:** :{type_color}[{type_val}]")
                trace_count = len(ts.traceability)
                if trace_count < 2:
                    st.error(f"⚠️ Only {trace_count} traceability reference(s) — minimum 2 required")
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
                    if comp.advantage and not comp.advantage_evidence:
                        st.error("❌ No evidence backing for this advantage claim")
                    elif comp.advantage_evidence:
                        st.caption(f"**Evidence:** {comp.advantage_evidence}")
                with c_vuln:
                    st.error(f"**Vulnerability:** {comp.vulnerability or 'None explicitly supported'}")
                    if comp.vulnerability and not comp.vulnerability_evidence:
                        st.error("❌ No evidence backing for this vulnerability claim")
                    elif comp.vulnerability_evidence:
                        st.caption(f"**Evidence:** {comp.vulnerability_evidence}")

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
                    st.markdown(f"{opt.description or 'N/A'}")
                with score_col:
                    st.metric("Composite Score", f"{opt.composite_score}/100")

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Evidence Support", f"{opt.evidence_support_score}/10")
                sc2.metric("Risk (↑=worse)", f"{opt.risk_score}/10")
                sc3.metric("Complexity (↑=harder)", f"{opt.complexity_score}/10")
                sc4.metric("Strategic Fit", f"{opt.strategic_fit_score}/10")

                chain = opt.traceability_chain
                if isinstance(chain, list):
                    chain = " -> ".join(chain)
                st.info(f"**Traceability:** {chain or 'N/A'}")

                if opt.generic_test_passed == "Yes":
                    st.success("✅ Passed Generic Test — specific to this entity and situation.")
                elif opt.generic_test_passed == "No":
                    st.error(f"❌ Failed Generic Test — {opt.rejection_reason or 'Too generic.'}")

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

        # Export — now includes quality scores and rejection log
        st.divider()
        export = {
            "entity_profile": entity.model_dump(),
            "fact_quality_report": {
                "verified_facts": [vf.model_dump() for vf in verified_facts],
                "rejected_facts": rejected_facts,
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