import os
import json
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from urllib.parse import urlparse

# ==========================================
# 1. SETUP
# ==========================================
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY") or st.secrets.get("GROQ_KEY", "")
llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.1-8b-instant", temperature=0.1)

st.set_page_config(page_title="Strategic Intelligence Engine", page_icon="♟️", layout="wide")
st.title("♟️ Strategic Intelligence Engine")
st.markdown("**Decision-Support Platform** · Evidence-Backed · Hard-Gate Validated · Root Cause Reasoning")
st.divider()

# ==========================================
# 2. TRUST & SCORING
# ==========================================
HIGH_TRUST_DOMAINS = [
    "reuters.com","bloomberg.com","cnbc.com","wsj.com","ft.com","sec.gov",
    "moneycontrol.com","economictimes.indiatimes.com","livemint.com",
    "businessstandard.com","thehindubusinessline.com","financialexpress.com",
    "bseindia.com","nseindia.com","sebi.gov.in","rbi.org.in",
]
MEDIUM_TRUST_DOMAINS = [
    "techcrunch.com","forbes.com","inc42.com","entrackr.com",
    "yourstory.com","themorningcontext.com","restofworld.org","fortune.com",
]
LOW_TRUST_DOMAINS = [
    "linkedin.com","reddit.com","quora.com","wikipedia.org",
    "medium.com","twitter.com","x.com","substack.com",
]
TRUST_SCORE_MAP = {"HIGH TRUST": 10, "MEDIUM TRUST": 6, "LOW TRUST": 2}

# Minimum thresholds — if not met, report flags insufficient evidence
MIN_VERIFIED_FACTS   = 2    # below this → insufficient evidence warning
MIN_REPORT_CONFIDENCE = 50  # below this → speculative flag shown
ENTITY_CONFIDENCE_THRESHOLD = 60  # below this → halt and warn

def evaluate_trust(url: str) -> str:
    domain = urlparse(url).netloc.lower().replace("www.", "")
    if any(h in domain for h in HIGH_TRUST_DOMAINS):   return "HIGH TRUST"
    if any(m in domain for m in MEDIUM_TRUST_DOMAINS): return "MEDIUM TRUST"
    if any(l in domain for l in LOW_TRUST_DOMAINS):    return "LOW TRUST"
    return "MEDIUM TRUST"

# ==========================================
# 3. DETERMINISTIC SCORING (all formula-based)
# ==========================================
def calculate_confidence(trust_label: str, board_relevance: int, strategic_impact: int) -> int:
    trust_score = TRUST_SCORE_MAP.get(trust_label.strip(), 5)
    raw = (trust_score * 0.4) + (board_relevance * 0.3) + (strategic_impact * 0.3)
    return int((raw / 10) * 100)

def calculate_entity_confidence(entity) -> tuple[int, str]:
    """
    Returns (score 0-100, explanation).
    Deterministic: penalises unknown fields and contamination warnings.
    """
    score = 100
    reasons = []
    if entity.industry == "Unknown":
        score -= 20; reasons.append("industry unknown")
    if entity.sector == "Unknown":
        score -= 10; reasons.append("sector unknown")
    if entity.business_model == "Unknown":
        score -= 15; reasons.append("business model unknown")
    if entity.primary_market == "Unknown":
        score -= 10; reasons.append("primary market unknown")
    if entity.known_competitors == "Unknown":
        score -= 10; reasons.append("competitors unknown")
    contamination = entity.contamination_warnings.lower()
    if "failed" in contamination:
        score -= 25; reasons.append("entity resolution failed")
    elif "none" not in contamination and contamination != "":
        score -= 20; reasons.append(f"contamination risk: {entity.contamination_warnings}")
    explanation = f"Entity confidence {score}%"
    if reasons:
        explanation += f" — Issues: {', '.join(reasons)}"
    return max(0, score), explanation

def calculate_health_score(verified_facts: list, signals: list, competitors: list) -> int:
    score = 50
    for f in verified_facts:
        if f.category == "Profitability":   score += 5 if f.strategic_impact >= 9 else 2
        if f.category == "Growth":          score += 4 if f.strategic_impact >= 9 else 1
        if f.category == "Competitive Advantage": score += 3
        if f.category == "Competitive Threat":    score -= 2
    for s in signals:
        if s.urgency == "IMMEDIATE":  score -= 8
        elif s.urgency == "90-DAY":   score -= 4
        if any(t in s.signal_type for t in ["Threat","Erosion","Compression","Regulatory","Disruption"]): score -= 3
        if any(t in s.signal_type for t in ["Opportunity","Strengthening","Inflection"]): score += 3
    for c in competitors:
        if any(t in c.threat_type for t in ["Largest Threat","Fastest Growing"]): score -= 5
    return max(0, min(100, score))

def calculate_report_confidence(verified_facts: list, total_facts: int) -> int:
    if not verified_facts or total_facts == 0:
        return 15
    gate_rate = len(verified_facts) / total_facts
    avg_conf  = sum(f.confidence for f in verified_facts) / len(verified_facts)
    return int((gate_rate * 0.4 + avg_conf / 100 * 0.6) * 100)

def get_evidence_sufficiency(verified_facts: list, report_confidence: int) -> tuple[bool, str]:
    """
    Returns (is_sufficient, message).
    If insufficient: downstream agents are warned and output is hedged.
    """
    if len(verified_facts) < MIN_VERIFIED_FACTS:
        return False, (
            f"Only {len(verified_facts)} fact(s) passed validation (minimum {MIN_VERIFIED_FACTS}). "
            "Recommendations will be speculative. Verify with primary sources before acting."
        )
    if report_confidence < MIN_REPORT_CONFIDENCE:
        return False, (
            f"Report confidence {report_confidence}% is below threshold ({MIN_REPORT_CONFIDENCE}%). "
            "Evidence quality is low. Treat conclusions as directional only."
        )
    return True, "Evidence sufficient for reliable recommendations."

# ==========================================
# 4. SEARCH
# ==========================================
def run_enhanced_search(company: str) -> str:
    queries = [
        f"{company} revenue profit margin earnings 2025",
        f"{company} market share competitor comparison 2025",
        f"{company} capital allocation acquisition fundraise 2025",
        f"{company} regulatory risk supply chain disruption 2025",
        f"{company} strategic pivot AI investment new market 2025",
    ]
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=2, timelimit="y"):
                    url = r.get("href", "")
                    results.append(
                        f"SOURCE: {url}\nTRUST: {evaluate_trust(url)}\n"
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
        if text.startswith("json"): text = text[4:]
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
    root_cause: str
    business_driver: str
    strategic_implication: str
    is_structural: str        # NEW: "Structural" or "Temporary" — is the change permanent?
    why_it_matters: str
    decision_relevance: str   # NEW: "Could this change a decision?" answer
    board_relevance: int
    strategic_impact: int
    source_url: str
    source_trust: str
    date_signal: str = "Undated"
    competitor_context: str = "No benchmark available"

class ValidatedFact(BaseModel):
    category: str
    fact: str
    root_cause: str
    business_driver: str
    strategic_implication: str
    is_structural: str
    why_it_matters: str
    decision_relevance: str
    source_url: str
    source_trust: str
    date_signal: str
    competitor_context: str
    board_relevance: int
    strategic_impact: int
    confidence: int

class StrategicSignal(BaseModel):
    signal_type: str
    signal: str          # must explain WHY a fact matters, not restate it
    urgency: str
    root_cause: str
    implication: str     # NEW: explicit "so what" — what decision does this affect?
    evidence_fact: str
    is_restatement: bool = False  # NEW: flag signals that merely restate a fact

class CompetitorIntel(BaseModel):
    competitor_name: str
    threat_type: str
    threat_summary: str
    root_cause_of_threat: str
    capability_driving_advantage: str   # NEW: which specific capability makes them dangerous?
    structural_or_temporary: str        # NEW: is this threat permanent or a short-term blip?
    advantage_summary: str
    recommended_response: str

class StrategicAction(BaseModel):
    framework: str
    evidence: str
    root_cause: str
    implication: str
    competitor_context: str
    action: str
    why_this_action: str        # NEW: explicit justification — why this over alternatives?
    why_now: str                # NEW: why is timing critical?
    why_this_evidence: str      # NEW: explicit evidence-to-action linkage
    expected_impact: str
    risk: str
    alternatives_considered: str  # NEW: what else was considered and why rejected?
    timeline: str
    evidence_backed: bool = True  # NEW: self-validation flag

class CEOBrief(BaseModel):
    company_health_score: int
    report_confidence: int
    evidence_sufficiency: str   # NEW: explicit statement on data quality
    entity_context: str
    narrative_what_changed: str
    narrative_root_cause: str
    narrative_is_structural: str   # NEW: is the change structural or temporary?
    narrative_why_now: str
    narrative_primary_move: str
    biggest_opportunity: str
    biggest_risk: str
    most_important_competitor: str
    key_decision: str
    do_not_do: str
    alternatives: str              # NEW: what alternatives exist to the primary move?
    board_message: str
    prioritized_actions: List[StrategicAction]

# ==========================================
# 7. PIPELINE
#
# Search
# ↓
# Entity Resolution → entity_confidence check → HALT if < threshold
# ↓
# Researcher (decision-relevance filter + structural analysis)
# ↓
# Hard-Gate Validation (programmatic)
# ↓ [only verified_facts proceed — hard gate]
# Data Sufficiency Check → WARN if insufficient
# ↓
# Competitor Intelligence (capability analysis + structural/temporary flag)
# ↓
# Signal Detector (anti-restatement filter + implication required)
# ↓
# Deterministic scoring
# ↓
# Strategist (self-validation: every action must answer why this/why now/why this evidence)
# ==========================================

FACT_CATEGORIES = [
    "Profitability", "Growth", "Competitive Threat",
    "Competitive Advantage", "Capital Allocation", "Strategic Shift"
]

# --- AGENT 1: Entity Resolution ---
def run_entity_resolution(company: str, raw_context: str) -> EntityProfile:
    prompt = f"""You are an Entity Resolution Specialist. Identify exactly which company the search results describe.

Return a JSON object:
{{
  "canonical_name": "official registered name e.g. FSN E-Commerce Ventures Ltd (Nykaa)",
  "industry": "specific industry e.g. Beauty E-Commerce",
  "sector": "sector e.g. Consumer Discretionary",
  "business_model": "how it makes money e.g. Inventory-led B2C + owned brands",
  "primary_market": "main geography e.g. India",
  "known_subsidiaries": "e.g. Nykaa Fashion, Nykaa Pro — or Unknown",
  "known_competitors": "e.g. Purplle, Reliance Beauty — or Unknown",
  "contamination_warnings": "None detected — OR describe any results that appear to be about a different entity"
}}

Company queried: {company}
Search context: {raw_context[:1500]}"""
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


# --- AGENT 2: Researcher ---
def run_researcher(company: str, entity: EntityProfile, raw_context: str,
                   evidence_warning: str) -> List[IntelligenceFact]:
    prompt = f"""You are a Goldman Sachs Research Analyst extracting decision-relevant intelligence for {entity.canonical_name}.

Entity context:
- Industry: {entity.industry} | Sector: {entity.sector} | Market: {entity.primary_market}
- Business model: {entity.business_model}
- Known competitors: {entity.known_competitors}
- Contamination warning: {entity.contamination_warnings}

CONTAMINATION GUARD: Only extract facts explicitly about {entity.canonical_name} or its subsidiaries ({entity.known_subsidiaries}).

PRIMARY FILTER — Before accepting any fact, answer: "Could this information change a decision?"
If NO — reject it. This is not about collection. It is about decision support.

Return a JSON object:
{{
  "facts": [
    {{
      "category": "one of: {', '.join(FACT_CATEGORIES)}",
      "fact": "specific verifiable fact with numbers/dates where present",
      "root_cause": "WHY did this happen? Underlying cause, not description.",
      "business_driver": "what business mechanism produced this outcome?",
      "strategic_implication": "what does this mean for competitive position?",
      "is_structural": "Structural (permanent shift) or Temporary (short-term blip) — explain briefly",
      "why_it_matters": "why would this disappearing change a board decision?",
      "decision_relevance": "what specific decision could this information change?",
      "board_relevance": 9,
      "strategic_impact": 9,
      "source_url": "https://...",
      "source_trust": "HIGH TRUST or MEDIUM TRUST or LOW TRUST",
      "date_signal": "Q1 2025 or Undated",
      "competitor_context": "vs [NamedCompetitor] or No benchmark available"
    }}
  ]
}}

Return EXACTLY 6 facts — one per category: {', '.join(FACT_CATEGORIES)}

HARD REJECT:
- Founding dates, history, awards, PR, executive bios, social media
- Product launches older than 18 months
- Facts about other companies
- Generic industry trends with no company-specific data
- Facts where board_relevance < 8 OR strategic_impact < 8

{f"DATA WARNING: {evidence_warning} — Flag uncertain facts explicitly." if evidence_warning else ""}

NEVER invent numbers. When no hard data exists: use qualitative language and flag uncertainty.
source_trust: copy EXACTLY from TRUST label (HIGH TRUST / MEDIUM TRUST / LOW TRUST).

Raw Context:
{raw_context}"""
    try:
        data = invoke_json(prompt)
        facts = []
        for f in data.get("facts", []):
            try:
                facts.append(IntelligenceFact(**f))
            except Exception:
                continue
        return facts
    except Exception as e:
        st.warning(f"Researcher error: {e}")
        return []


# --- GATE: Hard Validation (programmatic) ---
def run_hard_gate_validation(facts: List[IntelligenceFact]) -> List[ValidatedFact]:
    verified = []
    for f in facts:
        if f.board_relevance < 8 or f.strategic_impact < 8:
            continue
        if "LOW TRUST" in f.source_trust:
            continue
        confidence = calculate_confidence(f.source_trust, f.board_relevance, f.strategic_impact)
        if confidence < 70:
            continue
        if f.date_signal == "Undated" and "HIGH TRUST" not in f.source_trust:
            continue
        verified.append(ValidatedFact(
            category=f.category, fact=f.fact,
            root_cause=f.root_cause, business_driver=f.business_driver,
            strategic_implication=f.strategic_implication,
            is_structural=f.is_structural,
            why_it_matters=f.why_it_matters, decision_relevance=f.decision_relevance,
            source_url=f.source_url, source_trust=f.source_trust,
            date_signal=f.date_signal, competitor_context=f.competitor_context,
            board_relevance=f.board_relevance, strategic_impact=f.strategic_impact,
            confidence=confidence,
        ))
    return verified


# --- AGENT 3: Competitor Intelligence ---
def run_competitor_intel(company: str, entity: EntityProfile,
                         raw_context: str, evidence_sufficient: bool) -> List[CompetitorIntel]:
    sufficiency_note = (
        "NOTE: Evidence is limited. Only include competitors explicitly mentioned in the context."
        if not evidence_sufficient else ""
    )
    prompt = f"""You are a Competitive Intelligence Specialist analysing {entity.canonical_name}.
{sufficiency_note}

Known competitors: {entity.known_competitors}
Industry: {entity.industry} | Market: {entity.primary_market}

Do NOT describe competitors. Explain WHY they are succeeding or struggling.
Focus on capabilities that create advantage or vulnerabilities that create risk.

Return a JSON object:
{{
  "competitors": [
    {{
      "competitor_name": "exact company name",
      "threat_type": "one of: Fastest Growing, Largest Threat, Weakening Moat, Strengthening Moat, Competitive Surprise, Most Likely Future Threat",
      "threat_summary": "what specific move or metric signals threat — with data if available",
      "root_cause_of_threat": "WHY are they gaining? What underlying capability or dynamic drives this?",
      "capability_driving_advantage": "the specific capability (tech/distribution/cost/brand/network) making them dangerous",
      "structural_or_temporary": "is this threat a structural shift or a temporary surge? explain",
      "advantage_summary": "where {entity.canonical_name} still leads and why it is defensible",
      "recommended_response": "specific counter-move naming markets, product lines, or channels — not generic advice"
    }}
  ]
}}

Identify up to 3 NAMED competitors. Use real company names.
FORBIDDEN: improve innovation, focus on customers, optimize operations, review strategy.

Raw Context:
{raw_context}"""
    try:
        data = invoke_json(prompt)
        comps = []
        for c in data.get("competitors", []):
            try:
                comps.append(CompetitorIntel(**c))
            except Exception:
                continue
        return comps
    except Exception as e:
        st.warning(f"Competitor intel error: {e}")
        return []


# --- AGENT 4: Signal Detector (anti-restatement filter) ---
def run_signal_detector(company: str, verified_facts: List[ValidatedFact],
                        evidence_sufficient: bool) -> List[StrategicSignal]:
    if not verified_facts:
        return []

    fact_text = "\n".join([
        f"[{f.category} | {f.is_structural}] FACT: {f.fact}\n"
        f"  Root Cause: {f.root_cause}\n"
        f"  Implication: {f.strategic_implication}\n"
        f"  Decision Relevance: {f.decision_relevance}"
        for f in verified_facts
    ])
    sufficiency_note = (
        "NOTE: Evidence is limited. Only generate signals with strong factual grounding. "
        "Mark any uncertain signals with urgency=WATCH."
        if not evidence_sufficient else ""
    )

    prompt = f"""You are a Strategic Signal Detector for {company}.
{sufficiency_note}

CRITICAL RULE: A signal must explain WHY a fact matters — not restate the fact.
BAD signal: "Revenue grew 25% YoY" (restatement)
GOOD signal: "Revenue growth is outpacing cost structure, indicating emerging margin expansion opportunity"

Return a JSON object:
{{
  "signals": [
    {{
      "signal_type": "one of: Emerging Threat, Emerging Opportunity, Strategic Inflection, Capital Shift, Competitive Surprise, Moat Erosion, Moat Strengthening, Regulatory Risk, Margin Compression, Pricing Pressure, Technology Disruption",
      "signal": "what is changing — the inflection, not the fact",
      "urgency": "one of: IMMEDIATE, 90-DAY, 6-MONTH, WATCH",
      "root_cause": "underlying driver making this a strategic signal",
      "implication": "what specific decision or action does this signal affect?",
      "evidence_fact": "the exact validated fact that triggered this signal",
      "is_restatement": false
    }}
  ]
}}

For each signal ask: "Does this explain WHY the fact matters, or just repeat it?"
If it just repeats the fact, set is_restatement=true and the system will discard it.

Validated Facts (use ONLY these):
{fact_text}"""
    try:
        data = invoke_json(prompt)
        signals = []
        for s in data.get("signals", []):
            try:
                sig = StrategicSignal(**s)
                if not sig.is_restatement:   # filter restatements programmatically
                    signals.append(sig)
            except Exception:
                continue
        return signals
    except Exception as e:
        st.warning(f"Signal detector error: {e}")
        return []


# --- AGENT 5: Strategist (with self-validation) ---
def run_strategist(
    company: str, entity: EntityProfile,
    verified_facts: List[ValidatedFact], signals: List[StrategicSignal],
    competitors: List[CompetitorIntel],
    health_score: int, report_confidence: int,
    evidence_sufficient: bool, sufficiency_message: str
) -> Optional[CEOBrief]:

    fact_text = "\n".join([
        f"[{f.category} | {f.confidence}% | {f.is_structural}] {f.fact}\n"
        f"  Root Cause: {f.root_cause} | Implication: {f.strategic_implication}\n"
        f"  Decision relevance: {f.decision_relevance}"
        for f in verified_facts
    ]) if verified_facts else f"INSUFFICIENT EVIDENCE for {entity.canonical_name}."

    signal_text = "\n".join([
        f"[{s.signal_type}|{s.urgency}] {s.signal}\n  Root Cause: {s.root_cause} | Affects decision: {s.implication}"
        for s in signals
    ]) or "No validated signals."

    competitor_text = "\n".join([
        f"[{c.threat_type}|{c.structural_or_temporary}] {c.competitor_name}: {c.threat_summary}\n"
        f"  Why gaining: {c.root_cause_of_threat} | Key capability: {c.capability_driving_advantage}"
        for c in competitors
    ]) or "No competitor data."

    sufficiency_instruction = (
        f'  "evidence_sufficiency": "SPECULATIVE — {sufficiency_message}",'
        if not evidence_sufficient else
        f'  "evidence_sufficiency": "Sufficient — {sufficiency_message}",'
    )

    prompt = f"""You are a McKinsey Senior Partner presenting to the Board of {entity.canonical_name}.
Company: {entity.canonical_name} | Industry: {entity.industry} | Market: {entity.primary_market}

SELF-VALIDATION RULE: Before including any recommendation, answer all 3 questions:
1. "Can this action be directly justified by the validated evidence below?"
2. "Why this action over alternatives?"
3. "Why is timing critical now?"
If any answer is "I don't know" or requires invented data — exclude the recommendation.

Return a JSON object:
{{
  "company_health_score": {health_score},
  "report_confidence": {report_confidence},
{sufficiency_instruction}
  "entity_context": "{entity.canonical_name} | {entity.industry} | {entity.primary_market}",
  "narrative_what_changed": "specific recent shift with evidence citation",
  "narrative_root_cause": "WHY did this change? Structural driver, not surface observation.",
  "narrative_is_structural": "Is this change permanent (structural) or temporary? Explain.",
  "narrative_why_now": "specific catalyst making delay costly",
  "narrative_primary_move": "single most important action — names specific market/product/channel",
  "biggest_opportunity": "highest-upside move — evidence-backed, no invented numbers",
  "biggest_risk": "most dangerous unaddressed threat — evidence-backed",
  "most_important_competitor": "which competitor and specifically why they matter most now",
  "key_decision": "the one decision the board must make in 90 days — and what evidence drives it",
  "do_not_do": "most tempting but wrong move — explain why the evidence argues against it",
  "alternatives": "what alternatives to the primary move exist, and why the primary move is preferred",
  "board_message": "3 sentences: root-cause finding + urgency + single call to action. No generic phrases.",
  "prioritized_actions": [
    {{
      "framework": "STOP or START or DOUBLE DOWN",
      "evidence": "exact verified fact being acted on",
      "root_cause": "why this fact demands action",
      "implication": "what happens competitively if ignored",
      "competitor_context": "named competitor positioning on this dimension",
      "action": "directive naming specific market, product line, channel, or node",
      "why_this_action": "why this action over alternatives",
      "why_now": "why is timing critical — what changes if delayed 6 months?",
      "why_this_evidence": "explicit link: how does this evidence justify this specific action?",
      "expected_impact": "qualitative outcome — express uncertainty if evidence is weak",
      "risk": "primary risk if taken OR if ignored",
      "alternatives_considered": "what else was considered and why rejected",
      "timeline": "90 Days or 6 Months or Q3 2025 — future-dated only",
      "evidence_backed": true
    }}
  ]
}}

HARD RULES:
- EXACTLY 3 prioritized_actions, ranked highest impact first
- NEVER invent dollar values, percentages, market share, or financial forecasts
- When evidence is weak: use "Potentially..." or "Evidence suggests..." — never state as fact
- Actions must name specific markets, products, channels, or supply chain nodes
- FORBIDDEN language: improve innovation, focus on customers, optimize operations, review strategy, increase efficiency
- company_health_score: {health_score} (do not change)
- report_confidence: {report_confidence} (do not change)

Verified Evidence:
{fact_text}

Strategic Signals:
{signal_text}

Competitor Intelligence:
{competitor_text}"""

    try:
        data = invoke_json(prompt)
        actions = []
        for a in data.get("prioritized_actions", []):
            try:
                act = StrategicAction(**a)
                if act.evidence_backed:    # discard self-flagged unsupported actions
                    actions.append(act)
            except Exception:
                continue
        data["prioritized_actions"] = actions
        data["company_health_score"] = health_score
        data["report_confidence"]    = report_confidence
        return CEOBrief(**data)
    except Exception as e:
        st.error(f"Strategist error: {e}")
        return None


# ==========================================
# 8. STREAMLIT UI
# ==========================================
company = st.text_input("Target Company:", placeholder="e.g. Zomato, Reliance, Tesla, Nykaa...")

if st.button("Run Strategic Analysis", type="primary"):
    if not company:
        st.error("Please enter a company name.")
    else:
        with st.status(f"Compiling Decision Intelligence on {company}...", expanded=True) as status:

            # STEP 1
            st.write("📡 Search...")
            raw_context = run_enhanced_search(company)
            if not raw_context:
                st.error("Search returned no data.")
                st.stop()
            time.sleep(3)

            # STEP 2 — Entity resolution + confidence check
            st.write("🔍 Entity Resolution...")
            entity = run_entity_resolution(company, raw_context)
            entity_conf, entity_conf_msg = calculate_entity_confidence(entity)
            st.write(f"   → {entity.canonical_name} | Entity confidence: {entity_conf}%")

            if entity_conf < ENTITY_CONFIDENCE_THRESHOLD:
                st.warning(
                    f"⚠️ **Low Entity Confidence ({entity_conf}%)** — {entity_conf_msg}\n\n"
                    "The system cannot reliably identify the company from search results. "
                    "Results may be inaccurate. Try a more specific company name (e.g. include country or full legal name)."
                )
            time.sleep(3)

            # STEP 3 — Research
            st.write("📊 Researcher — decision-relevance filter + structural analysis...")
            raw_facts = run_researcher(company, entity, raw_context[:3000], "")

            # STEP 4 — Hard gate
            st.write("🔒 Hard-Gate Validation...")
            verified_facts = run_hard_gate_validation(raw_facts)
            st.write(f"   → {len(raw_facts)} extracted · {len(verified_facts)} passed gate")

            # STEP 5 — Data sufficiency check (before any downstream agent sees data)
            report_confidence_prelim = calculate_report_confidence(verified_facts, len(raw_facts))
            evidence_sufficient, sufficiency_message = get_evidence_sufficiency(
                verified_facts, report_confidence_prelim
            )
            if not evidence_sufficient:
                st.warning(f"⚠️ **Evidence Sufficiency Warning:** {sufficiency_message}")

            time.sleep(4)

            # STEP 6 — Competitor Intel
            st.write("🎯 Competitor Intelligence — capability analysis...")
            competitors = run_competitor_intel(company, entity, raw_context[:2000], evidence_sufficient)

            time.sleep(4)

            # STEP 7 — Signal Detector (validated facts only, anti-restatement)
            st.write("🔭 Signal Detector — anti-restatement filter active...")
            signals = run_signal_detector(company, verified_facts, evidence_sufficient)

            # STEP 8 — Final deterministic scores
            health_score      = calculate_health_score(verified_facts, signals, competitors)
            report_confidence = calculate_report_confidence(verified_facts, len(raw_facts))

            time.sleep(4)

            # STEP 9 — Strategist with self-validation
            st.write("📋 Strategist — self-validating recommendations...")
            final_brief = run_strategist(
                company, entity, verified_facts, signals, competitors,
                health_score, report_confidence, evidence_sufficient, sufficiency_message
            )

            status.update(label="Analysis Complete", state="complete")

        if not final_brief:
            st.error("Strategist failed. Try again.")
            st.stop()

        # ==========================================
        # DISPLAY
        # ==========================================

        # Sufficiency banner
        if not evidence_sufficient:
            st.error(
                f"⚠️ **SPECULATIVE REPORT** — {sufficiency_message}\n\n"
                "Do not act on these recommendations without verifying against primary sources."
            )
        else:
            st.success("✅ Evidence sufficient for reliable recommendations.")

        # Pipeline metrics
        st.subheader("🛡️ Intelligence Pipeline")
        total  = len(raw_facts)
        passed = len(verified_facts)
        rate   = int(passed / total * 100) if total else 0

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Facts Extracted",   total)
        m2.metric("Passed Hard Gate",  passed)
        m3.metric("Gate Pass Rate",    f"{rate}%")
        m4.metric("Signals",           len(signals))
        m5.metric("Competitors",       len(competitors))
        m6.metric("Entity Confidence", f"{entity_conf}%")

        with st.expander("🔬 Full Pipeline Detail"):

            # Entity
            st.markdown("**🏢 Entity Profile**")
            with st.container(border=True):
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Company:** {entity.canonical_name}")
                    st.markdown(f"**Industry:** {entity.industry} | **Sector:** {entity.sector}")
                    st.markdown(f"**Business Model:** {entity.business_model}")
                with c2:
                    st.markdown(f"**Market:** {entity.primary_market}")
                    st.markdown(f"**Subsidiaries:** {entity.known_subsidiaries}")
                    st.markdown(f"**Competitors:** {entity.known_competitors}")
                st.caption(entity_conf_msg)
                if "None" not in entity.contamination_warnings and entity.contamination_warnings:
                    st.warning(f"⚠️ Contamination: {entity.contamination_warnings}")

            # Verified facts
            st.divider()
            st.markdown("**✅ Verified Facts**")
            for vf in verified_facts:
                with st.container(border=True):
                    st.success(f"**[{vf.category} | {vf.confidence}% | {vf.source_trust} | {vf.date_signal}]**\n\n{vf.fact}")
                    st.markdown(f"🔍 **Root Cause:** {vf.root_cause}")
                    st.markdown(f"⚙️ **Business Driver:** {vf.business_driver}")
                    st.markdown(f"📐 **Structural or Temporary:** {vf.is_structural}")
                    st.markdown(f"🎯 **Strategic Implication:** {vf.strategic_implication}")
                    st.markdown(f"📋 **Decision Relevance:** {vf.decision_relevance}")
                    st.caption(f"Competitor context: {vf.competitor_context} | Source: {vf.source_url}")

            # Rejected facts
            st.divider()
            st.markdown("**❌ Rejected Facts**")
            rejected = [f for f in raw_facts if not any(vf.fact == f.fact for vf in verified_facts)]
            for rf in rejected:
                conf = calculate_confidence(rf.source_trust, rf.board_relevance, rf.strategic_impact)
                reasons = []
                if rf.board_relevance < 8:        reasons.append(f"board_relevance={rf.board_relevance}")
                if rf.strategic_impact < 8:        reasons.append(f"strategic_impact={rf.strategic_impact}")
                if "LOW TRUST" in rf.source_trust: reasons.append("LOW TRUST")
                if conf < 70:                      reasons.append(f"confidence={conf}%")
                if rf.date_signal == "Undated":    reasons.append("Undated")
                st.error(f"**[{rf.category} | {conf}%]** {rf.fact}\n\n*Rejected: {' · '.join(reasons) or 'gate criteria'}*")

            # Signals
            if signals:
                st.divider()
                st.markdown("**🔭 Strategic Signals (restatements filtered out)**")
                for s in signals:
                    icon = "🔴" if s.urgency=="IMMEDIATE" else "🟡" if s.urgency=="90-DAY" else "🟢"
                    with st.container(border=True):
                        st.info(f"{icon} **[{s.signal_type} | {s.urgency}]** {s.signal}")
                        st.markdown(f"🔍 **Root Cause:** {s.root_cause}")
                        st.markdown(f"📋 **Decision affected:** {s.implication}")
                        st.caption(f"Evidence: {s.evidence_fact}")

            # Competitor intel
            if competitors:
                st.divider()
                st.markdown("**🎯 Competitor Intelligence**")
                for c in competitors:
                    with st.container(border=True):
                        st.warning(f"**[{c.threat_type}] {c.competitor_name}:** {c.threat_summary}")
                        st.markdown(f"🔍 **Why gaining:** {c.root_cause_of_threat}")
                        st.markdown(f"⚙️ **Key capability:** {c.capability_driving_advantage}")
                        st.markdown(f"📐 **Structural or temporary:** {c.structural_or_temporary}")

        # Board brief
        st.divider()
        h1,h2,h3 = st.columns([3,1,1])
        with h1: st.header(f"Board-Level Strategic Brief — {entity.canonical_name.upper()}")
        with h2: st.metric("Health Score",     f"{final_brief.company_health_score}/100")
        with h3: st.metric("Report Confidence",f"{final_brief.report_confidence}%")

        st.caption(f"Entity: {final_brief.entity_context} | Evidence: {final_brief.evidence_sufficiency}")

        # Board message
        st.markdown("### 📢 Board Message")
        with st.container(border=True):
            st.markdown(f"*{final_brief.board_message}*")

        # Key decision
        st.markdown("### ⚡ Key Decision")
        with st.container(border=True):
            st.error(f"**{final_brief.key_decision}**")

        # Narrative
        st.markdown("### Strategic Narrative")
        with st.container(border=True):
            st.markdown(f"**📉 What Changed:** {final_brief.narrative_what_changed}")
            st.markdown(f"**🔍 Root Cause:** {final_brief.narrative_root_cause}")
            st.markdown(f"**📐 Structural or Temporary:** {final_brief.narrative_is_structural}")
            st.markdown(f"**⏳ Why Now:** {final_brief.narrative_why_now}")
            st.markdown(f"**🎯 Primary Move:** {final_brief.narrative_primary_move}")
            st.markdown(f"**🔀 Alternatives:** {final_brief.alternatives}")

        # Opp / Risk / Do Not Do
        o1,o2,o3 = st.columns(3)
        with o1:
            with st.container(border=True):
                st.markdown("**🚀 Biggest Opportunity**"); st.success(final_brief.biggest_opportunity)
        with o2:
            with st.container(border=True):
                st.markdown("**⚠️ Biggest Risk**"); st.error(final_brief.biggest_risk)
        with o3:
            with st.container(border=True):
                st.markdown("**🚫 Do NOT Do**"); st.warning(final_brief.do_not_do)

        # Competitor benchmarks
        if competitors:
            st.markdown("### 🏆 Competitor Benchmarks")
            st.info(f"**Most Important Competitor:** {final_brief.most_important_competitor}")
            for c in competitors:
                with st.container(border=True):
                    st.markdown(f"#### ⚔️ {c.competitor_name} — {c.threat_type}")
                    ca,cb = st.columns(2)
                    with ca:
                        st.markdown("**Their Threat**"); st.error(c.threat_summary)
                        st.caption(f"Key capability: {c.capability_driving_advantage}")
                        st.caption(f"Structural or temporary: {c.structural_or_temporary}")
                    with cb:
                        st.markdown(f"**{entity.canonical_name}'s Edge**"); st.success(c.advantage_summary)
                    st.markdown(f"**Counter-Move:** {c.recommended_response}")

        # Actions
        st.markdown("### Prioritized Strategic Directives")
        for i, action in enumerate(final_brief.prioritized_actions, 1):
            icon = "🔴" if action.framework=="STOP" else "🟢" if action.framework=="START" else "🔥"
            with st.container(border=True):
                st.markdown(f"#### #{i} {icon} **{action.framework}**: {action.action}")
                a1,a2 = st.columns(2)
                with a1:
                    st.markdown("**1. Evidence**");           st.info(f"*{action.evidence}*")
                    st.markdown("**2. Root Cause**");         st.markdown(f"🔍 {action.root_cause}")
                    st.markdown("**3. Implication**");        st.warning(action.implication)
                    st.markdown("**4. Competitor Context**"); st.caption(action.competitor_context)
                    st.markdown("**5. Alternatives Considered**"); st.caption(action.alternatives_considered)
                with a2:
                    st.markdown("**6. Why This Action**");   st.markdown(action.why_this_action)
                    st.markdown("**7. Why Now**");            st.markdown(action.why_now)
                    st.markdown("**8. Timeline**");           st.write(f"📅 {action.timeline}")
                    st.markdown("**9. Expected Impact**");    st.success(action.expected_impact)
                    st.markdown("**10. Risk**");              st.error(action.risk)

        # Export
        st.divider()
        export = {
            "company": company,
            "entity_profile": entity.model_dump(),
            "entity_confidence": entity_conf,
            "evidence_sufficient": evidence_sufficient,
            "sufficiency_message": sufficiency_message,
            "pipeline": {"extracted": total, "passed": passed, "rate_pct": rate,
                         "signals": len(signals), "competitors": len(competitors)},
            "verified_facts": [vf.model_dump() for vf in verified_facts],
            "signals": [s.model_dump() for s in signals],
            "competitor_intel": [c.model_dump() for c in competitors],
            "board_brief": final_brief.model_dump(),
        }
        st.download_button(
            "Download Full Intelligence Package (JSON)",
            data=json.dumps(export, indent=2),
            file_name=f"{company.replace(' ','_')}_board_brief.json",
            mime="application/json"
        )