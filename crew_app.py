import os
import json
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from urllib.parse import urlparse

# ==========================================
# 1. SETUP
# ==========================================
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY") or st.secrets.get("GROQ_KEY", "")
llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.1-8b-instant", temperature=0.1)

st.set_page_config(page_title="Strategic Intelligence Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ Strategic Intelligence Engine")
st.markdown("**Expert Reasoning System** · Strict Evidence Validation · Anti-Hallucination · Decision Support")
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

MIN_VERIFIED_FACTS   = 2
MIN_REPORT_CONFIDENCE = 50
ENTITY_CONFIDENCE_THRESHOLD = 60

def evaluate_trust(url: str) -> str:
    domain = urlparse(url).netloc.lower().replace("www.", "")
    if any(h in domain for h in HIGH_TRUST_DOMAINS):   return "HIGH TRUST"
    if any(m in domain for m in MEDIUM_TRUST_DOMAINS): return "MEDIUM TRUST"
    if any(l in domain for l in LOW_TRUST_DOMAINS):    return "LOW TRUST"
    return "MEDIUM TRUST"

# ==========================================
# 3. DETERMINISTIC SCORING
# ==========================================
def calculate_confidence(trust_label: str, board_relevance: int, strategic_impact: int) -> int:
    trust_score = TRUST_SCORE_MAP.get(trust_label.strip(), 5)
    raw = (trust_score * 0.4) + (board_relevance * 0.3) + (strategic_impact * 0.3)
    return int((raw / 10) * 100)

def calculate_entity_confidence(entity) -> tuple[int, str]:
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

def calculate_report_confidence(verified_facts: list, total_facts: int) -> int:
    if not verified_facts or total_facts == 0:
        return 15
    gate_rate = len(verified_facts) / total_facts
    avg_conf  = sum(f.confidence for f in verified_facts) / len(verified_facts)
    return int((gate_rate * 0.4 + avg_conf / 100 * 0.6) * 100)

def get_evidence_sufficiency(verified_facts: list, report_confidence: int) -> tuple[bool, str]:
    if len(verified_facts) < MIN_VERIFIED_FACTS:
        return False, f"Only {len(verified_facts)} fact(s) passed validation. Insufficient evidence."
    if report_confidence < MIN_REPORT_CONFIDENCE:
        return False, f"Report confidence {report_confidence}% is below threshold. Evidence quality is low."
    return True, "Evidence sufficient for reliable conclusions."

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
# 6. PYDANTIC MODELS (STRICT REASONING CHAIN)
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

class StrategicSignal(BaseModel):
    signal_type: str
    signal: str
    urgency: str
    implication: str

# --- EXPERT REASONING MODELS ---

class RootCauseValidation(BaseModel):
    cause: str = Field(description="The underlying mechanism driving the observation.")
    classification: str = Field(description="Must be exactly one of: CONFIRMED, LIKELY, HYPOTHESIS, UNKNOWN")
    supporting_evidence: str = Field(description="Exact fact or data point validating this cause.")
    confidence: str = Field(description="Score out of 100, justified by evidence quality.")

class StrategicTheme(BaseModel):
    theme: str = Field(description="The larger pattern indicated by multiple observations.")
    supporting_observations: List[str] = Field(description="List of isolated facts that together form this theme.")
    confidence: str = Field(description="Confidence level in this theme based on evidence synthesis.")

class CompetitiveImplication(BaseModel):
    competitor_name: str
    capability_advantage: str = Field(description="Specific capability creating an advantage.")
    capability_vulnerability: str = Field(description="Specific capability creating vulnerability.")
    advantage_durability: str = Field(description="Structural/Durable or Temporary?")
    supporting_evidence: str = Field(description="Evidence linking this competitive reality.")

class OptionEvaluation(BaseModel):
    option_name: str
    benefits: str
    risks: str
    complexity: str = Field(description="Assessment of implementation difficulty.")
    resource_requirements: str = Field(description="Capital, time, or talent needed.")
    evidence_support: str = Field(description="Evidence proving this option is viable.")

class StrategicDecision(BaseModel):
    recommended_decision: str
    why_selected: str = Field(description="Why this is superior, linked strictly to evidence.")
    why_alternatives_rejected: str = Field(description="Why other evaluated options were discarded.")
    supporting_evidence: str = Field(description="The exact evidence tracing back to the source.")
    contradicting_evidence: str = Field(description="Evidence that challenges this decision.")
    assumptions: str = Field(description="Assumptions made in selecting this recommendation.")
    specific_risks: str = Field(description="Risks of executing this decision.")

class DecisionIntelligenceBrief(BaseModel):
    status: str = Field(description="Must be exactly 'SUFFICIENT' or 'INSUFFICIENT_EVIDENCE'")
    reason: str = Field(description="Explanation of the sufficiency status.")
    evidence_summary: List[str] = Field(description="List of the core verified facts relied upon.")
    observations: List[str] = Field(description="Factual observations (WHAT happened, NOT why).")
    root_causes: List[RootCauseValidation] = Field(description="Validation of why things occurred.")
    strategic_themes: List[StrategicTheme] = Field(description="Patterns emerging from multiple observations.")
    competitive_implications: List[CompetitiveImplication]
    options_considered: List[OptionEvaluation]
    recommended_decisions: List[StrategicDecision]
    uncertainties: List[str] = Field(description="Explicitly state what is unknown, missing, or speculative.")
    confidence_assessment: str = Field(description="Assessment of confidence strictly reflecting evidence quality, inputs used, and weighting logic.")

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

def run_researcher(company: str, entity: EntityProfile, raw_context: str) -> List[IntelligenceFact]:
    prompt = f"""You are a Fact Extraction System. Extract highly specific, verifiable data for {entity.canonical_name}.
Return a JSON object:
{{
  "facts": [
    {{
      "category": "one of: {', '.join(FACT_CATEGORIES)}",
      "fact": "specific verifiable fact with numbers/dates where present",
      "source_url": "https://...",
      "source_trust": "HIGH TRUST or MEDIUM TRUST or LOW TRUST",
      "date_signal": "Q1 2025 or Undated",
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
            try: facts.append(IntelligenceFact(**f))
            except Exception: continue
        return facts
    except Exception as e:
        return []

def run_hard_gate_validation(facts: List[IntelligenceFact]) -> List[ValidatedFact]:
    verified = []
    for f in facts:
        if f.board_relevance < 8 or f.strategic_impact < 8: continue
        if "LOW TRUST" in f.source_trust: continue
        confidence = calculate_confidence(f.source_trust, f.board_relevance, f.strategic_impact)
        if confidence < 70: continue
        if f.date_signal == "Undated" and "HIGH TRUST" not in f.source_trust: continue
        verified.append(ValidatedFact(
            category=f.category, fact=f.fact, source_url=f.source_url, 
            source_trust=f.source_trust, date_signal=f.date_signal, 
            board_relevance=f.board_relevance, strategic_impact=f.strategic_impact, confidence=confidence,
        ))
    return verified

def run_signal_detector(company: str, verified_facts: List[ValidatedFact]) -> List[StrategicSignal]:
    if not verified_facts: return []
    fact_text = "\n".join([f"[{f.category}] FACT: {f.fact}" for f in verified_facts])
    prompt = f"""You are a Strategic Signal Detector.
Return a JSON object:
{{
  "signals": [
    {{
      "signal_type": "Emerging Threat, Strategic Inflection, Moat Erosion, etc.",
      "signal": "what is changing",
      "urgency": "IMMEDIATE, 90-DAY, 6-MONTH, WATCH",
      "implication": "specific decision affected"
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
    except Exception as e:
        return []

# --- AGENT 5: EXPERT REASONING SYSTEM ---
def run_expert_reasoner(
    company: str, entity: EntityProfile, verified_facts: List[ValidatedFact], 
    signals: List[StrategicSignal], evidence_sufficient: bool, sufficiency_message: str
) -> Optional[DecisionIntelligenceBrief]:

    fact_text = "\n".join([f"- {f.fact} (Source Trust: {f.source_trust})" for f in verified_facts]) if verified_facts else "INSUFFICIENT EVIDENCE."
    signal_text = "\n".join([f"- {s.signal} ({s.urgency})" for s in signals]) or "No validated signals."

    prompt = f"""You are an expert decision-intelligence architect and reasoning-system designer.
Your objective is to generate the most reliable decision possible from the available evidence.

==================================================
PRIMARY OBJECTIVE
Maximize: Accuracy, Decision quality, Evidence quality, Reasoning quality, Explainability.
Minimize: Assumptions, Hallucinations, Unsupported conclusions, Generic recommendations.
Accuracy is more important than confidence. Evidence is more important than narrative.

==================================================
DATA SUFFICIENCY GATE
If evidence is insufficient to make reliable decisions, you MUST set "status" to "INSUFFICIENT_EVIDENCE".
If status is INSUFFICIENT_EVIDENCE, leave themes, options, and decisions empty arrays.

==================================================
REQUIRED REASONING CHAIN
Every analysis must follow: Verified Evidence → Observation → Root Cause → Strategic Theme → Implication → Options → Decision → Risk Assessment.

- OBSERVATIONS: Describe WHAT the evidence shows. Do NOT explain why.
- ROOT CAUSES: Explain WHY. Must classify as: CONFIRMED, LIKELY, HYPOTHESIS, or UNKNOWN.
- STRATEGIC THEMES: Must emerge from MULTIPLE observations. Do not use isolated facts.
- COMPETITIVE IMPLICATIONS: Focus on capability advantage/vulnerability and durability.
- OPTIONS GENERATION: Evaluate Benefits, Risks, Complexity, Resources, and Evidence Support for at least 3 options.
- RECOMMENDATION VALIDATION (BOARD CHALLENGE): Why this option over alternatives? What evidence supports it? What evidence contradicts it?

==================================================
ANTI-HALLUCINATION RULES
Never invent facts, causes, metrics, percentages, market positions, outcomes, timelines, or confidence values.

Evidence Sufficiency Input: {'SUFFICIENT' if evidence_sufficient else 'INSUFFICIENT_EVIDENCE'} ({sufficiency_message})
Verified Evidence:
{fact_text}

Strategic Signals:
{signal_text}

OUTPUT STRICT JSON MATCHING THE PROVIDED PYDANTIC SCHEMA:
"""
    # Note: We rely on the `invoke_json` function and LLM understanding the schema from prompt context implicitly, 
    # but providing the exact structure guarantees compliance.
    
    schema_hint = """
{
  "status": "SUFFICIENT or INSUFFICIENT_EVIDENCE",
  "reason": "...",
  "evidence_summary": ["..."],
  "observations": ["Observation 1 (what, not why)", "..."],
  "root_causes": [{"cause": "...", "classification": "CONFIRMED/LIKELY/HYPOTHESIS/UNKNOWN", "supporting_evidence": "...", "confidence": "..."}],
  "strategic_themes": [{"theme": "...", "supporting_observations": ["..."], "confidence": "..."}],
  "competitive_implications": [{"competitor_name": "...", "capability_advantage": "...", "capability_vulnerability": "...", "advantage_durability": "...", "supporting_evidence": "..."}],
  "options_considered": [{"option_name": "...", "benefits": "...", "risks": "...", "complexity": "...", "resource_requirements": "...", "evidence_support": "..."}],
  "recommended_decisions": [{"recommended_decision": "...", "why_selected": "...", "why_alternatives_rejected": "...", "supporting_evidence": "...", "contradicting_evidence": "...", "assumptions": "...", "specific_risks": "..."}],
  "uncertainties": ["..."],
  "confidence_assessment": "..."
}"""

    try:
        data = invoke_json(prompt + schema_hint)
        return DecisionIntelligenceBrief(**data)
    except Exception as e:
        st.error(f"Reasoning Engine error: {e}")
        return None

# ==========================================
# 8. STREAMLIT UI
# ==========================================
company = st.text_input("Target Company / Entity:", placeholder="e.g. Zomato, Reliance, Tesla...")

if st.button("Run Expert Reasoning Analysis", type="primary"):
    if not company:
        st.error("Please enter an entity name.")
    else:
        with st.status(f"Executing Decision Intelligence Pipeline for {company}...", expanded=True) as status:

            st.write("📡 Search...")
            raw_context = run_enhanced_search(company)
            if not raw_context:
                st.error("Search returned no data.")
                st.stop()
            time.sleep(1)

            st.write("🔍 Entity Resolution...")
            entity = run_entity_resolution(company, raw_context)
            entity_conf, entity_conf_msg = calculate_entity_confidence(entity)
            if entity_conf < ENTITY_CONFIDENCE_THRESHOLD:
                st.warning(f"⚠️ **Low Entity Confidence** — {entity_conf_msg}")
            time.sleep(1)

            st.write("📊 Fact Extraction System...")
            raw_facts = run_researcher(company, entity, raw_context[:3000])

            st.write("🔒 Hard-Gate Evidence Validation...")
            verified_facts = run_hard_gate_validation(raw_facts)

            report_confidence_prelim = calculate_report_confidence(verified_facts, len(raw_facts))
            evidence_sufficient, sufficiency_message = get_evidence_sufficiency(verified_facts, report_confidence_prelim)
            if not evidence_sufficient:
                st.warning(f"⚠️ **Evidence Warning:** {sufficiency_message}")
            
            st.write("🔭 Signal Detector...")
            signals = run_signal_detector(company, verified_facts)
            time.sleep(1)

            st.write("⚖️ Expert Reasoning System — Executing analytical chain...")
            final_brief = run_expert_reasoner(
                company, entity, verified_facts, signals,
                evidence_sufficient, sufficiency_message
            )

            status.update(label="Analysis Complete", state="complete")

        if not final_brief:
            st.error("Reasoning Engine failed. Try again.")
            st.stop()

        # ==========================================
        # DISPLAY: EXPERT REASONING LAYER
        # ==========================================
        st.divider()
        st.header(f"Decision Intelligence Brief — {entity.canonical_name.upper()}")
        st.caption(f"**Entity Context:** {entity.industry} | {entity.sector} | {entity.primary_market}")

        if final_brief.status == "INSUFFICIENT_EVIDENCE":
            st.error(f"🛑 **DATA SUFFICIENCY GATE FAILED**")
            st.warning(f"**Reason:** {final_brief.reason}")
            st.info("Reliable conclusions cannot be generated from the available evidence. Strategy generation aborted.")
            st.stop()
        else:
            st.success(f"✅ **DATA SUFFICIENCY GATE PASSED**\n{final_brief.reason}")

        # 1. EVIDENCE SUMMARY
        st.markdown("### 1. Verified Evidence Summary")
        with st.container(border=True):
            for ev in final_brief.evidence_summary:
                st.markdown(f"- {ev}")

        # 2. OBSERVATIONS (WHAT)
        st.markdown("### 2. Verified Observations")
        with st.container(border=True):
            for obs in final_brief.observations:
                st.markdown(f"- {obs}")

        # 3. ROOT CAUSE VALIDATION (WHY)
        st.markdown("### 3. Root Cause Validation")
        for rc in final_brief.root_causes:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**Cause:** {rc.cause}")
                c1.markdown(f"**Evidence:** {rc.supporting_evidence}")
                c2.markdown(f"**Class:** `{rc.classification}`")
                c2.markdown(f"**Confidence:** {rc.confidence}")

        # 4. STRATEGIC THEMES
        st.markdown("### 4. Strategic Themes")
        for theme in final_brief.strategic_themes:
            with st.container(border=True):
                st.subheader(theme.theme)
                st.markdown("**Supporting Observations:**")
                for obs in theme.supporting_observations:
                    st.markdown(f"- {obs}")
                st.caption(f"Theme Confidence: {theme.confidence}")

        # 5. COMPETITIVE IMPLICATIONS
        st.markdown("### 5. Competitive Implications")
        for comp in final_brief.competitive_implications:
            with st.container(border=True):
                st.markdown(f"**Competitor:** {comp.competitor_name}")
                st.markdown(f"- **Advantage:** {comp.capability_advantage} *(Durability: {comp.advantage_durability})*")
                st.markdown(f"- **Vulnerability:** {comp.capability_vulnerability}")
                st.markdown(f"- **Evidence:** {comp.supporting_evidence}")

        # 6. OPTIONS ANALYSIS
        st.markdown("### 6. Options Evaluated")
        if final_brief.options_considered:
            tabs = st.tabs([opt.option_name for opt in final_brief.options_considered])
            for idx, tab in enumerate(tabs):
                with tab:
                    opt = final_brief.options_considered[idx]
                    c1, c2 = st.columns(2)
                    with c1:
                        st.success(f"**Benefits:** {opt.benefits}")
                        st.info(f"**Evidence Support:** {opt.evidence_support}")
                        st.markdown(f"**Resource Req:** {opt.resource_requirements}")
                    with c2:
                        st.error(f"**Risks:** {opt.risks}")
                        st.warning(f"**Complexity:** {opt.complexity}")
        else:
            st.info("No viable options generated from evidence.")

        # 7. RECOMMENDED DECISIONS (BOARD CHALLENGE TEST)
        st.markdown("### 7. Strategic Decisions")
        for i, dec in enumerate(final_brief.recommended_decisions, 1):
            with st.container(border=True):
                st.subheader(f"Decision #{i}: {dec.recommended_decision}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Why Selected:**")
                    st.success(dec.why_selected)
                    st.markdown("**Why Alternatives Rejected:**")
                    st.warning(dec.why_alternatives_rejected)
                    st.markdown("**Assumptions Made:**")
                    st.markdown(dec.assumptions)
                with c2:
                    st.markdown("**Supporting Evidence:**")
                    st.info(dec.supporting_evidence)
                    st.markdown("**Contradicting Evidence:**")
                    st.error(dec.contradicting_evidence)
                    st.markdown("**Specific Risks:**")
                    st.error(dec.specific_risks)

        # 8. UNCERTAINTIES & CONFIDENCE
        st.markdown("### 8. Risk & Uncertainty Assessment")
        with st.container(border=True):
            st.error("**Known Uncertainties & Missing Evidence:**")
            for unc in final_brief.uncertainties:
                st.markdown(f"- {unc}")
            st.divider()
            st.markdown("**System Confidence Assessment:**")
            st.markdown(final_brief.confidence_assessment)

        # Export
        st.divider()
        export = {
            "entity_profile": entity.model_dump(),
            "verified_facts": [vf.model_dump() for vf in verified_facts],
            "reasoning_brief": final_brief.model_dump(),
        }
        st.download_button(
            "Download Reasoning Package (JSON)",
            data=json.dumps(export, indent=2),
            file_name=f"{company.replace(' ','_')}_decision_intelligence.json",
            mime="application/json"
        )