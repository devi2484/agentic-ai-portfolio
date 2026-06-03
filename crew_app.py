import os
import json
import time
import streamlit as st
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
st.markdown("**Evidence-Based Reasoning Engine** · Strict Traceability · Decision Validation")
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
    if entity.industry == "Unknown": score -= 20; reasons.append("industry unknown")
    if entity.sector == "Unknown": score -= 10; reasons.append("sector unknown")
    if entity.business_model == "Unknown": score -= 15; reasons.append("business model unknown")
    if entity.primary_market == "Unknown": score -= 10; reasons.append("primary market unknown")
    if entity.known_competitors == "Unknown": score -= 10; reasons.append("competitors unknown")
    
    contamination = entity.contamination_warnings.lower()
    if "failed" in contamination: score -= 25; reasons.append("entity resolution failed")
    elif "none" not in contamination and contamination != "": score -= 20; reasons.append(f"contamination risk: {entity.contamination_warnings}")
    
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
# 6. PYDANTIC MODELS (BULLETPROOF SCHEMAS)
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

# --- EXPERT REASONING MODELS (FULLY OPTIONALIZED) ---

class EvidenceLog(BaseModel):
    evidence: Optional[str] = None
    observation: Optional[str] = None
    root_cause_and_class: Optional[str] = Field(default=None, description="Format: [Cause] | [CONFIRMED/LIKELY/HYPOTHESIS/UNKNOWN]")

class ThemeSignal(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = Field(default=None, description="Must be STRATEGIC THEME or EMERGING SIGNAL")
    traceability: List[str] = Field(default_factory=list, description="List of observations/facts supporting this")

class CompetitiveLandscape(BaseModel):
    competitor: Optional[str] = None
    advantage: Optional[str] = None
    advantage_evidence: Optional[str] = None
    vulnerability: Optional[str] = None
    vulnerability_evidence: Optional[str] = None

class EvaluatedOption(BaseModel):
    description: Optional[str] = None
    # Allowed to be a string or a list to prevent crashes when LLM formats it as steps
    traceability_chain: Union[str, List[str], None] = Field(default=None, description="Supported by Theme [X] -> Root Cause [Y] -> Observation [Z] -> Evidence [A]")
    generic_test_passed: Optional[str] = Field(default=None, description="Yes or No")

class DecisionIntelligenceBrief(BaseModel):
    status: str = Field(description="Must be exactly 'SUFFICIENT' or 'INSUFFICIENT_EVIDENCE'")
    reason: Optional[str] = Field(default=None, description="Explanation of the sufficiency status based on input gates.")
    
    evidence_and_observation_log: List[EvidenceLog] = Field(default_factory=list)
    strategic_themes_and_signals: List[ThemeSignal] = Field(default_factory=list)
    competitive_landscape: List[CompetitiveLandscape] = Field(default_factory=list)
    evaluated_options: List[EvaluatedOption] = Field(default_factory=list)
    
    recommended_decision: Optional[str] = Field(default=None, description="The final recommendation, if evidence permits.")
    contradicting_evidence: Optional[str] = Field(default=None, description="Evidence challenging the decision, if applicable.")
    confidence_assessment: Optional[str] = Field(default=None, description="System confidence, if a conclusion was reached.")

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

    prompt = f"""# SYSTEM INSTRUCTIONS: EVIDENCE-BASED REASONING ENGINE

## ROLE & OBJECTIVE
You are a strict Evidence-Based Reasoning Engine. Your primary responsibility is NOT to generate conclusions, but to rigorously construct and verify reasoning chains derived EXCLUSIVELY from provided data. 
You must maximize accuracy, traceability, and reliability. Do not attempt to sound persuasive, strategic, or artificially confident.

## CORE PRINCIPLE: STRICT TRACEABILITY
A conclusion is valid ONLY if every reasoning step is explicitly traced back to the original evidence. Plausible reasoning, general business wisdom, and industry assumptions are strictly prohibited. 

Every output MUST follow this exact traceability chain:
[Evidence] -> [Observation] -> [Root Cause] -> [Theme] -> [Option] -> [Decision]

If any step in this chain is broken or missing, you must REJECT the conclusion.

---

## EXECUTION PROTOCOL & VALIDATION GATES

### GATE 0: DATA SUFFICIENCY
If evidence is insufficient to make reliable decisions, you MUST set "status" to "INSUFFICIENT_EVIDENCE".
If status is INSUFFICIENT_EVIDENCE, populate the reason, but leave arrays empty or null for decisions.

### GATE 1: OBSERVATION VALIDATION
* **Rule:** Observations must describe EXACTLY what the evidence shows. 
* **Constraint:** You may NOT introduce new information or explanations. 
* *Example:* If evidence says "Revenue increased," the observation is "Revenue increased." "Customer demand increased" is an invalid observation (it is an explanation).

### GATE 2: ROOT CAUSE CLASSIFICATION
Every identified root cause MUST be strictly classified into one of four categories:
* **CONFIRMED:** Explicitly and directly supported by provided evidence.
* **LIKELY:** Strongly suggested by multiple converging data points.
* **HYPOTHESIS:** Plausible based on data, but unproven.
* **UNKNOWN:** Insufficient evidence to determine a cause.
* *Constraint:* Never classify a cause as CONFIRMED unless the text explicitly proves it.

### GATE 3: THEME & SIGNAL VALIDATION
* **STRATEGIC THEME:** Requires a minimum of TWO (2) independent observations OR THREE (3) supporting facts.
* **EMERGING SIGNAL:** If the threshold for a Theme is not met, it must be labeled as an Emerging Signal. Do not build strategic decisions solely on Emerging Signals.

### GATE 4: COMPETITIVE VALIDATION
* **Rule:** Do NOT provide general descriptions of competitors.
* **Constraint:** Only identify explicitly evidence-backed advantages and evidence-backed vulnerabilities. If none exist in the data, output exactly: "Insufficient evidence."

### GATE 5: OPTION GENERATION & DECISION TRACEABILITY
Every Option and Final Decision must pass the following rigorous tests before output:
1.  **The Traceability Test:** Does this option explicitly link to a Supporting Theme, Root Cause, Observation, and piece of Evidence? (If NO: Reject).
2.  **The Generic Recommendation Detector:** Could this recommendation apply to a completely different company/situation without modification? (If YES: Reject).
3.  **The Contradiction Test:** Is there any evidence in the dataset that contradicts this decision? (If YES: Note it explicitly and downgrade confidence).

---

## CONFIDENCE & KNOWLEDGE RULES
1.  **ZERO External Knowledge:** All reasoning must originate from the provided information. Do not fill gaps with your own training data.
2.  **Confidence Scoring:** Confidence must be calculated based strictly on: Evidence Quality, Evidence Quantity, Evidence Consistency, and Traceability Completeness. It must NEVER be based on writing style or linguistic certainty.

==================================================
Evidence Sufficiency Input: {'SUFFICIENT' if evidence_sufficient else 'INSUFFICIENT_EVIDENCE'} ({sufficiency_message})
Verified Evidence:
{fact_text}

Strategic Signals:
{signal_text}

OUTPUT STRICT JSON MATCHING THE PROVIDED PYDANTIC SCHEMA.
"""
    schema_hint = """
{
  "status": "SUFFICIENT or INSUFFICIENT_EVIDENCE",
  "reason": "...",
  "evidence_and_observation_log": [{"evidence": "...", "observation": "...", "root_cause_and_class": "..."}],
  "strategic_themes_and_signals": [{"name": "...", "type": "STRATEGIC THEME or EMERGING SIGNAL", "traceability": ["..."]}],
  "competitive_landscape": [{"competitor": "...", "advantage": "...", "advantage_evidence": "...", "vulnerability": "...", "vulnerability_evidence": "..."}],
  "evaluated_options": [{"description": "...", "traceability_chain": "...", "generic_test_passed": "Yes/No"}],
  "recommended_decision": "...",
  "contradicting_evidence": "...",
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

if st.button("Run Evidence-Based Reasoning", type="primary"):
    if not company:
        st.error("Please enter an entity name.")
    else:
        with st.status(f"Executing Evidence Pipeline for {company}...", expanded=True) as status:

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

            st.write("⚖️ Reasoning Engine — Enforcing Traceability Chain...")
            final_brief = run_expert_reasoner(
                company, entity, verified_facts, signals,
                evidence_sufficient, sufficiency_message
            )

            status.update(label="Analysis Complete", state="complete")

        if not final_brief:
            st.error("Reasoning Engine failed to produce valid JSON. Try again.")
            st.stop()

        # ==========================================
        # DISPLAY: EVIDENCE-BASED REASONING LAYER
        # ==========================================
        st.divider()
        st.header(f"Decision Validation Brief — {entity.canonical_name.upper()}")
        st.caption(f"**Entity Context:** {entity.industry} | {entity.sector} | {entity.primary_market}")

        if final_brief.status == "INSUFFICIENT_EVIDENCE":
            st.error(f"🛑 **DATA SUFFICIENCY GATE FAILED**")
            st.warning(f"**Reason:** {final_brief.reason or 'Insufficient evidence provided.'}")
            st.info("Reliable conclusions cannot be generated from the available evidence. Strategy generation aborted.")
            st.stop()
        else:
            st.success(f"✅ **DATA SUFFICIENCY GATE PASSED**\n{final_brief.reason or 'Evidence meets threshold.'}")

        # 1. EVIDENCE & OBSERVATION LOG
        st.markdown("### 1. Evidence & Observation Log")
        for log in final_brief.evidence_and_observation_log:
            with st.container(border=True):
                st.markdown(f"**Evidence:** `{log.evidence or 'N/A'}`")
                st.info(f"**Observation:** {log.observation or 'N/A'}")
                st.warning(f"**Root Cause & Class:** {log.root_cause_and_class or 'N/A'}")

        # 2. STRATEGIC THEMES & SIGNALS
        st.markdown("### 2. Strategic Themes & Signals")
        c1, c2 = st.columns(2)
        for i, ts in enumerate(final_brief.strategic_themes_and_signals):
            col = c1 if i % 2 == 0 else c2
            with col.container(border=True):
                st.subheader(ts.name or "Unnamed Theme")
                type_val = ts.type or "UNKNOWN"
                type_color = "green" if "THEME" in type_val else "orange"
                st.markdown(f"**Type:** :{type_color}[{type_val}]")
                st.markdown("**Traceability:**")
                for trace in ts.traceability:
                    st.markdown(f"- {trace}")

        # 3. COMPETITIVE LANDSCAPE
        st.markdown("### 3. Competitive Landscape (Strict)")
        for comp in final_brief.competitive_landscape:
            with st.container(border=True):
                st.markdown(f"**Competitor:** {comp.competitor or 'N/A'}")
                c_adv, c_vuln = st.columns(2)
                with c_adv:
                    st.success(f"**Advantage:** {comp.advantage or 'None explicitly supported'}")
                    st.caption(f"**Evidence:** {comp.advantage_evidence or 'N/A'}")
                with c_vuln:
                    st.error(f"**Vulnerability:** {comp.vulnerability or 'None explicitly supported'}")
                    st.caption(f"**Evidence:** {comp.vulnerability_evidence or 'N/A'}")

        # 4. EVALUATED OPTIONS
        st.markdown("### 4. Evaluated Options")
        for opt in final_brief.evaluated_options:
            with st.container(border=True):
                st.markdown(f"**Option:** {opt.description or 'N/A'}")
                
                # Handle chain rendering safely whether it's a list or a string
                chain = opt.traceability_chain
                if isinstance(chain, list):
                    chain = "\n-> ".join(chain)
                st.info(f"**Traceability Chain:**\n{chain or 'N/A'}")
                
                passed_test = opt.generic_test_passed or ""
                if "Yes" in passed_test:
                    st.success("✅ **Passed Generic Test:** Uniquely applies to this situation.")
                elif "No" in passed_test:
                    st.error("❌ **Failed Generic Test:** Recommendation is too generic.")
                else:
                    st.warning("⚠️ **Generic Test:** Status Unknown")

        # 5. FINAL DECISION & INTEGRITY CHECK
        st.markdown("### 5. Final Decision & Integrity Check")
        with st.container(border=True):
            st.subheader("Recommended Decision")
            
            if final_brief.recommended_decision:
                st.success(final_brief.recommended_decision)
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
            confidence = final_brief.confidence_assessment or "N/A"
            st.markdown(f"`{confidence}`")

        # Export
        st.divider()
        export = {
            "entity_profile": entity.model_dump(),
            "verified_facts": [vf.model_dump() for vf in verified_facts],
            "reasoning_brief": final_brief.model_dump(),
        }
        st.download_button(
            "Download Evidence-Based Reasoning Package (JSON)",
            data=json.dumps(export, indent=2),
            file_name=f"{company.replace(' ','_')}_evidence_reasoning.json",
            mime="application/json"
        )