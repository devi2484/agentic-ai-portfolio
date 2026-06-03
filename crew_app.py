import os
import json
import time
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
    "hbr.org", "mckinsey.com", "bain.com", "bcg.com", "economist.com", 
    "statista.com", "nyse.com", "nasdaq.com"
]
MEDIUM_TRUST_DOMAINS = [
    "techcrunch.com","forbes.com","inc42.com","entrackr.com",
    "yourstory.com","themorningcontext.com","restofworld.org","fortune.com",
    "nytimes.com", "theguardian.com", "bbc.co.uk", "bbc.com", "cnn.com"
]
LOW_TRUST_DOMAINS = [
    "linkedin.com","reddit.com","quora.com","wikipedia.org",
    "medium.com","twitter.com","x.com","substack.com",
]
TRUST_SCORE_MAP = {"HIGH TRUST": 10, "MEDIUM TRUST": 6, "LOW TRUST": 2}

MIN_VERIFIED_FACTS   = 2
MIN_REPORT_CONFIDENCE = 50
ENTITY_CONFIDENCE_THRESHOLD = 60

def evaluate_trust(url: str, company: str = "") -> str:
    domain = urlparse(url).netloc.lower().replace("www.", "")
    # Self-disclosures for private companies are inherently high-trust primary evidence
    if company and company.lower().replace(" ", "") in domain.replace("-", ""):
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
    current_year = datetime.now().year
    queries = [
        f"{company} corporate profile industry sector business model structure",
        f"{company} revenue profit margin earnings {current_year}",
        f"{company} market share competitor performance {current_year}",
        f"{company} capital allocation investment factory expansion {current_year}",
        f"{company} regulatory risk sustainability supply chain {current_year}",
        f"{company} strategic transformation product lineup roadmap {current_year}",
    ]
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                # Removed hard timelimit constraint to allow deep profile captures
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
# 6. PYDANTIC MODELS (UPGRADED INFERENCE & OPTIONS)
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

class EvidenceLog(BaseModel):
    evidence: Optional[str] = None
    observation: Optional[str] = None
    root_cause: Optional[str] = Field(default=None, description="Must explain observation or be UNKNOWN. Never restates observation.")
    inference: Optional[str] = Field(default=None, description="Format: [Inference] | [CONFIRMED/LIKELY/HYPOTHESIS]")

class ThemeSignal(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = Field(default=None, description="Must be STRATEGIC THEME or EMERGING SIGNAL")
    traceability: List[str] = Field(default_factory=list, description="Min 2 observations or 3 facts required")

class CompetitiveLandscape(BaseModel):
    competitor: Optional[str] = None
    advantage: Optional[str] = None
    advantage_evidence: Optional[str] = None
    vulnerability: Optional[str] = None
    vulnerability_evidence: Optional[str] = None

class EvaluatedOption(BaseModel):
    option_type: Optional[str] = Field(default=None, description="Must be Conservative, Balanced, or Aggressive")
    description: Optional[str] = None
    traceability_chain: Union[str, List[str], None] = Field(default=None, description="Theme [X] -> Inference [Y] -> Observation [Z]")
    evidence_support: Optional[str] = None
    risk: Optional[str] = None
    complexity: Optional[str] = None
    strategic_fit: Optional[str] = None
    generic_test_passed: Optional[str] = Field(default=None, description="Yes or No")

class DecisionIntelligenceBrief(BaseModel):
    status: str = Field(description="Must be exactly 'SUFFICIENT' or 'INSUFFICIENT_EVIDENCE'")
    reason: Optional[str] = Field(default=None)
    
    evidence_and_observation_log: List[EvidenceLog] = Field(default_factory=list)
    strategic_themes_and_signals: List[ThemeSignal] = Field(default_factory=list)
    competitive_landscape: List[CompetitiveLandscape] = Field(default_factory=list)
    evaluated_options: List[EvaluatedOption] = Field(default_factory=list)
    
    recommended_decision: Optional[str] = Field(default=None, description="Must reference 1 Observation, 1 Inference, 1 Theme, 1 Option.")
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
    prompt = f"""You are a Fact Extraction System. Extract highly specific, verifiable operational and financial metrics for {entity.canonical_name}.
Extract at least 4-6 distinct structural facts from the raw context text if available.

CRITICAL INSTRUCTION: For every item you construct, match the source tracking indicators exactly. Copy the tracking link into "source_url" and the structural trust level token ("HIGH TRUST", "MEDIUM TRUST", or "LOW TRUST") into "source_trust".

Return a JSON object:
{{
  "facts": [
    {{
      "category": "one of: {', '.join(FACT_CATEGORIES)}",
      "fact": "verifiable claim containing specific figures, percentages, geographic transformations or product milestones",
      "source_url": "The exact absolute tracking URL identified directly above the context segment",
      "source_trust": "The tracking trust configuration token value",
      "date_signal": "Specific timeline tag. If the document references ongoing, structural or recent outcomes, explicitly record '{current_year}'",
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
                # Sanitize field anomalies where string tags get mixed into integer definitions
                if "board_relevance" in f and isinstance(f["board_relevance"], str):
                    f["board_relevance"] = int(''.join(filter(str.isdigit, f["board_relevance"])) or 9)
                if "strategic_impact" in f and isinstance(f["strategic_impact"], str):
                    f["strategic_impact"] = int(''.join(filter(str.isdigit, f["strategic_impact"])) or 9)
                facts.append(IntelligenceFact(**f))
            except Exception: 
                continue
        return facts
    except Exception as e:
        return []

def run_hard_gate_validation(facts: List[IntelligenceFact]) -> List[ValidatedFact]:
    verified = []
    for f in facts:
        # Adjusted alignment parameters to stop filtering deep structural updates
        if f.board_relevance < 7 or f.strategic_impact < 7: continue
        if "LOW TRUST" in f.source_trust.upper(): continue
        confidence = calculate_confidence(f.source_trust, f.board_relevance, f.strategic_impact)
        if confidence < 70: continue
        if f.date_signal == "Undated" and "HIGH TRUST" not in f.source_trust.upper(): continue
        verified.append(ValidatedFact(
            category=f.category, fact=f.fact, source_url=f.source_url, 
            source_trust=f.source_trust.upper(), date_signal=f.date_signal, 
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
You are a strict Evidence-Based Reasoning Engine. Your primary responsibility is to rigorously construct and verify reasoning chains derived EXCLUSIVELY from provided data. Maximize accuracy, traceability, and reliability.

## REQUIRED TRACEABILITY CHAIN
Every output MUST follow this exact traceability chain:
[Evidence] -> [Observation] -> [Root Cause] -> [Inference] -> [Theme] -> [Options] -> [Decision]

---

## EXECUTION PROTOCOL & VALIDATION GATES

### GATE 0: DATA SUFFICIENCY
If evidence is insufficient to make reliable decisions, you MUST set "status" to "INSUFFICIENT_EVIDENCE", populate the reason, and leave arrays empty.

### GATE 1: OBSERVATION RULE
Observations describe what happened. They never explain why. 

### GATE 2: ROOT CAUSE RULE
A root cause must explain an observation. If evidence does not explain the observation: Return UNKNOWN. Never restate the observation.

### GATE 3: INFERENCE LAYER (MOST IMPORTANT)
This is where reasoning happens. Connect the Observation/Root Cause to strategic meaning.
Inference can ONLY be classified as: CONFIRMED, LIKELY, or HYPOTHESIS.

### GATE 4: THEME RULE
Themes explain patterns across observations. A theme requires a MINIMUM of 2 observations OR 3 facts. Facts/Observations are not themes.

### GATE 5: OPTION SELECTION RULE
Always generate exactly 3 evaluated options:
1. Conservative Option
2. Balanced Option
3. Aggressive Option

### GATE 6: DECISION TRACEABILITY RULE
Every final recommendation must explicitly reference:
- 1 Observation
- 1 Inference
- 1 Theme
- 1 Option

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
  "evidence_and_observation_log": [
    {
      "evidence": "...", 
      "observation": "...", 
      "root_cause": "...", 
      "inference": "... | CONFIRMED/LIKELY/HYPOTHESIS"
    }
  ],
  "strategic_themes_and_signals": [{"name": "...", "type": "STRATEGIC THEME or EMERGING SIGNAL", "traceability": ["..."]}],
  "competitive_landscape": [{"competitor": "...", "advantage": "...", "advantage_evidence": "...", "vulnerability": "...", "vulnerability_evidence": "..."}],
  "evaluated_options": [
    {
      "option_type": "Conservative/Balanced/Aggressive",
      "description": "...",
      "traceability_chain": "...",
      "evidence_support": "...",
      "risk": "...",
      "complexity": "...",
      "strategic_fit": "...",
      "generic_test_passed": "Yes/No"
    }
  ],
  "recommended_decision": "Recommendation referencing 1 Obs, 1 Inf, 1 Theme, 1 Option...",
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
            # Expanded processing slice window to avoid string cutoffs
            raw_facts = run_researcher(company, entity, raw_context[:12000])

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

        # 1. EVIDENCE, OBSERVATION & INFERENCE LOG
        st.markdown("### 1. Evidence, Observation & Inference Log")
        for log in final_brief.evidence_and_observation_log:
            with st.container(border=True):
                st.markdown(f"**Evidence:** `{log.evidence or 'N/A'}`")
                st.info(f"**Observation:** {log.observation or 'N/A'}")
                c1, c2 = st.columns(2)
                with c1:
                    st.warning(f"**Root Cause:** {log.root_cause or 'UNKNOWN'}")
                with c2:
                    st.success(f"**Inference:** {log.inference or 'N/A'}")

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

        # 4. EVALUATED OPTIONS (WITH SCORES)
        st.markdown("### 4. Evaluated Options")
        for opt in final_brief.evaluated_options:
            with st.container(border=True):
                opt_type = opt.option_type or "Unknown"
                color = "blue" if "Conservative" in opt_type else "orange" if "Balanced" in opt_type else "red"
                st.markdown(f"**Option Type:** :{color}[{opt_type}]")
                st.markdown(f"**Description:** {opt.description or 'N/A'}")
                
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Evidence Support", opt.evidence_support or "N/A")
                sc2.metric("Risk", opt.risk or "N/A")
                sc3.metric("Complexity", opt.complexity or "N/A")
                sc4.metric("Strategic Fit", opt.strategic_fit or "N/A")

                chain = opt.traceability_chain
                if isinstance(chain, list):
                    chain = "\n-> ".join(chain)
                st.info(f"**Traceability Chain:**\n{chain or 'N/A'}")
                
                passed_test = opt.generic_test_passed or ""
                if "Yes" in passed_test:
                    st.success("✅ **Passed Generic Test:** Uniquely applies to this situation.")
                elif "No" in passed_test:
                    st.error("❌ **Failed Generic Test:** Recommendation is too generic.")

        # 5. FINAL DECISION & INTEGRITY CHECK
        st.markdown("### 5. Final Decision & Traceability Integrity")
        with st.container(border=True):
            st.subheader("Recommended Decision")
            if final_brief.recommended_decision:
                st.success(final_brief.recommended_decision)
                st.caption("*Integrity Check: Ensure the recommendation explicitly references 1 Observation, 1 Inference, 1 Theme, and 1 Option.*")
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