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
st.markdown("**Strict Evidence-Based Reasoning** · Mathematical Scoring · Deterministic Validation")
st.divider()

# ==========================================
# 2. TRUST & SCORING (SOURCES)
# ==========================================
HIGH_TRUST_DOMAINS = [
    "reuters.com","bloomberg.com","cnbc.com","wsj.com","ft.com","sec.gov",
    "marketwatch.com","barrons.com","morningstar.com","spglobal.com",
    "finance.yahoo.com","moodys.com","fitchratings.com","nytimes.com",
    "moneycontrol.com","economictimes.indiatimes.com","livemint.com",
    "businessstandard.com","thehindubusinessline.com","financialexpress.com",
    "bseindia.com","nseindia.com","sebi.gov.in","rbi.org.in",
]
MEDIUM_TRUST_DOMAINS = ["techcrunch.com","forbes.com","fortune.com", "seekingalpha.com"]
LOW_TRUST_DOMAINS = ["linkedin.com","reddit.com","quora.com","wikipedia.org","twitter.com","x.com"]
TRUST_SCORE_MAP = {"HIGH TRUST": 10, "MEDIUM TRUST": 6, "LOW TRUST": 2}

MIN_VERIFIED_FACTS = 2

def evaluate_trust(url: str) -> str:
    domain = urlparse(url).netloc.lower().replace("www.", "")
    if any(h in domain for h in HIGH_TRUST_DOMAINS):   return "HIGH TRUST"
    if any(m in domain for m in MEDIUM_TRUST_DOMAINS): return "MEDIUM TRUST"
    if any(l in domain for l in LOW_TRUST_DOMAINS):    return "LOW TRUST"
    return "MEDIUM TRUST"

def calculate_confidence(trust_label: str, board_relevance: int, strategic_impact: int) -> int:
    trust_score = TRUST_SCORE_MAP.get(trust_label.strip().upper(), 5)
    raw = (trust_score * 0.4) + (board_relevance * 0.3) + (strategic_impact * 0.3)
    return int((raw / 10) * 100)

def calculate_entity_confidence(entity) -> tuple[int, str]:
    score = 100
    reasons = []
    if entity.industry == "Unknown": score -= 20; reasons.append("industry unknown")
    if entity.contamination_warnings.lower() != "none detected" and entity.contamination_warnings != "": 
        score -= 20
        reasons.append("contamination risk")
    explanation = f"Entity confidence {score}%"
    if reasons: explanation += f" — Issues: {', '.join(reasons)}"
    return max(0, score), explanation

# ==========================================
# 3. RULE 7: DETERMINISTIC CONFIDENCE FORMULA
# ==========================================
def calculate_strict_confidence(brief, total_facts: int) -> str:
    if not brief or brief.status == "INSUFFICIENT_EVIDENCE":
        return "N/A - Insufficient Evidence"
        
    num_obs = len(brief.evidence_and_observation_log)
    num_themes = len(brief.strategic_themes_and_signals)
    num_comp = len(brief.competitive_landscape)
    
    if total_facts >= 3 and num_obs >= 2 and num_themes >= 1 and num_comp >= 1:
        return "HIGH (Met Strict Rule: 3+ facts, 2+ obs, 1+ theme, 1+ comp signal)"
    elif total_facts >= 2 and num_themes >= 1:
        return "MEDIUM (Met Strict Rule: 2+ facts, 1 theme)"
    else:
        return "LOW (Failed threshold tests for higher confidence)"

# ==========================================
# 4. SEARCH & JSON INVOKE (OPTIMIZED & BUG FIXED)
# ==========================================
def run_enhanced_search(company: str) -> str:
    queries = [
        f"{company} recent financial results revenue profit margin earnings",
        f"{company} market share competitive advantage vulnerabilities",
        f"{company} strategic pivot capital allocation recent acquisitions"
    ]
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                time.sleep(1) # Prevent silent rate-limiting from the search engine
                for r in ddgs.text(q, max_results=3, timelimit="y"):
                    url = r.get("href", "")
                    results.append(f"SOURCE: {url}\nTRUST: {evaluate_trust(url)}\nCONTENT: {r.get('title','')} — {r.get('body','')}\n{'-'*40}")
    except Exception as e:
        st.error(f"Search error: {e}")
    return "\n".join(results)

def invoke_json(prompt: str) -> dict:
    messages = [
        SystemMessage(content="You are a precise JSON-only responder. Output ONLY valid JSON. No markdown, no explanation."),
        HumanMessage(content=prompt)
    ]
    resp = llm.invoke(messages)
    text = resp.content.strip()
    
    # Safely strip markdown code blocks without causing syntax errors
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"): 
            text = text[4:]
            
    text = text.strip().rstrip("```").strip()
    return json.loads(text)

# ==========================================
# 5. PYDANTIC MODELS (UPGRADED FOR RULES 1-8)
# ==========================================
class EntityProfile(BaseModel):
    canonical_name: str
    industry: str
    business_model: str
    contamination_warnings: str

class ValidatedFact(BaseModel):
    fact: str
    source_url: str
    source_trust: str
    confidence: int

class EvidenceLog(BaseModel):
    evidence: Optional[str] = None
    observation: Optional[str] = Field(default=None, description="Direct restatement of evidence. NO REASONING.")
    inference: Optional[str] = Field(default=None, description="The ONLY place where reasoning occurs. Must be CONFIRMED, LIKELY, or HYPOTHESIS.")

class ThemeSignal(BaseModel):
    name: Optional[str] = Field(default=None, description="Must be abstract (e.g., Margin Pressure). Not 'Revenue Growth'.")
    traceability: List[str] = Field(default_factory=list, description="Must list 2+ observations AND 1+ inference.")

class CompetitiveLandscape(BaseModel):
    competitor: Optional[str] = None
    advantage: Optional[str] = Field(default=None, description="MUST explicitly come from evidence. No generic branding assumptions.")

class EvaluatedOption(BaseModel):
    option_type: Optional[str] = Field(default=None, description="Conservative, Balanced, or Aggressive")
    description: Optional[str] = None
    evidence_score: Optional[int] = Field(default=0, description="Score 0-10")
    strategic_fit_score: Optional[int] = Field(default=0, description="Score 0-10")
    risk_score: Optional[int] = Field(default=0, description="Score 0-10")
    total_score: Optional[int] = Field(default=0, description="Sum of the three scores.")
    traceability_chain: Union[str, List[str], None] = Field(default=None, description="Evidence -> Obs -> Inference -> Theme -> Option -> Decision")

class DecisionIntelligenceBrief(BaseModel):
    status: str = Field(description="Must be exactly 'SUFFICIENT' or 'INSUFFICIENT_EVIDENCE'")
    reason: Optional[str] = Field(default=None)
    evidence_and_observation_log: List[EvidenceLog] = Field(default_factory=list)
    strategic_themes_and_signals: List[ThemeSignal] = Field(default_factory=list)
    competitive_landscape: List[CompetitiveLandscape] = Field(default_factory=list)
    evaluated_options: List[EvaluatedOption] = Field(default_factory=list)
    recommended_decision: Optional[str] = Field(default=None, description="Must EXACTLY match the highest scoring option.")

# ==========================================
# 6. PIPELINE AGENTS
# ==========================================
def run_entity_resolution(company: str, raw_context: str) -> EntityProfile:
    prompt = f"""Identify exactly which company the search results describe. Return JSON: {{"canonical_name":"", "industry":"", "business_model":"", "contamination_warnings":""}}
    Query: {company}\nContext: {raw_context[:1500]}"""
    try: 
        return EntityProfile(**invoke_json(prompt))
    except: 
        return EntityProfile(canonical_name=company, industry="Unknown", business_model="Unknown", contamination_warnings="Failed")

def run_researcher(company: str, entity: EntityProfile, raw_context: str) -> List[ValidatedFact]:
    prompt = f"""Extract verifiable data for {entity.canonical_name}. Preserve SOURCE URL and TRUST labels EXACTLY.
    Return JSON: {{"facts": [ {{"category":"", "fact":"", "source_url":"", "source_trust":"", "date_signal":"", "board_relevance":9, "strategic_impact":9}} ]}}
    Context: {raw_context[:5000]}"""
    try:
        data = invoke_json(prompt)
        verified = []
        for f in data.get("facts", []):
            trust = f.get("source_trust", "").upper()
            if "LOW" in trust: continue
            conf = calculate_confidence(trust, f.get("board_relevance",5), f.get("strategic_impact",5))
            if conf >= 70:
                verified.append(ValidatedFact(fact=f.get("fact",""), source_url=f.get("source_url",""), source_trust=trust, confidence=conf))
        return verified
    except: 
        return []

# --- AGENT 3: EXPERT REASONING SYSTEM (RULES 1-8 ENFORCED) ---
def run_expert_reasoner(entity: EntityProfile, verified_facts: List[ValidatedFact]) -> Optional[DecisionIntelligenceBrief]:
    fact_text = "\n".join([f"- FACT: {f.fact} (Source: {f.source_trust})" for f in verified_facts]) if verified_facts else "INSUFFICIENT EVIDENCE."

    prompt = f"""# SYSTEM INSTRUCTIONS: STRICT EVIDENCE-BASED REASONING ENGINE

You are constrained by 8 absolute laws of reasoning. Read and obey exactly.

RULE 1: OBSERVATION RULE
Observation MUST be a direct restatement of evidence. NO reasoning. NO explanations.
- Good: "Revenue increased."
- Bad: "Revenue increased because demand improved."

RULE 2: INFERENCE LAYER (THE ONLY REASONING LAYER)
Inference is the ONLY place where reasoning occurs. Observations cannot infer. Themes cannot infer. Options cannot infer.
- Inference must be labeled: CONFIRMED, LIKELY, or HYPOTHESIS.

RULE 3: THEME FORMATION
A Theme requires a Minimum of: 2 observations AND 1 inference. 
Themes explain patterns. Facts/Observations (like "Revenue Growth") are NOT themes.

RULE 4: ABSTRACT THEME NAMES
Theme names must be abstract systemic patterns.
- Good: Margin Pressure, Portfolio Transformation, Capital Discipline.
- Bad: Revenue Growth, Profit Increase.

RULE 5: COMPETITIVE INTELLIGENCE
Must come STRICTLY from evidence. Do NOT write generic brand descriptions. If not in evidence, omit or leave blank.

RULE 6: DECISION SCORING
Always generate 3 Options (Conservative, Balanced, Aggressive).
For each, calculate: Evidence Score (0-10), Strategic Fit Score (0-10), Risk Score (0-10).
Calculate Total Score.
DECISION RULE: The Final Recommended Decision MUST be the option with the highest Total Score. It is not an LLM preference.

RULE 8: TRACEABILITY RULE
Every branch must mathematically trace: Evidence -> Observation -> Inference -> Theme -> Option -> Decision.

==================================================
Verified Evidence:
{fact_text}

OUTPUT STRICT JSON MATCHING THE PROVIDED PYDANTIC SCHEMA.
"""
    schema_hint = """
{
  "status": "SUFFICIENT or INSUFFICIENT_EVIDENCE",
  "reason": "...",
  "evidence_and_observation_log": [{"evidence": "...", "observation": "...", "inference": "... | CONFIRMED"}],
  "strategic_themes_and_signals": [{"name": "Abstract Name", "traceability": ["Obs 1", "Obs 2", "Inference 1"]}],
  "competitive_landscape": [{"competitor": "...", "advantage": "..."}],
  "evaluated_options": [
    {
      "option_type": "Conservative/Balanced/Aggressive",
      "description": "...",
      "evidence_score": 8,
      "strategic_fit_score": 7,
      "risk_score": 6,
      "total_score": 21,
      "traceability_chain": "..."
    }
  ],
  "recommended_decision": "MUST match the description of the highest scoring Evaluated Option."
}"""
    try: 
        return DecisionIntelligenceBrief(**invoke_json(prompt + schema_hint))
    except Exception as e: 
        st.error(f"Reasoning Engine error: {e}")
        return None

# ==========================================
# 7. STREAMLIT UI
# ==========================================
company = st.text_input("Target Company / Entity:", placeholder="e.g. Zomato, Reliance, Tesla...")

if st.button("Run Strict Evidence Reasoning", type="primary"):
    if not company: 
        st.error("Please enter an entity name.")
    else:
        with st.status(f"Executing Strict Pipeline for {company}...", expanded=True) as status:
            st.write("📡 Search & Fact Extraction...")
            raw_context = run_enhanced_search(company)
            entity = run_entity_resolution(company, raw_context)
            verified_facts = run_researcher(company, entity, raw_context)

            if len(verified_facts) < MIN_VERIFIED_FACTS:
                st.warning("⚠️ Insufficient evidence passed strict validation.")
                st.stop()
            
            st.write("⚖️ Reasoning Engine — Enforcing Rules 1-8...")
            final_brief = run_expert_reasoner(entity, verified_facts)
            status.update(label="Analysis Complete", state="complete")

        if not final_brief:
            st.error("Engine failed Traceability checks.")
            st.stop()

        # ==========================================
        # RENDER UI
        # ==========================================
        st.divider()
        st.header(f"Strict Traceability Brief — {entity.canonical_name.upper()}")
        
        # RULE 7 (CALCULATED IN PYTHON)
        deterministic_confidence = calculate_strict_confidence(final_brief, len(verified_facts))
        st.info(f"📐 **Rule 7 Deterministic Confidence:** {deterministic_confidence}")

        # RULES 1 & 2
        st.markdown("### 1. Evidence, Observation & Inference Layer")
        for log in final_brief.evidence_and_observation_log:
            with st.container(border=True):
                st.markdown(f"📖 **Evidence:** `{log.evidence}`")
                st.info(f"🔍 **Observation (Rule 1):** {log.observation}")
                st.success(f"🧠 **Inference (Rule 2):** {log.inference}")

        # RULES 3 & 4
        st.markdown("### 2. Abstract Strategic Themes (Rules 3 & 4)")
        for ts in final_brief.strategic_themes_and_signals:
            with st.container(border=True):
                st.subheader(ts.name or "Unnamed Theme")
                st.caption("Traceability (Requires 2+ Obs, 1+ Inference):")
                for trace in ts.traceability: 
                    st.markdown(f"- {trace}")

        # RULE 6 (SCORING MATRIX)
        st.markdown("### 3. Evaluated Options & Mathematical Scoring (Rule 6)")
        highest_score = -1
        winning_option = None
        
        for opt in final_brief.evaluated_options:
            with st.container(border=True):
                st.markdown(f"**Option:** {opt.description} *(Type: {opt.option_type})*")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Evidence Score", opt.evidence_score)
                c2.metric("Strategic Fit", opt.strategic_fit_score)
                c3.metric("Risk Score", opt.risk_score)
                c4.metric("Total Score", opt.total_score)
                
                # Check for winner
                if opt.total_score and opt.total_score > highest_score:
                    highest_score = opt.total_score
                    winning_option = opt.description

                chain = opt.traceability_chain
                if isinstance(chain, list): 
                    chain = "\n-> ".join(chain)
                st.caption(f"**Traceability:**\n{chain}")

        st.markdown("### 4. Final Recommended Decision (Rule 8)")
        if final_brief.recommended_decision:
            st.success(f"🏆 {final_brief.recommended_decision}")
            if winning_option and final_brief.recommended_decision.strip()[:20] != winning_option.strip()[:20]:
                st.warning("⚠️ **Rule 6 Alert:** The LLM's text description may not perfectly align with the mathematically highest scoring option.")
        else:
            st.error("Traceability failed: No decision reached.")
            
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