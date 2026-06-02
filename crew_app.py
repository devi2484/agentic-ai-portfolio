import os
import streamlit as st
from langchain_groq import ChatGroq
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from typing import List, Any
from urllib.parse import urlparse

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY") or st.secrets.get("GROQ_KEY", "")

# 70b model for strict JSON adherence and deep analytical reasoning
llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.3-70b-versatile", temperature=0.1)

st.set_page_config(page_title="Business Intelligence Engine", page_icon="⚡", layout="wide")
st.title("⚡ Business Intelligence Engine (v2)")
st.markdown("**Consulting-Grade Pipeline** · Competitor Benchmarking · Implication Layer · Action Prioritization")
st.divider()

# ==========================================
# 2. TRUST SCORING & SEARCH
# ==========================================
HIGH_TRUST = ["reuters.com", "bloomberg.com", "cnbc.com", "wsj.com", "ft.com", "sec.gov", "techcrunch.com", "forbes.com"]
LOW_TRUST = ["linkedin.com", "reddit.com", "quora.com", "wikipedia.org", "medium.com"]

def evaluate_trust(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if any(ht in domain for ht in HIGH_TRUST):
        return "HIGH TRUST"
    if any(lt in domain for lt in LOW_TRUST):
        return "LOW TRUST"
    return "MEDIUM TRUST"

def run_enhanced_search(company: str) -> str:
    queries = [
        f"{company} unit economics profit margins 2025",
        f"{company} vs top competitors market share 2025",
        f"{company} supply chain retail strategy vulnerabilities",
        f"{company} capital allocation strategic pivot"
    ]
    
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=2, timelimit="y"):
                    url = r.get('href', '')
                    trust = evaluate_trust(url)
                    results.append(
                        f"SOURCE: {url}\n"
                        f"TRUST RATING: {trust}\n"
                        f"CONTENT: {r.get('title', '')} - {r.get('body', '')}\n"
                        f"{'-'*40}"
                    )
    except Exception as e:
        st.error(f"Search API Error: {e}")
    
    return "\n".join(results)

# ==========================================
# 3. HELPER MATH (PROGRAMMATIC CONFIDENCE)
# ==========================================
def coerce_to_int(v: Any) -> int:
    if isinstance(v, str):
        try:
            cleaned = ''.join(c for c in v if c.isdigit())
            return int(cleaned) if cleaned else 0
        except Exception:
            return 0
    elif isinstance(v, (int, float)):
        return int(v)
    return 0

def calculate_programmatic_confidence(verified_facts: list, total_extracted: int) -> int:
    """Calculates confidence algorithmically based on data volume and audit scores."""
    if not verified_facts:
        return 0
    
    # 1. Validation Score (Max 50 points) - Average of the challenger audit scores
    avg_audit = sum(f.confidence_score for f in verified_facts) / len(verified_facts)
    validation_score = (avg_audit / 100) * 50
    
    # 2. Fact Count (Max 30 points) - Rewards pipelines that found lots of data (5 points per fact)
    fact_count_score = min(len(verified_facts) * 5, 30)
    
    # 3. Source Trust Yield (Max 20 points) - How many facts survived the audit vs were extracted?
    yield_ratio = len(verified_facts) / max(total_extracted, 1)
    source_trust_score = yield_ratio * 20
    
    return int(validation_score + fact_count_score + source_trust_score)


# ==========================================
# 4. PYDANTIC SCHEMAS
# ==========================================
class IntelligenceFact(BaseModel):
    category: str
    fact: str
    competitor_context: str
    relevance_score: int
    source_trust: str

    @field_validator('relevance_score', mode='before')
    @classmethod
    def fix_relevance(cls, v): return coerce_to_int(v)

class ResearchReport(BaseModel):
    company: str
    facts: List[IntelligenceFact]

class FactCheckResult(BaseModel):
    original_fact: str
    is_verified: bool
    confidence_score: int
    reasoning: str

    @field_validator('confidence_score', mode='before')
    @classmethod
    def fix_confidence(cls, v): return coerce_to_int(v)

class ChallengerReport(BaseModel):
    verifications: List[FactCheckResult]

class CompetitorBenchmark(BaseModel):
    competitor: str = Field(description="Name of the specific primary competitor.")
    company_position: str = Field(description="Leader, Challenger, Laggard, or Niche.")
    advantage: str = Field(description="Specific, structural advantage the target company holds.")
    threat: str = Field(description="Specific threat this competitor poses to the target company.")

class StrategicAction(BaseModel):
    framework: str = Field(description="STOP, START, DOUBLE DOWN")
    evidence: str = Field(description="Hard metrics only. No vanity metrics like 'largest market cap'.")
    implication: str = Field(description="Why this fact changes the strategic landscape.")
    competitor_context: str = Field(description="How this specific action defends against or attacks the named competitor.")
    action: str = Field(description="Hyper-specific operational directive.")
    expected_impact: str = Field(description="Quantifiable business impact.")
    risk: str = Field(description="Primary execution risk or expected competitor counter-move.")
    timeline: str

class CEOBrief(BaseModel):
    narrative_what_changed: str
    narrative_why_now: str
    narrative_primary_move: str
    competitor_benchmarks: List[CompetitorBenchmark]
    prioritized_actions: List[StrategicAction]

# ==========================================
# 5. CORE AGENTS
# ==========================================
def run_researcher(company: str, raw_search_context: str) -> ResearchReport:
    structured_llm = llm.with_structured_output(ResearchReport)
    prompt = f"""
    You are a Research Analyst extracting precise intelligence for {company}.
    RULES: Find exact numbers for unit economics and market share. Benchmark against a specific competitor whenever possible.
    Context: {raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_challenger(facts: List[IntelligenceFact], raw_search_context: str) -> ChallengerReport:
    structured_llm = llm.with_structured_output(ChallengerReport)
    high_relevance_facts = [f for f in facts if f.relevance_score >= 7]
    fact_strings = "\n".join([f"[{f.category}] (Relevance {f.relevance_score}/10) | {f.source_trust}: {f.fact} | Benchmark: {f.competitor_context}" for f in high_relevance_facts])
    
    prompt = f"""
    You are a BCG compliance auditor. Verify these facts against the raw context.
    Assign a confidence score (0-100). Penalize heavily for LOW TRUST sources or vague metrics like "market cap".
    Facts: {fact_strings}
    Context: {raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_strategist(company: str, verified_facts: List[FactCheckResult]) -> CEOBrief:
    structured_llm = llm.with_structured_output(CEOBrief)
    valid_data = [f.original_fact for f in verified_facts if f.is_verified and f.confidence_score >= 70]
    
    if not valid_data:
        valid_data = [f"Intelligence failure. No verified data found for {company}. Freeze capital allocation."]
    
    prompt = f"""
    You are a Strategy Partner advising the CEO of {company}. 
    Based ONLY on the verified evidence, draft a board-level strategic brief.
    
    CRITICAL RULES:
    1. HYPER-SPECIFICITY: Actions must be executable at the operational level.
       BAD: "Launch targeted marketing campaigns" or "Leverage AI."
       GOOD: "Bundle Powerwall with residential solar deployments in Germany and Netherlands."
    2. NO MBA FLUFF: Ban the words 'synergy', 'leverage', 'optimize', and 'holistic'.
    3. STRICT EVIDENCE: Do not use vanity metrics (e.g. 'largest automaker by market cap'). Evidence MUST be unit economics, margins, or explicit market share shifts.
    4. COMPETITIVE CONTEXT: You must explicitly state why {company} is winning/losing against a specific competitor, and frame actions around that threat.
    
    Verified Evidence:
    {valid_data}
    """
    return structured_llm.invoke(prompt)

# ==========================================
# 6. STREAMLIT UI
# ==========================================
company = st.text_input("Target Company:", placeholder="e.g. Zomato, Reliance, Tesla, Nykaa...")

if st.button("Run Strategic Analysis", type="primary"):
    if not company:
        st.error("Please enter a company name.")
    else:
        with st.status(f"Compiling Enterprise Intelligence on {company}...", expanded=True) as status:
            st.write("📡 Executing competitive benchmarking search...")
            raw_context = run_enhanced_search(company)
            
            if not raw_context:
                st.error("Search API failed to return data.")
                st.stop()
                
            st.write("📊 Extracting unit economics and market share data...")
            research_data = run_researcher(company, raw_context)
            total_extracted = len(research_data.facts)
            
            st.write("⚖️ Auditing evidence and source trust...")
            audit_data = run_challenger(research_data.facts, raw_context)
            verified_facts = [f for f in audit_data.verifications if f.is_verified and f.confidence_score >= 70]
            
            st.write("📋 Synthesizing implication layer and strategic actions...")
            final_brief = run_strategist(company, audit_data.verifications)
            
            # CALCULATE PROGRAMMATIC CONFIDENCE
            prog_confidence = calculate_programmatic_confidence(verified_facts, total_extracted)
            
            status.update(label="Analysis Complete", state="complete")

        # --- UI DISPLAY ---
        
        # 1. Fact Pipeline Transparency
        st.subheader("🛡️ Intelligence Pipeline & Verification Logs")
        st.caption(f"Raw Facts Extracted: {total_extracted} → Verified & High Trust: {len(verified_facts)}")
        
        with st.expander("View Underlying Data & Challenger Audit"):
            for fc in audit_data.verifications:
                if fc.is_verified and fc.confidence_score >= 70:
                    st.success(f"**[{fc.confidence_score}% Trust]** {fc.original_fact}")
                else:
                    st.error(f"**Rejected/Low Confidence ({fc.confidence_score}%):** {fc.original_fact} \n*Reason: {fc.reasoning}*")

        # 2. Executive Brief Header
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header(f"Board-Level Strategic Brief — {company.upper()}")
        with col2:
            st.metric(label="Calculated Report Confidence", value=f"{prog_confidence}/100", 
                      help="Computed programmatically via source trust + fact volume + audit validation score.")
            
        # 3. Competitor Benchmarking Block
        st.markdown("### Competitor Benchmarking")
        for bench in final_brief.competitor_benchmarks:
            with st.container(border=True):
                st.markdown(f"**Vs. {bench.competitor}** — Target Position: `{bench.company_position}`")
                st.markdown(f"🟢 **Advantage:** {bench.advantage}")
                st.markdown(f"🔴 **Threat:** {bench.threat}")

        # 4. The Strategic Narrative
        st.markdown("### The Strategic Narrative")
        with st.container(border=True):
            st.markdown(f"**📉 What Changed:** {final_brief.narrative_what_changed}")
            st.markdown(f"**⏳ Why Now (Catalyst):** {final_brief.narrative_why_now}")
            st.markdown(f"**🎯 The Primary Move:** {final_brief.narrative_primary_move}")
        
        # 5. Implication & Action Matrix
        st.markdown("### Prioritized Strategic Directives")
        for action in final_brief.prioritized_actions:
            color = "🔴" if action.framework == "STOP" else "🟢" if action.framework == "START" else "🔥"
            
            with st.container(border=True):
                st.markdown(f"#### {color} **{action.framework}**: {action.action}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**1. The Evidence (No Vanity Metrics)**")
                    st.info(f"*{action.evidence}*")
                    st.markdown("**2. The Implication**")
                    st.warning(action.implication)
                with c2:
                    st.markdown("**3. Competitor Context**")
                    st.write(action.competitor_context)
                    st.markdown("**4. Impact & Risk Matrix**")
                    st.success(f"**Impact:** {action.expected_impact}")
                    st.error(f"**Risk:** {action.risk}")

        # 6. Export
        st.divider()
        st.download_button(
            "Download Strategy JSON", 
            data=final_brief.model_dump_json(indent=2), 
            file_name=f"{company}_board_brief.json", 
            mime="application/json"
        )