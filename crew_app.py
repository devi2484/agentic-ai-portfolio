import os
import streamlit as st
from langchain_groq import ChatGroq
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from urllib.parse import urlparse

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY") or st.secrets.get("GROQ_KEY", "")

# 70b model for strict JSON adherence and deep analytical reasoning
llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.3-70b-versatile", temperature=0.1)

st.set_page_config(page_title="AI Intelligence Engine", page_icon="♟️", layout="wide")
st.title("♟️ Strategic Intelligence Engine")
st.markdown("**Consulting-Grade Pipeline** · Deterministic Scoring · Competitor Benchmarking · Implication Layer")
st.divider()

# ==========================================
# 2. TRUST SCORING, SEARCH & MATH
# ==========================================
HIGH_TRUST = ["reuters.com", "bloomberg.com", "cnbc.com", "wsj.com", "ft.com", "sec.gov", "techcrunch.com", "forbes.com"]
LOW_TRUST = ["linkedin.com", "reddit.com", "quora.com", "wikipedia.org", "medium.com"]

def evaluate_trust(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if any(ht in domain for ht in HIGH_TRUST):
        return "HIGH TRUST (Tier 1 Publisher/Primary Source)"
    if any(lt in domain for lt in LOW_TRUST):
        return "LOW TRUST (UGC/Social Media - Verify Carefully)"
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

def calculate_deterministic_confidence(raw_facts: list, verified_facts: list) -> int:
    """Calculates confidence programmatically rather than letting the LLM guess."""
    if not raw_facts:
        return 0
    
    # Metrics
    total_extracted = len(raw_facts)
    total_verified = sum(1 for f in verified_facts if f.is_verified)
    high_trust_count = sum(1 for f in raw_facts if "HIGH TRUST" in f.source_trust)
    
    # Weighting
    verification_rate = (total_verified / total_extracted) * 100 if total_extracted > 0 else 0
    trust_ratio = (high_trust_count / total_extracted) * 100 if total_extracted > 0 else 0
    volume_bonus = min(20, total_verified * 2) # Reward having more verified facts (up to 20 points)
    
    # Base score heavily favors verified facts from high trust sources
    score = (verification_rate * 0.5) + (trust_ratio * 0.3) + volume_bonus
    return min(99, int(score))

# ==========================================
# 3. PYDANTIC SCHEMAS (BULLETPROOF)
# ==========================================
class IntelligenceFact(BaseModel):
    category: str = Field(description="Must be exactly one of: Revenue, Profit, Competitive Benchmark, Product, Strategic, Capital Allocation")
    fact: str = Field(description="Specific, verifiable fact. Must include numbers/dates. Ignore older than 18 months.")
    competitor_context: str = Field(description="How this fact compares to a direct competitor. If no competitor data is present, output 'N/A'.")
    url: str = Field(description="The source URL")
    relevance_score: int = Field(description="1-10 score. 1=Trivia, 10=Board-level strategic importance.")
    source_trust: str = Field(description="Trust level provided in the raw context.")

class ResearchReport(BaseModel):
    company: str
    facts: List[IntelligenceFact]

class FactCheckResult(BaseModel):
    original_fact: str
    is_verified: bool = Field(description="True ONLY if explicitly backed by the raw search text.")
    reasoning: str = Field(description="Why it passed/failed verification.")

class ChallengerReport(BaseModel):
    verifications: List[FactCheckResult]

class CompetitorAnalysis(BaseModel):
    competitor: str = Field(description="Name of the specific primary competitor")
    target_company_position: str = Field(description="Leader, Challenger, Laggard, or Niche")
    advantage: str = Field(description="The target company's specific advantage over this competitor")
    threat: str = Field(description="The specific existential or margin threat this competitor poses")
    recommended_response: str = Field(description="Specific operational response to neutralize this threat")

class StrategicAction(BaseModel):
    framework: str = Field(description="Must be exactly one of: STOP, START, DOUBLE DOWN")
    evidence: str = Field(description="The specific verified fact driving this recommendation.")
    implication: str = Field(description="The 'So What?'. Why this fact fundamentally changes the strategic landscape.")
    competitor_context: str = Field(description="How this action specifically counters or outmaneuvers a named competitor.")
    action: str = Field(description="Hyper-specific operational directive. Generic advice like 'launch marketing' or 'review costs' is forbidden.")
    expected_impact: str = Field(description="Quantifiable business impact.")
    risk: str = Field(description="Primary execution risk or black swan vulnerability.")

class CEOBrief(BaseModel):
    narrative_what_changed: str = Field(description="What fundamental shift occurred in their market or unit economics recently?")
    narrative_why_now: str = Field(description="Why is immediate action required? What is the catalyst?")
    narrative_primary_move: str = Field(description="What is the single most important strategic pivot management must execute?")
    competitor_benchmarks: List[CompetitorAnalysis] = Field(description="Analysis of top 2 primary competitors.")
    prioritized_actions: List[StrategicAction] = Field(description="Top prioritized actions ranked by ROI/Impact.")

# ==========================================
# 4. CORE AGENTS
# ==========================================
def run_researcher(company: str, raw_search_context: str) -> ResearchReport:
    structured_llm = llm.with_structured_output(ResearchReport)
    prompt = f"""
    You are a Research Analyst. Extract precise intelligence for {company}.
    RULES: Find exact numbers for unit economics, margins, and market share. Benchmark against specific competitors. 
    Score relevance strictly. Ignore anything older than 18 months.
    Context: {raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_challenger(facts: List[IntelligenceFact], raw_search_context: str) -> ChallengerReport:
    structured_llm = llm.with_structured_output(ChallengerReport)
    high_relevance_facts = [f for f in facts if f.relevance_score >= 7]
    fact_strings = "\n".join([f"[{f.category}] (Relevance {f.relevance_score}/10) | {f.source_trust}: {f.fact} | Benchmark: {f.competitor_context}" for f in high_relevance_facts])
    prompt = f"""
    You are a BCG compliance auditor. Verify these high-relevance facts against the raw context.
    Penalize heavily if the source is LOW TRUST or the number is an estimation not backed by data.
    Facts to check: \n{fact_strings}\n\nRaw Context: \n{raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_strategist(company: str, verified_facts: List[FactCheckResult]) -> CEOBrief:
    structured_llm = llm.with_structured_output(CEOBrief)
    valid_data = [f.original_fact for f in verified_facts if f.is_verified]
    
    if not valid_data:
        valid_data = [f"Intelligence failure. No verified data found for {company}. Recommend capital freeze until primary data is sourced."]
    
    prompt = f"""
    You are a Strategy Partner advising the CEO of {company}. Based ONLY on the verified evidence below, draft a strategic brief.
    
    CRITICAL RULES:
    1. IMPLICATION & COMPETITOR LAYER: Every action MUST connect evidence to an implication, and MUST explicitly explain how it outmaneuvers a named competitor.
    2. ANTI-BOILERPLATE: "Launch marketing campaign", "Review costs", or "Improve product" will result in immediate termination. You must name specific geographies, product bundles, supply chain nodes, or margin targets.
    3. Benchmark against their top 2 most dangerous competitors.
    
    Verified Evidence:
    {valid_data}
    """
    return structured_llm.invoke(prompt)

# ==========================================
# 5. STREAMLIT UI
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
            
            st.write("⚖️ Auditing evidence and source trust (Anti-Hallucination)...")
            audit_data = run_challenger(research_data.facts, raw_context)
            
            st.write("📈 Calculating deterministic report confidence...")
            # Calculate confidence programmatically in Python
            calculated_confidence = calculate_deterministic_confidence(research_data.facts, audit_data.verifications)
            
            st.write("📋 Synthesizing implication layer and strategic actions...")
            final_brief = run_strategist(company, audit_data.verifications)
            
            status.update(label="Analysis Complete", state="complete")

        # --- UI DISPLAY ---
        
        # 1. Fact Pipeline Transparency
        st.subheader("🛡️ Intelligence Pipeline & Verification Logs")
        with st.expander("View Underlying Data & Challenger Audit"):
            for fc in audit_data.verifications:
                if fc.is_verified:
                    st.success(f"**✅ Verified:** {fc.original_fact}")
                else:
                    st.error(f"**❌ Rejected:** {fc.original_fact} \n*Reason: {fc.reasoning}*")

        # 2. Executive Brief
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header(f"Board-Level Strategic Brief — {company.upper()}")
        with col2:
            st.metric(label="Calculated Data Confidence", value=f"{calculated_confidence}%", 
                      help="Computed programmatically based on High-Trust source density and Challenger verification rates.")
            
        # 3. The Strategic Narrative
        st.markdown("### The Strategic Narrative")
        with st.container(border=True):
            st.markdown(f"**📉 What Changed:** {final_brief.narrative_what_changed}")
            st.markdown(f"**⏳ Why Now (Catalyst):** {final_brief.narrative_why_now}")
            st.markdown(f"**🎯 The Primary Move:** {final_brief.narrative_primary_move}")

        # 4. Competitor Benchmarking Block
        st.markdown("### ⚔️ Competitive Benchmarking")
        c1, c2 = st.columns(2)
        for i, comp in enumerate(final_brief.competitor_benchmarks[:2]):
            col = c1 if i % 2 == 0 else c2
            with col:
                with st.container(border=True):
                    st.markdown(f"#### vs. {comp.competitor}")
                    st.caption(f"Position: **{comp.target_company_position}**")
                    st.markdown(f"**🛡️ Advantage:** {comp.advantage}")
                    st.markdown(f"**⚠️ Threat:** {comp.threat}")
                    st.markdown(f"**⚡ Counter-Move:** {comp.recommended_response}")
        
        # 5. Implication & Action Matrix
        st.markdown("### 🎯 Prioritized Strategic Directives")
        for action in final_brief.prioritized_actions:
            color = "🔴" if action.framework == "STOP" else "🟢" if action.framework == "START" else "🔥"
            
            with st.container(border=True):
                st.markdown(f"#### {color} **{action.framework}**: {action.action}")
                
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**1. The Evidence**")
                    st.info(f"*{action.evidence}*")
                    st.markdown("**2. The Implication (So What?)**")
                    st.warning(action.implication)
                with colB:
                    st.markdown("**3. Competitor Context**")
                    st.error(action.competitor_context)
                    st.markdown("**4. Impact & Risk**")
                    st.success(f"**Impact:** {action.expected_impact}")
                    st.caption(f"**Risk:** {action.risk}")

        # 6. Export
        st.divider()
        st.download_button(
            "Download Strategy JSON", 
            data=final_brief.model_dump_json(indent=2), 
            file_name=f"{company}_board_brief.json", 
            mime="application/json"
        )