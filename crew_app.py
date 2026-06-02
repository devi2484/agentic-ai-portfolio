import os
import streamlit as st
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
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
        return "HIGH TRUST (Tier 1 Publisher/Primary Source)"
    if any(lt in domain for lt in LOW_TRUST):
        return "LOW TRUST (UGC/Social Media - Verify Carefully)"
    return "MEDIUM TRUST"

def run_enhanced_search(company: str) -> str:
    """Multi-query search returning URLs, dates, and trust scores. Specifically targeting benchmarks."""
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
# 3. PYDANTIC SCHEMAS (JSON VALIDATION)
# ==========================================
class IntelligenceFact(BaseModel):
    category: Literal["Revenue", "Profit", "Competitive Benchmark", "Product", "Strategic", "Capital Allocation"]
    fact: str = Field(description="Specific, verifiable fact. Must include numbers/dates. Ignore older than 18 months.")
    competitor_context: Optional[str] = Field(description="How this fact compares to a direct competitor (if available).")
    url: str = Field(description="The source URL")
    relevance_score: int = Field(description="1-10 score. 1=Trivia, 10=Board-level strategic importance.")
    source_trust: str = Field(description="Trust level provided in the raw context.")

class ResearchReport(BaseModel):
    company: str
    facts: List[IntelligenceFact]

class FactCheckResult(BaseModel):
    original_fact: str
    is_verified: bool = Field(description="True ONLY if explicitly backed by the raw search text.")
    confidence_score: int = Field(description="0-100 based on source trust and text clarity.")
    reasoning: str = Field(description="Why it passed/failed verification.")

class ChallengerReport(BaseModel):
    verifications: List[FactCheckResult]

class StrategicAction(BaseModel):
    framework: Literal["STOP", "START", "DOUBLE DOWN"]
    evidence: str = Field(description="The specific verified fact driving this recommendation.")
    implication: str = Field(description="The 'So What?'. Why this fact fundamentally changes the strategic landscape.")
    action: str = Field(description="Hyper-specific operational directive. Generic advice like 'review strategy' is forbidden. Name specific markets, product lines, or supply chain nodes.")
    expected_impact: str = Field(description="Quantifiable business impact.")
    timeline: str = Field(description="e.g., 90 Days, 6 Months, Q3.")
    confidence: int = Field(description="1-100 score based on the strength of the underlying evidence.")

class CEOBrief(BaseModel):
    report_confidence: int = Field(description="0-100 score based on overall data quality and verification.")
    narrative_what_changed: str = Field(description="What fundamental shift occurred in their market or unit economics recently?")
    narrative_why_now: str = Field(description="Why is immediate action required? What is the catalyst?")
    narrative_primary_move: str = Field(description="What is the single most important strategic pivot management must execute?")
    prioritized_actions: List[StrategicAction] = Field(description="Top prioritized actions ranked by ROI/Impact.")

# ==========================================
# 4. CORE AGENTS
# ==========================================
def run_researcher(company: str, raw_search_context: str) -> ResearchReport:
    """Extracts facts, categorizes them, and explicitly hunts for competitive benchmarks."""
    structured_llm = llm.with_structured_output(ResearchReport)
    prompt = f"""
    You are a Research Analyst. Extract precise intelligence for {company}.
    
    RULES:
    1. Find exact numbers for unit economics, margins, and market share.
    2. Whenever possible, benchmark against a specific competitor (e.g., {company} vs X).
    3. Score relevance strictly. 10 = existential threat/massive opportunity.
    4. Ignore anything older than 18 months.
    
    Context:
    {raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_challenger(facts: List[IntelligenceFact], raw_search_context: str) -> ChallengerReport:
    """Audits facts, filters low relevance, and assigns confidence scores."""
    structured_llm = llm.with_structured_output(ChallengerReport)
    
    high_relevance_facts = [f for f in facts if f.relevance_score >= 7]
    fact_strings = "\n".join([f"[{f.category}] (Relevance {f.relevance_score}/10) | {f.source_trust}: {f.fact} | Benchmark: {f.competitor_context}" for f in high_relevance_facts])
    
    prompt = f"""
    You are a BCG compliance auditor. Verify these high-relevance facts against the raw context.
    Assign a confidence score (0-100). Penalize heavily if the source is LOW TRUST or the number is an estimation not backed by data.
    
    Facts to check: 
    {fact_strings}
    
    Raw Context:
    {raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_strategist(company: str, verified_facts: List[FactCheckResult]) -> CEOBrief:
    """Drafts the final prioritized brief incorporating the implication layer."""
    structured_llm = llm.with_structured_output(CEOBrief)
    
    valid_data = [f.original_fact for f in verified_facts if f.is_verified and f.confidence_score >= 70]
    
    # SAFEGUARD: Company-specific blind spot identification instead of generic fallback
    if not valid_data:
        valid_data = [f"Intelligence failure. No verified, high-trust data found for {company}'s recent unit economics or competitive market share. Recommend freezing capital allocation until primary data is sourced for their specific supply chain and retail nodes."]
    
    prompt = f"""
    You are a Strategy Partner advising the CEO of {company}. 
    Based ONLY on the verified evidence below, draft a strategic brief.
    
    CRITICAL RULES:
    1. THE IMPLICATION LAYER: You must connect every piece of evidence to a strategic implication before recommending an action.
    2. HYPER-SPECIFICITY: Actions like "optimize costs" or "review strategy" are forbidden. You must name specific geographical markets, product lines, retail channels, or supply chain logistics.
    3. Focus heavily on unit economics and competitive positioning.
    4. Provide no more than 3 actionable directives.
    
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
            
            st.write("📋 Synthesizing implication layer and strategic actions...")
            final_brief = run_strategist(company, audit_data.verifications)
            
            status.update(label="Analysis Complete", state="complete")

        # --- UI DISPLAY ---
        
        # 1. Fact Pipeline Transparency
        st.subheader("🛡️ Intelligence Pipeline & Verification Logs")
        total_extracted = len(research_data.facts)
        passed_audit = sum(1 for f in audit_data.verifications if f.is_verified and f.confidence_score >= 70)
        
        st.caption(f"Raw Facts Extracted: {total_extracted} → Verified & High Trust: {passed_audit}")
        
        with st.expander("View Underlying Data & Challenger Audit"):
            for fc in audit_data.verifications:
                if fc.is_verified and fc.confidence_score >= 70:
                    st.success(f"**[{fc.confidence_score}% Trust]** {fc.original_fact}")
                else:
                    st.error(f"**Rejected/Low Confidence ({fc.confidence_score}%):** {fc.original_fact} \n*Reason: {fc.reasoning}*")

        # 2. Executive Brief
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header(f"Board-Level Strategic Brief — {company.upper()}")
        with col2:
            st.metric(label="Report Confidence", value=f"{final_brief.report_confidence}%")
            
        # 3. The Strategic Narrative
        st.markdown("### The Strategic Narrative")
        with st.container(border=True):
            st.markdown(f"**📉 What Changed:** {final_brief.narrative_what_changed}")
            st.markdown(f"**⏳ Why Now (Catalyst):** {final_brief.narrative_why_now}")
            st.markdown(f"**🎯 The Primary Move:** {final_brief.narrative_primary_move}")
        
        # 4. Implication & Action Matrix
        st.markdown("### Prioritized Strategic Directives")
        for action in final_brief.prioritized_actions:
            color = "🔴" if action.framework == "STOP" else "🟢" if action.framework == "START" else "🔥"
            
            with st.container(border=True):
                st.markdown(f"#### {color} **{action.framework}**: {action.action}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**1. The Evidence**")
                    st.info(f"*{action.evidence}*")
                    st.markdown("**2. The Implication (So What?)**")
                    st.warning(action.implication)
                with c2:
                    st.markdown("**3. Execution Timeline**")
                    st.write(f"📅 {action.timeline}")
                    st.markdown("**4. Expected Business Impact**")
                    st.success(action.expected_impact)
                    st.caption(f"Action Confidence: {action.confidence}/100")

        # 5. Export
        st.divider()
        st.download_button(
            "Download Strategy JSON", 
            data=final_brief.model_dump_json(indent=2), 
            file_name=f"{company}_board_brief.json", 
            mime="application/json"
        )