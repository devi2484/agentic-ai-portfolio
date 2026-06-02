import os
import streamlit as st
from langchain_groq import ChatGroq
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from urllib.parse import urlparse

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY") or st.secrets.get("GROQ_KEY", "")

# Using 70b for complex reasoning and strict JSON adherence
llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.3-70b-versatile", temperature=0.1)

st.set_page_config(page_title="AI Market Intelligence Crew", page_icon="📈", layout="wide")
st.title("📈 Enterprise Intelligence Engine")
st.markdown("**Data-Driven Strategy** · Trust Scoring · Relevance Filtering · Action Prioritization")
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
    """Multi-query search returning URLs, dates, and trust scores."""
    queries = [
        f"{company} recent earnings revenue profit 2025",
        f"{company} market share vs competitors 2025",
        f"{company} strategic acquisitions capital allocation",
        f"{company} supply chain retail strategy issues"
    ]
    
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=2, timelimit="y"): # time awareness builtin
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
    category: Literal["Revenue", "Profit", "Competitive", "Product", "Strategic", "Capital Allocation"]
    fact: str = Field(description="Specific, verifiable fact with numbers/dates. Ignore older than 18 months.")
    url: str = Field(description="The source URL")
    relevance_score: int = Field(description="1-10 score. 1=Trivia, 10=Board-level strategic importance.")
    source_trust: str = Field(description="Trust level provided in the raw context.")

class ResearchReport(BaseModel):
    company: str
    facts: List[IntelligenceFact]

class FactCheckResult(BaseModel):
    original_fact: str
    relevance_score: int
    is_verified: bool = Field(description="True ONLY if explicitly backed by the raw search text.")
    confidence_score: int = Field(description="0-100 based on source trust and text clarity.")
    reasoning: str = Field(description="Why it passed/failed verification.")

class ChallengerReport(BaseModel):
    verifications: List[FactCheckResult]

class StrategicAction(BaseModel):
    framework: Literal["STOP", "START", "DOUBLE DOWN"]
    evidence: str = Field(description="The specific verified fact driving this recommendation.")
    why_it_matters: str = Field(description="Strategic implication for unit economics, retail positioning, or supply chain.")
    action: str = Field(description="Highly specific operational action. Generic advice is forbidden.")
    expected_impact: str = Field(description="Quantifiable business impact.")
    risk: str = Field(description="Primary execution risk or black swan vulnerability.")

class CEOBrief(BaseModel):
    report_confidence: int = Field(description="0-100 score based on overall data quality and verification.")
    strategic_narrative: str = Field(description="What changed? Why now? What should management do?")
    prioritized_actions: List[StrategicAction] = Field(description="Exactly 3 top actions ranked by ROI/Impact.")

# ==========================================
# 4. CORE AGENTS
# ==========================================
def run_researcher(company: str, raw_search_context: str) -> ResearchReport:
    """Extracts facts, categorizes them, and scores them for CEO relevance."""
    structured_llm = llm.with_structured_output(ResearchReport)
    prompt = f"""
    You are a Goldman Sachs research analyst. Extract precise intelligence for {company}.
    
    RULES:
    1. Find exactly ONE fact for each category if possible: Revenue, Profit, Competitive, Product, Strategic, Capital Allocation.
    2. Score relevance strictly. 10 = existential threat/massive opportunity. 2 = minor PR update.
    3. Ignore anything older than 24 months.
    
    Context:
    {raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_challenger(facts: List[IntelligenceFact], raw_search_context: str) -> ChallengerReport:
    """Audits facts, filters low relevance, and assigns confidence scores."""
    structured_llm = llm.with_structured_output(ChallengerReport)
    
    # PRIORITY 1: Pre-filter low relevance facts before validation
    high_relevance_facts = [f for f in facts if f.relevance_score >= 8]
    fact_strings = "\n".join([f"[{f.category}] (Relevance {f.relevance_score}/10) | {f.source_trust}: {f.fact}" for f in high_relevance_facts])
    
    prompt = f"""
    You are a BCG compliance auditor. Verify the following highly-relevant facts against the raw context.
    Assign a confidence score (0-100). Penalize heavily if the source is LOW TRUST or evidence is weak.
    
    Facts to check: 
    {fact_strings}
    
    Raw Context:
    {raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_strategist(company: str, verified_facts: List[FactCheckResult]) -> CEOBrief:
    """Drafts the final prioritized brief based on evidence."""
    structured_llm = llm.with_structured_output(CEOBrief)
    
    valid_data = [f.original_fact for f in verified_facts if f.is_verified and f.confidence_score >= 70]
    
    # SAFEGUARD: If no facts survived, feed it a controlled failure state
    if not valid_data:
        valid_data = ["No verified, high-trust data was found in recent searches. Cannot form evidence-based strategy."]
    
    prompt = f"""
    You are a McKinsey Partner advising the CEO of {company}. 
    Based ONLY on the verified evidence below, draft a strategic brief.
    
    RULES:
    1. You MUST provide exactly 3 prioritized actions. Rank by Impact, Cost, and Time-to-value.
    2. Use the STOP / START / DOUBLE DOWN framework.
    3. No generic advice ("improve marketing"). Be hyper-specific about operations, positioning, or unit economics.
    4. If the evidence states 'No verified data', advise the CEO that market intelligence is currently insufficient and recommend an internal data audit as the primary action.
    
    Verified Evidence:
    {valid_data}
    """
    return structured_llm.invoke(prompt)

# ==========================================
# 5. STREAMLIT UI
# ==========================================
company = st.text_input("Target Company:", placeholder="e.g. Zomato, Reliance, Tesla...")

if st.button("Deploy Intelligence Engine", type="primary"):
    if not company:
        st.error("Please enter a company name.")
    else:
        with st.status(f"Compiling Enterprise Intelligence on {company}...", expanded=True) as status:
            st.write("📡 Executing targeted multi-query search & trust scoring...")
            raw_context = run_enhanced_search(company)
            
            if not raw_context:
                st.error("Search API failed to return data.")
                st.stop()
                
            st.write("📊 Extracting, categorizing, and scoring strategic relevance...")
            research_data = run_researcher(company, raw_context)
            
            st.write("⚖️ Filtering trivia & auditing sources (Anti-Hallucination)...")
            audit_data = run_challenger(research_data.facts, raw_context)
            
            st.write("📋 Drafting Prioritized CEO Brief...")
            final_brief = run_strategist(company, audit_data.verifications)
            
            status.update(label="Analysis Complete", state="complete")

        # --- UI DISPLAY ---
        
        # 1. Fact Pipeline Transparency
        st.subheader("🛡️ Data Pipeline & Filtering")
        total_extracted = len(research_data.facts)
        high_rel = len([f for f in research_data.facts if f.relevance_score >= 8])
        passed_audit = sum(1 for f in audit_data.verifications if f.is_verified and f.confidence_score >= 70)
        
        st.caption(f"Extracted: {total_extracted} → High Relevance (>=8/10): {high_rel} → Verified & High Trust: {passed_audit}")
        
        with st.expander("View Data Verification Logs"):
            for fc in audit_data.verifications:
                if fc.is_verified and fc.confidence_score >= 70:
                    st.success(f"**[{fc.confidence_score}% Confidence]** {fc.original_fact}")
                else:
                    st.error(f"**Rejected/Low Confidence ({fc.confidence_score}%):** {fc.original_fact} \n*Reason: {fc.reasoning}*")

        # 2. Executive Brief
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header(f"Strategic Brief — {company.upper()}")
        with col2:
            st.metric(label="Overall Report Confidence", value=f"{final_brief.report_confidence}%")
            
        st.markdown("### The Strategic Narrative")
        st.info(final_brief.strategic_narrative)
        
        st.markdown("### 🎯 Top 3 Prioritized Actions")
        for action in final_brief.prioritized_actions:
            # Color coding the framework
            color = "🔴" if action.framework == "STOP" else "🟢" if action.framework == "START" else "🔥"
            
            with st.container(border=True):
                st.markdown(f"#### {color} **{action.framework}**: {action.action}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Evidence & Implication**")
                    st.write(f"*{action.evidence}*")
                    st.caption(f"**Why it matters:** {action.why_it_matters}")
                with c2:
                    st.markdown("**Impact & Risk**")
                    st.write(f"📈 **Expected Impact:** {action.expected_impact}")
                    st.write(f"⚠️ **Risk:** {action.risk}")

        # 3. Export
        st.divider()
        st.download_button(
            "Download Strategy JSON", 
            data=final_brief.model_dump_json(indent=2), 
            file_name=f"{company}_strategy.json", 
            mime="application/json"
        )
st.caption("v2.0 Architecture · Relevance Filtering · Stop/Start/Double Down Framework")