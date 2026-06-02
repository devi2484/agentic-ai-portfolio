import os
import streamlit as st
from langchain_groq import ChatGroq
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY") or st.secrets.get("GROQ_KEY", "")

# Using llama-3.3-70b for the structured output as it is much more reliable for JSON
llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.3-70b-versatile", temperature=0.1)

st.set_page_config(page_title="AI Market Intelligence Crew", page_icon="🔍", layout="wide")
st.title("🔍 Corporate Market Intelligence Crew")
st.markdown("**3 AI agents** — Goldman Sachs Researcher → BCG Auditor → McKinsey Strategist")
st.divider()

# ==========================================
# 2. PYDANTIC SCHEMAS (JSON VALIDATION)
# ==========================================
class IntelligenceFact(BaseModel):
    fact: str = Field(description="A specific, verifiable fact with numbers/dates.")
    source_context: str = Field(description="Brief snippet of the raw search text where this was found.")
    confidence_score: int = Field(description="Score 1-10 based on how explicitly it is stated in the text.")
    strategic_implication: str = Field(description="Why this matters for the business model.")

class ResearchReport(BaseModel):
    company: str
    competitor_mentions: List[str] = Field(description="Names of competitors found in the research.")
    facts: List[IntelligenceFact]

class FactCheckResult(BaseModel):
    original_fact: str
    is_verified: bool = Field(description="True ONLY if the fact is explicitly backed by the raw search text.")
    reasoning: str = Field(description="Explanation of why it passed or failed.")
    corrected_fact: Optional[str] = Field(description="If false, provide the corrected fact based on text, or leave null.")

class ChallengerReport(BaseModel):
    verifications: List[FactCheckResult]

class StrategicAction(BaseModel):
    urgency: str = Field(description="HIGH, MEDIUM, or LOW")
    recommended_action: str = Field(description="Specific, non-generic action step.")
    expected_impact: str = Field(description="Quantifiable or strategic business impact.")

class CEOBrief(BaseModel):
    threat_level: str = Field(description="Overall threat assessment (e.g., SEVERE, MODERATE).")
    competitive_intelligence: str = Field(description="Summary of market positioning vs competitors.")
    key_insights: List[str] = Field(description="Core insights regarding unit economics, supply chain, or retail strategy.")
    actions: List[StrategicAction]

# ==========================================
# 3. CORE FUNCTIONS & AGENTS
# ==========================================
def run_enhanced_search(company: str) -> str:
    """Multi-query search for broader competitive intelligence."""
    queries = [
        f"{company} financial results unit economics 2025",
        f"{company} market share competitors 2025",
        f"{company} supply chain retail strategy recent news"
    ]
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                # Limit to 2 results per query to keep context window manageable
                for r in ddgs.text(q, max_results=2):
                    results.append(f"[{r.get('href', 'url')}] {r.get('title', '')}: {r.get('body', '')}")
    except Exception as e:
        st.error(f"Search API Error: {e}")
    
    return "\n".join(results)

def run_researcher(company: str, raw_search_context: str) -> ResearchReport:
    """Extracts raw facts and structures them into JSON."""
    structured_llm = llm.with_structured_output(ResearchReport)
    prompt = f"""
    You are a Goldman Sachs research analyst. Extract precise intelligence for {company} from the provided raw search context.
    Focus on hard numbers, dates, and direct competitor mentions. 
    Do NOT invent information. If a metric is not in the text, do not guess.
    
    Raw Search Context:
    {raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_challenger(facts: List[IntelligenceFact], raw_search_context: str) -> ChallengerReport:
    """Audits the researcher's output against the raw text to catch hallucinations."""
    structured_llm = llm.with_structured_output(ChallengerReport)
    
    fact_strings = "\n".join([f"- {f.fact}" for f in facts])
    
    prompt = f"""
    You are a ruthless BCG compliance auditor. Check if the following facts are explicitly supported by the raw context.
    If a fact is an assumption or hallucination, mark it unverified and provide a correction based ONLY on the text.
    
    Facts to check: 
    {fact_strings}
    
    Raw Context:
    {raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_strategist(company: str, verified_facts: List[FactCheckResult]) -> CEOBrief:
    """Drafts the final executive brief based only on verified data."""
    structured_llm = llm.with_structured_output(CEOBrief)
    
    # Filter for only verified or successfully corrected facts
    valid_data = [f.original_fact if f.is_verified else f.corrected_fact for f in verified_facts if f.is_verified or f.corrected_fact]
    
    prompt = f"""
    You are a McKinsey Partner. Based STRICTLY on the verified data below, draft a CEO Brief for {company}.
    Focus heavily on unit economics, supply chain vulnerabilities, and retail psychology/positioning.
    Ensure recommendations are highly specific (e.g., pivot specific product lines, alter logistics routes), NOT generic (e.g., 'increase marketing').
    
    Verified Data:
    {valid_data}
    """
    return structured_llm.invoke(prompt)

# ==========================================
# 4. STREAMLIT UI
# ==========================================
company = st.text_input("Company Name:", placeholder="e.g. Zomato, Reliance, Nykaa...")

if st.button("Deploy Corporate Crew", type="primary"):
    if not company:
        st.error("Please enter a company name.")
    else:
        with st.status(f"Conducting Enterprise Analysis on {company}...", expanded=True) as status:
            st.write("📡 Running targeted market search...")
            raw_context = run_enhanced_search(company)
            
            if not raw_context:
                st.error("Could not retrieve search data. The search API might be rate-limited.")
                st.stop()
                
            st.write("📊 Goldman Sachs Researcher structuring intelligence...")
            research_data = run_researcher(company, raw_context)
            
            st.write("⚖️ BCG Auditor verifying facts (Anti-Hallucination)...")
            audit_data = run_challenger(research_data.facts, raw_context)
            
            st.write("📋 McKinsey Strategist drafting CEO Brief...")
            final_brief = run_strategist(company, audit_data.verifications)
            
            status.update(label="Mission complete!", state="complete")

        # --- UI DISPLAY ---
        
        # 1. Fact Check & Verification Tab
        st.subheader("🛡️ Data Verification & Filtering")
        cols = st.columns(2)
        verified_count = sum(1 for f in audit_data.verifications if f.is_verified)
        st.caption(f"Passed {verified_count}/{len(audit_data.verifications)} facts through the Challenger Agent.")
        
        for fc in audit_data.verifications:
            if fc.is_verified:
                st.success(f"**✅ Verified:** {fc.original_fact} \n\n*Reasoning: {fc.reasoning}*")
            else:
                st.error(f"**❌ Hallucination Flagged:** {fc.original_fact} \n\n*Correction:* {fc.corrected_fact}")

        # 2. Executive Brief
        st.divider()
        st.header(f"Executive Strategic Brief — {company.upper()}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🔴 Threat Level:** {final_brief.threat_level}")
        with col2:
            st.markdown(f"**⚔️ Competitors Monitored:** {', '.join(research_data.competitor_mentions)}")
            
        st.markdown("### Market Positioning")
        st.info(final_brief.competitive_intelligence)
        
        st.markdown("### Key Strategic Insights")
        for insight in final_brief.key_insights:
            st.markdown(f"- {insight}")
            
        st.markdown("### 🎯 Recommended Strategic Actions")
        for action in final_brief.actions:
            urgency_color = "🔴" if action.urgency.upper() == "HIGH" else "🟡" if action.urgency.upper() == "MEDIUM" else "🟢"
            with st.expander(f"{urgency_color} [{action.urgency}] {action.recommended_action}"):
                st.markdown(f"**Expected Business Impact:**")
                st.write(action.expected_impact)

        # 3. Export
        st.divider()
        raw_export = f"RAW CONTEXT:\n{raw_context}\n\nSTRATEGIC BRIEF:\n{final_brief.model_dump_json(indent=2)}"
        st.download_button("Download Full Data Report (JSON/TXT)", data=raw_export, file_name=f"{company}_intelligence.txt", mime="text/plain")

st.caption("Advanced AI Crew: JSON Validation · Fact Checking · Competitive Intelligence")