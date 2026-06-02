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

st.set_page_config(page_title="AI Management Consultant", page_icon="📊", layout="wide")
st.title("📊 Strategic Venture Consultant Engine")
st.markdown("**Data-Driven Market Research** · Anti-Hallucination Pipeline · Financial Strategy · Framework Synthesis")
st.divider()

# ==========================================
# 2. TRUST SCORING, SEARCH & MATH
# ==========================================
HIGH_TRUST = ["mintel.com", "mckinsey.com", "bain.com", "bcg.com", "statista.com", "gartner.com", "ibisworld.com", "bloomberg.com", "reuters.com", "wsj.com"]
LOW_TRUST = ["linkedin.com", "reddit.com", "quora.com", "wikipedia.org", "medium.com"]

def evaluate_trust(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if any(ht in domain for ht in HIGH_TRUST):
        return "HIGH TRUST (Tier 1 Research/Publisher)"
    if any(lt in domain for lt in LOW_TRUST):
        return "LOW TRUST (UGC/Social Media - Verify Carefully)"
    return "MEDIUM TRUST"

def run_market_search(idea: str, geography: str, industry: str) -> str:
    """Multi-query search targeting market size, demographics, competitors, and unit economics."""
    queries = [
        f"{industry} market size TAM growth rate {geography} 2024 2025",
        f"target demographics consumer behavior {industry} {geography}",
        f"top competitors market share {industry} {geography}",
        f"average profit margins cost structure {industry} startups"
    ]
    
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=3, timelimit="y"):
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
    """Calculates confidence programmatically to prevent LLM guessing."""
    if not raw_facts:
        return 0
    
    total_extracted = len(raw_facts)
    total_verified = sum(1 for f in verified_facts if f.is_verified)
    high_trust_count = sum(1 for f in raw_facts if "HIGH TRUST" in f.source_trust)
    
    verification_rate = (total_verified / total_extracted) * 100 if total_extracted > 0 else 0
    trust_ratio = (high_trust_count / total_extracted) * 100 if total_extracted > 0 else 0
    volume_bonus = min(20, total_verified * 2) 
    
    score = (verification_rate * 0.5) + (trust_ratio * 0.3) + volume_bonus
    return min(99, int(score))

# ==========================================
# 3. PYDANTIC SCHEMAS (BULLETPROOF FOR GROQ)
# ==========================================
class MarketFact(BaseModel):
    category: str = Field(description="Must be one of: Market Size, Trends, Demographics, Competitors, Unit Economics")
    fact: str = Field(description="Specific, verifiable data point (numbers/dates).")
    url: str = Field(description="The source URL")
    source_trust: str = Field(description="Trust level provided in the raw context.")

class ResearchOutput(BaseModel):
    facts: List[MarketFact]

class FactCheckResult(BaseModel):
    original_fact: str
    is_verified: bool = Field(description="True ONLY if explicitly backed by the raw search text.")
    reasoning: str = Field(description="Why it passed/failed verification.")

class ChallengerReport(BaseModel):
    verifications: List[FactCheckResult]

class ConsultantReport(BaseModel):
    # Added strict instructions to escape newlines to prevent Groq JSON crashes
    phase_1_industry: str = Field(description="Markdown text with bullet points. Cover TAM/SAM/SOM estimates, Megatrends, and Barriers to Entry. CRITICAL: Escape newlines as '\\n'.")
    phase_2_target: str = Field(description="Markdown text with bullet points. Cover Demographics, Psychographics, Needs & Pain Points, and Buying Habits. CRITICAL: Escape newlines as '\\n'.")
    phase_3_competitive: str = Field(description="Markdown text. Include a Markdown TABLE comparing 2-3 direct and 1-2 indirect competitors. Cover SWOT, Market Share, and Value Prop. CRITICAL: Escape newlines as '\\n'.")
    phase_4_strategy: str = Field(description="Markdown text with bullet points. Cover Product structure, Pricing strategy (with justification), Place (distribution), and Promotion channels. CRITICAL: Escape newlines as '\\n'.")
    phase_5_financials: str = Field(description="Markdown text with bullet points. Cover Cost Structure (fixed/variable), Target Margins, Sales Forecast drivers, and Break-Even levers. CRITICAL: Escape newlines as '\\n'.")
    porters_five_forces: str = Field(description="Markdown text analyzing the 5 forces based on the data. CRITICAL: Escape newlines as '\\n'.")
    five_cs_analysis: str = Field(description="Markdown text summarizing Company, Collaborators, Customers, Competitors, Context. CRITICAL: Escape newlines as '\\n'.")

# ==========================================
# 4. CORE AGENTS
# ==========================================
def run_researcher(idea: str, geography: str, industry: str, raw_search_context: str) -> ResearchOutput:
    """Extracts raw market data and categorizes it."""
    structured_llm = llm.with_structured_output(ResearchOutput)
    prompt = f"""
    You are a Market Research Analyst gathering intelligence for a new venture:
    Idea: {idea}
    Location: {geography}
    Industry: {industry}
    
    RULES:
    1. Extract hard numbers for TAM/SAM/SOM, demographic stats, competitor market shares, and average profit margins.
    2. Ignore anything older than 24 months.
    
    Raw Context:
    {raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_challenger(facts: List[MarketFact], raw_search_context: str) -> ChallengerReport:
    """Audits facts to ensure they exist in the text, preventing hallucination."""
    structured_llm = llm.with_structured_output(ChallengerReport)
    fact_strings = "\n".join([f"[{f.category}] | {f.source_trust}: {f.fact}" for f in facts])
    
    prompt = f"""
    You are a Data Quality Auditor. Verify these market facts against the raw context.
    If a market size number, margin, or competitor stat is an LLM hallucination not found in the text, mark is_verified = False.
    
    Facts to check: 
    {fact_strings}
    
    Raw Context:
    {raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_consultant(idea: str, geography: str, industry: str, verified_facts: List[FactCheckResult]) -> ConsultantReport:
    """Synthesizes the verified data into the 5-phase consulting report."""
    structured_llm = llm.with_structured_output(ConsultantReport)
    valid_data = [f.original_fact for f in verified_facts if f.is_verified]
    
    if not valid_data:
        valid_data = ["No specific verified numerical data found in recent searches. Provide strategic estimations based on general industry benchmarks, but explicitly state where data is estimated."]
    
    prompt = f"""
    You are an Expert Management Consultant and Financial Strategist advising on a new venture.
    Idea: {idea}
    Geography: {geography}
    Industry: {industry}
    
    CRITICAL RULES:
    1. Base your entire analysis ONLY on the verified evidence provided below.
    2. Format your output using clear Markdown. 
    3. You MUST use a Markdown Table in Phase 3 (Competitive Analysis) to compare competitors.
    4. Prioritize concrete estimates, actual industry benchmarks, and realistic strategic insights over generic advice. Be specific to the geography.
    
    Verified Evidence:
    {valid_data}
    """
    return structured_llm.invoke(prompt)

# ==========================================
# 5. STREAMLIT UI
# ==========================================
with st.sidebar:
    st.header("📝 Venture Details")
    idea = st.text_area("Business/Product Idea:", placeholder="e.g., A direct-to-consumer science-backed skincare line focusing on transparency, or a home-based authentic food business")
    geography = st.text_input("Target Geography/Market:", placeholder="e.g., Ahmedabad, India, or National/Global")
    industry = st.text_input("Industry/Sector:", placeholder="e.g., Beauty & Personal Care, or F&B Retail")
    run_btn = st.button("Generate Strategic Analysis", type="primary", use_container_width=True)

if run_btn:
    if not idea or not geography or not industry:
        st.error("Please fill out all venture details in the sidebar.")
    else:
        with st.status("Assembling Consultant AI Crew...", expanded=True) as status:
            st.write("📡 Running targeted market size & competitor search...")
            raw_context = run_market_search(idea, geography, industry)
            
            if not raw_context:
                st.error("Search API failed to return data. Rate limit may be exceeded.")
                st.stop()
                
            st.write("📊 Research Analyst extracting hard data and unit economics...")
            research_data = run_researcher(idea, geography, industry, raw_context)
            
            st.write("⚖️ Auditing evidence to prevent hallucination...")
            audit_data = run_challenger(research_data.facts, raw_context)
            
            st.write("📈 Calculating data confidence score...")
            calculated_confidence = calculate_deterministic_confidence(research_data.facts, audit_data.verifications)
            
            st.write("📋 Chief Strategist drafting 5-Phase Analysis & Frameworks...")
            final_report = run_consultant(idea, geography, industry, audit_data.verifications)
            
            status.update(label="Analysis Complete", state="complete")

        # --- UI DISPLAY ---
        st.header(f"Strategic Market Analysis: {industry}")
        st.caption(f"**Target Market:** {geography} | **Concept:** {idea}")
        
        st.metric(label="Data Integrity & Confidence Score", value=f"{calculated_confidence}%", 
                  help="Programmatically calculated based on the ratio of verified facts to hallucinations, and the density of high-trust primary sources.")
        
        # Pipeline Transparency Log
        with st.expander("🛡️ View Raw Data & Verification Logs"):
            for fc in audit_data.verifications:
                if fc.is_verified:
                    st.success(f"**✅ Verified Data:** {fc.original_fact}")
                else:
                    st.error(f"**❌ Hallucination Filtered:** {fc.original_fact} \n*Reason: {fc.reasoning}*")

        st.divider()

        # Phase Display
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "1. Industry", "2. Target Market", "3. Competition", 
            "4. 4 Ps Strategy", "5. Financials", "Strategic Frameworks"
        ])
        
        with tab1:
            st.subheader("Phase 1: Industry Overview")
            st.markdown(final_report.phase_1_industry)
            
        with tab2:
            st.subheader("Phase 2: The Customer")
            st.markdown(final_report.phase_2_target)
            
        with tab3:
            st.subheader("Phase 3: Competitive Analysis")
            st.markdown(final_report.phase_3_competitive)
            
        with tab4:
            st.subheader("Phase 4: Marketing & Product (The 4 Ps)")
            st.markdown(final_report.phase_4_strategy)
            
        with tab5:
            st.subheader("Phase 5: Financial Analysis & Unit Economics")
            st.markdown(final_report.phase_5_financials)
            
        with tab6:
            st.subheader("Porter's Five Forces")
            st.markdown(final_report.porters_five_forces)
            st.divider()
            st.subheader("5Cs Analysis")
            st.markdown(final_report.five_cs_analysis)

        # Export
        st.divider()
        st.download_button(
            "Download Strategy JSON", 
            data=final_report.model_dump_json(indent=2), 
            file_name="market_analysis_report.json", 
            mime="application/json"
        )