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

st.set_page_config(page_title="Elite Market Intelligence", page_icon="📈", layout="wide")
st.title("📈 Elite Market Intelligence & Financial Strategist")
st.markdown("**Granular Data Pipeline** · Real Competitor Extraction · CapEx Benchmarking · Unit Economics")
st.divider()

# ==========================================
# 2. TRUST SCORING, SEARCH & MATH
# ==========================================
HIGH_TRUST = ["mintel.com", "mckinsey.com", "bain.com", "bcg.com", "statista.com", "gartner.com", "ibisworld.com", "bloomberg.com", "reuters.com", "wsj.com", "inc42.com", "entrackr.com", "moneycontrol.com"]
LOW_TRUST = ["linkedin.com", "reddit.com", "quora.com", "wikipedia.org", "medium.com"]

def evaluate_trust(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if any(ht in domain for ht in HIGH_TRUST):
        return "HIGH TRUST (Tier 1 Research/Publisher)"
    if any(lt in domain for lt in LOW_TRUST):
        return "LOW TRUST (UGC/Social Media - Verify Carefully)"
    return "MEDIUM TRUST"

def run_market_search(idea: str, geography: str, industry: str) -> str:
    """Highly targeted search queries designed to find real competitors and actual investment amounts."""
    queries = [
        f"{industry} market size TAM growth rate {geography} 2024 2025",
        f"top specific competitors brand names {idea} {industry} {geography}", # Targeted for real names
        f"average startup costs initial investment CapEx {industry} {geography}", # Targeted for real money
        f"unit economics profit margins pricing strategy {industry} competitors"
    ]
    
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                # Increased max_results to 3 to ensure we capture actual competitor names
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
    # Forced the LLM to specifically categorize Competitor Names and Investment Costs
    category: str = Field(description="Must be one of: Market Size, Specific Competitor Brand Name, Investment/CapEx, Unit Economics, Trends")
    fact: str = Field(description="Specific, verifiable data point. If a competitor, name the brand. If investment, state the currency/amount.")
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
    # Integrated the granular 5-phase structure and maintained newline escaping for Groq safety
    phase_1_industry: str = Field(description="Markdown. Cover quantitative TAM/SAM/SOM sizing with math/formulas, 5-year CAGR, deep-dive trend matrix, and structural barriers (including CapEx). CRITICAL: Escape newlines as '\\n'.")
    phase_2_target: str = Field(description="Markdown. Cover primary/secondary archetypes, a Pain Point Root-Cause Analysis TABLE, and micro-buying habits. CRITICAL: Escape newlines as '\\n'.")
    phase_3_competitive: str = Field(description="Markdown. Include a Markdown TABLE analyzing 2 direct and 2 indirect named competitors (Position, Price Tier, Value Prop, Distribution, Vulnerability). Cover Deep SWOT of top incumbent and White-Space differentiation. CRITICAL: Escape newlines as '\\n'.")
    phase_4_strategy: str = Field(description="Markdown. Cover product specifications, advanced pricing architecture, omni-channel distribution funnel, and high-ROI promotion playbook. CRITICAL: Escape newlines as '\\n'.")
    phase_5_financials: str = Field(description="Markdown. Cover granular cost structures (fixed/variable), target unit economics, strategic break-even volume, and 3-year scalability roadblocks. CRITICAL: Escape newlines as '\\n'.")
    framework_synthesis: str = Field(description="Markdown. Synthesize Porter's Five Forces (with intensity ratings and empirical justification) and a 5Cs Analysis. CRITICAL: Escape newlines as '\\n'.")

# ==========================================
# 4. CORE AGENTS
# ==========================================
def run_researcher(idea: str, geography: str, industry: str, raw_search_context: str) -> ResearchOutput:
    structured_llm = llm.with_structured_output(ResearchOutput)
    prompt = f"""
    You are an Elite Market Intelligence Researcher for a new venture:
    Idea: {idea}
    Location: {geography}
    Industry: {industry}
    
    RULES:
    1. You MUST extract specific, real-world COMPETITOR BRAND NAMES from the text.
    2. You MUST extract actual numbers for INVESTMENT, STARTUP COSTS, or CapEx.
    3. Extract hard numbers for TAM/SAM/SOM and unit economics.
    4. Do not invent names or numbers. If not in the text, do not extract it.
    
    Raw Context:
    {raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_challenger(facts: List[MarketFact], raw_search_context: str) -> ChallengerReport:
    structured_llm = llm.with_structured_output(ChallengerReport)
    fact_strings = "\n".join([f"[{f.category}] | {f.source_trust}: {f.fact}" for f in facts])
    
    prompt = f"""
    You are a Data Quality Auditor. Verify these market facts against the raw context.
    If a market size, investment cost, or competitor brand name is an LLM hallucination NOT found in the text, mark is_verified = False.
    
    Facts to check: 
    {fact_strings}
    
    Raw Context:
    {raw_search_context}
    """
    return structured_llm.invoke(prompt)

def run_consultant(idea: str, geography: str, industry: str, verified_facts: List[FactCheckResult]) -> ConsultantReport:
    structured_llm = llm.with_structured_output(ConsultantReport)
    valid_data = [f.original_fact for f in verified_facts if f.is_verified]
    
    if not valid_data:
        valid_data = ["No specific verified numerical data found in recent searches. Provide strategic estimations based on general industry benchmarks, but explicitly state where data is estimated."]
    
    prompt = f"""
    Act as an elite Market Intelligence Director, Consumer Insights Lead, and Financial Econometrician. I need a comprehensive, highly granular, and data-backed market research analysis for a new venture. Do not provide high-level summaries, generic advice, or filler text. If exact figures are unavailable, provide realistic, industry-benchmarked proxy estimates based on current market data and clearly state your assumptions.

    Idea: {idea}
    Geography: {geography}
    Industry: {industry}
    
    CRITICAL RULES:
    1. Base your entire analysis heavily on the verified evidence provided below. Use the real competitor names and actual investment figures provided.
    2. Format your output using clear Markdown. 
    3. You MUST use Markdown Tables in Phase 2 (Pain Points) and Phase 3 (Competitor Matrix).
    4. Provide the exact math/formulas used for TAM/SAM/SOM estimations and Unit Economics.
    5. JSON FORMATTING (CRITICAL): You must output valid JSON. Escape all line breaks as '\\n'. Do NOT use raw/unescaped line breaks inside the string fields.
    
    Verified Evidence:
    {valid_data}
    """
    return structured_llm.invoke(prompt)

# ==========================================
# 5. STREAMLIT UI
# ==========================================
with st.sidebar:
    st.header("📝 Venture Details")
    idea = st.text_area("Business/Product Core:", placeholder="e.g., A premium, traditional homemade Gujarati condiment brand using clean ingredients")
    geography = st.text_input("Target Geography & Scope:", placeholder="e.g., Ahmedabad, India - urban/semi-urban segments")
    industry = st.text_input("Sector/Sub-vertical:", placeholder="e.g., FMCG, artisanal food retail, D2C packaged goods")
    run_btn = st.button("Generate Strategic Analysis", type="primary", use_container_width=True)

if run_btn:
    if not idea or not geography or not industry:
        st.error("Please fill out all venture details in the sidebar.")
    else:
        with st.status("Assembling Intelligence Crew...", expanded=True) as status:
            st.write("📡 Running deep-dive search for real competitors and CapEx data...")
            raw_context = run_market_search(idea, geography, industry)
            
            if not raw_context:
                st.error("Search API failed to return data. Rate limit may be exceeded.")
                st.stop()
                
            st.write("📊 Extracting competitor names and financial benchmarks...")
            research_data = run_researcher(idea, geography, industry, raw_context)
            
            st.write("⚖️ Auditing evidence to destroy hallucinations...")
            audit_data = run_challenger(research_data.facts, raw_context)
            
            st.write("📈 Calculating data confidence score...")
            calculated_confidence = calculate_deterministic_confidence(research_data.facts, audit_data.verifications)
            
            st.write("📋 Director drafting 5-Phase Analysis & Frameworks...")
            final_report = run_consultant(idea, geography, industry, audit_data.verifications)
            
            status.update(label="Analysis Complete", state="complete")

        # --- UI DISPLAY ---
        st.header(f"Elite Strategic Analysis: {industry}")
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
            "1. Macro & Industry", "2. Target Consumers", "3. Competitive Intel", 
            "4. GTM Strategy", "5. Financial Modeling", "Framework Synthesis"
        ])
        
        with tab1:
            st.subheader("PHASE 1: RIGOROUS INDUSTRY & MACRO-ENVIRONMENT ANALYSIS")
            st.markdown(final_report.phase_1_industry)
            
        with tab2:
            st.subheader("PHASE 2: GRANULAR TARGET MARKET & CONSUMER ARCHETYPES")
            st.markdown(final_report.phase_2_target)
            
        with tab3:
            st.subheader("PHASE 3: ADVERSARIAL COMPETITIVE INTELLIGENCE")
            st.markdown(final_report.phase_3_competitive)
            
        with tab4:
            st.subheader("PHASE 4: GRANULAR MARKETING & GO-TO-MARKET (GTM) STRATEGY")
            st.markdown(final_report.phase_4_strategy)
            
        with tab5:
            st.subheader("PHASE 5: COMPREHENSIVE FINANCIAL MODELING & UNIT ECONOMICS")
            st.markdown(final_report.phase_5_financials)
            
        with tab6:
            st.subheader("FRAMEWORK SYNTHESIS")
            st.markdown(final_report.framework_synthesis)

        # Export
        st.divider()
        st.download_button(
            "Download Strategy JSON", 
            data=final_report.model_dump_json(indent=2), 
            file_name="elite_market_analysis.json", 
            mime="application/json"
        )