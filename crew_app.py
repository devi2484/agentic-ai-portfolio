import os
import json
import time
import streamlit as st
from langchain_groq import ChatGroq
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Literal
from urllib.parse import urlparse

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY") or st.secrets.get("GROQ_KEY", "")

# 70b — deep reasoning agents (researcher, strategist)
llm_heavy = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.3-70b-versatile", temperature=0.1)
# 8b — lightweight structured agents (signal detector, competitor intel)
llm_fast = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.1-8b-instant", temperature=0.1)

st.set_page_config(page_title="AI Intelligence Engine", page_icon="♟️", layout="wide")
st.title("♟️ Strategic Intelligence Engine")
st.markdown("**Consulting-Grade Pipeline** · Hard-Gate Validation · Signal Detection · Board-Level Briefings")
st.divider()

# ==========================================
# 2. TRUST SCORING & SEARCH
# ==========================================
HIGH_TRUST_DOMAINS = [
    "reuters.com", "bloomberg.com", "cnbc.com", "wsj.com", "ft.com",
    "sec.gov", "moneycontrol.com", "economictimes.indiatimes.com",
    "livemint.com", "businessstandard.com", "thehindubusinessline.com",
    "financialexpress.com", "bseindia.com", "nseindia.com"
]
MEDIUM_TRUST_DOMAINS = [
    "techcrunch.com", "forbes.com", "inc42.com", "entrackr.com",
    "yourstory.com", "themorningcontext.com", "restofworld.org"
]
LOW_TRUST_DOMAINS = [
    "linkedin.com", "reddit.com", "quora.com", "wikipedia.org",
    "medium.com", "twitter.com", "x.com"
]

# Numeric trust scores used in programmatic confidence calculation
TRUST_SCORE_MAP = {
    "HIGH TRUST": 10,
    "MEDIUM TRUST": 6,
    "LOW TRUST": 2,
}

def evaluate_trust(url: str) -> str:
    domain = urlparse(url).netloc.lower().replace("www.", "")
    if any(ht in domain for ht in HIGH_TRUST_DOMAINS):
        return "HIGH TRUST"
    if any(mt in domain for mt in MEDIUM_TRUST_DOMAINS):
        return "MEDIUM TRUST"
    if any(lt in domain for lt in LOW_TRUST_DOMAINS):
        return "LOW TRUST"
    return "MEDIUM TRUST"

def calculate_confidence(trust_label: str, board_relevance: int, strategic_impact: int) -> int:
    """
    Programmatic confidence — never LLM-invented.
    Formula: (trust_score * 0.4) + (board_relevance * 0.3) + (strategic_impact * 0.3)
    Scaled to 0-100.
    """
    trust_score = TRUST_SCORE_MAP.get(trust_label.split("(")[0].strip(), 5)
    raw = (trust_score * 0.4) + (board_relevance * 0.3) + (strategic_impact * 0.3)
    return int((raw / 10) * 100)

def run_enhanced_search(company: str) -> str:
    """5-query search — highest signal-to-token ratio queries only."""
    queries = [
        f"{company} revenue profit margin 2025",
        f"{company} market share competitors 2025",
        f"{company} capital allocation acquisition strategic pivot 2025",
        f"{company} regulatory risk supply chain pressure 2025",
        f"{company} AI investment pricing power 2025",
    ]
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=2, timelimit="y"):
                    url = r.get("href", "")
                    trust = evaluate_trust(url)
                    results.append(
                        f"SOURCE: {url}\nTRUST: {trust}\n"
                        f"CONTENT: {r.get('title', '')} — {r.get('body', '')}\n"
                        f"{'-' * 40}"
                    )
    except Exception as e:
        st.error(f"Search API Error: {e}")
    return "\n".join(results)

# ==========================================
# 3. PYDANTIC SCHEMAS
# ==========================================

# --- Research Agent ---
class IntelligenceFact(BaseModel):
    category: Literal[
        "Profitability", "Growth", "Competitive Threat",
        "Competitive Advantage", "Capital Allocation", "Strategic Shift"
    ]
    fact: str = Field(
        description=(
            "One specific, verifiable fact from the source. "
            "Include numbers or dates where present. "
            "HARD REJECT: founding dates, awards, executive bios, "
            "company descriptions, social media, product launches >18 months old."
        )
    )
    why_it_matters: str = Field(
        description="Why would the board care if this fact disappeared tomorrow? If they wouldn't — reject it."
    )
    board_relevance: int = Field(description="1-10. Only include if >= 8.")
    strategic_impact: int = Field(description="1-10. Only include if >= 8.")
    source_url: str = Field(description="Exact source URL.")
    source_trust: str = Field(description="Copy exactly from TRUST label in raw context: HIGH TRUST, MEDIUM TRUST, or LOW TRUST.")
    date_signal: str = Field(
        description="Date from the source e.g. 'Q1 2025'. If absent write 'Undated'."
    )
    competitor_context: str = Field(
        description="Named competitor comparison. If none exists write 'No benchmark available'."
    )

class ResearchReport(BaseModel):
    company: str
    facts: List[IntelligenceFact] = Field(
        description=(
            "Return EXACTLY 6 facts — one per category: "
            "Profitability, Growth, Competitive Threat, Competitive Advantage, "
            "Capital Allocation, Strategic Shift. "
            "If a category has no qualifying data write that explicitly."
        )
    )

# --- Validated Fact (produced programmatically, not by LLM) ---
class ValidatedFact(BaseModel):
    category: str
    fact: str
    why_it_matters: str
    source_url: str
    source_trust: str
    date_signal: str
    competitor_context: str
    board_relevance: int
    strategic_impact: int
    confidence: int  # Calculated programmatically

# --- Signal Detector (runs ONLY on validated facts) ---
class StrategicSignal(BaseModel):
    signal_type: Literal[
        "Emerging Threat", "Emerging Opportunity", "Strategic Inflection",
        "Capital Shift", "Competitive Surprise", "Moat Erosion", "Moat Strengthening",
        "Regulatory Risk", "Margin Compression", "Pricing Pressure"
    ]
    signal: str = Field(description="The specific inflection point observed.")
    urgency: Literal["IMMEDIATE", "90-DAY", "6-MONTH", "WATCH"]
    evidence_fact: str = Field(description="The exact validated fact that triggered this signal.")

class SignalReport(BaseModel):
    signals: List[StrategicSignal]

# --- Competitor Intelligence ---
class CompetitorIntel(BaseModel):
    competitor_name: str
    threat_type: Literal[
        "Fastest Growing", "Largest Threat", "Weakening Moat",
        "Strengthening Moat", "Competitive Surprise", "Most Likely Future Threat"
    ]
    threat_summary: str = Field(description="Specific move or metric making them a threat.")
    advantage_summary: str = Field(description="Where the target company still has an edge.")
    recommended_response: str = Field(
        description=(
            "Hyper-specific counter-move naming markets, product lines, or channels. "
            "FORBIDDEN: 'improve innovation', 'focus on customers', 'optimize operations', "
            "'review strategy', 'increase efficiency'."
        )
    )

class CompetitorReport(BaseModel):
    competitors: List[CompetitorIntel]

# --- Strategic Action ---
class StrategicAction(BaseModel):
    framework: Literal["STOP", "START", "DOUBLE DOWN"]
    evidence: str = Field(description="The specific validated fact driving this recommendation.")
    implication: str = Field(description="The 'So What?' — why this changes competitive dynamics.")
    competitor_context: str = Field(description="How a named competitor is positioned on this dimension.")
    action: str = Field(
        description=(
            "Hyper-specific directive naming markets, product lines, channels, or supply chain nodes. "
            "FORBIDDEN: 'improve innovation', 'focus on customers', 'optimize operations', "
            "'review strategy', 'increase efficiency', 'enhance marketing', 'explore opportunities'."
        )
    )
    expected_impact: str = Field(
        description=(
            "Use qualitative language only. "
            "NEVER invent dollar values, percentages, or market share figures not in evidence."
        )
    )
    risk: str = Field(description="Primary risk if this action is taken or ignored.")
    timeline: str = Field(description="Future-dated only. e.g. '90 Days', '6 Months', 'Q3 2025'.")
    confidence: int = Field(description="1-100 based on evidence quality.")

# --- Board Brief ---
class CEOBrief(BaseModel):
    company_health_score: int = Field(
        description="0-100 composite: profitability + growth trajectory + competitive position."
    )
    report_confidence: int = Field(description="0-100 based on validated fact quality and source trust.")
    narrative_what_changed: str = Field(
        description="Specific recent shift in market, unit economics, or competitive position — with evidence."
    )
    narrative_why_now: str = Field(description="The specific catalyst demanding action now, not in 6 months.")
    narrative_primary_move: str = Field(description="Single most important strategic pivot — hyper-specific.")
    biggest_opportunity: str = Field(description="Highest-upside move supported by evidence.")
    biggest_risk: str = Field(description="Most dangerous unaddressed threat if left alone.")
    do_not_do: str = Field(description="Most tempting but strategically wrong move given current evidence.")
    board_message: str = Field(
        description="3-sentence executive summary: urgency + evidence-backed insight + call to action."
    )
    prioritized_actions: List[StrategicAction] = Field(
        description="Exactly 3 actions ranked by strategic impact, highest first."
    )

# ==========================================
# 4. PIPELINE — CORRECT ORDER
#
# Search
# ↓
# Research Agent
# ↓
# Hard-Gate Validation (programmatic — no LLM)
# ↓  [only verified_facts proceed past this point]
# Competitor Intelligence
# ↓
# Signal Detector  ← runs ONLY on validated facts
# ↓
# Strategist
# ==========================================

def run_researcher(company: str, raw_context: str) -> ResearchReport:
    """Goldman Sachs Research Analyst — extracts strategic signals, not trivia. Uses 70b for reasoning depth."""
    structured_llm = llm_heavy.with_structured_output(ResearchReport)
    prompt = f"""You are a Goldman Sachs Research Analyst preparing a fact pack for a Managing Director.

GOLDEN RULE: Before accepting any fact ask: "If this fact disappeared tomorrow, would the board care?"
If NO — reject it immediately.

MANDATORY OUTPUT: Exactly 6 facts, one per category:
1. Profitability — revenue, margins, EBITDA, unit economics
2. Growth — GMV, user growth, market expansion, revenue trajectory  
3. Competitive Threat — a named competitor gaining ground on {company}
4. Competitive Advantage — where {company} is winning vs competitors
5. Capital Allocation — fundraising, capex, acquisitions, burn rate
6. Strategic Shift — pivot, new market, AI investment, regulatory response

HARD REJECT LIST:
- Company founding dates or history
- Product launches older than 18 months
- Awards, PR announcements, rankings
- Executive biographies or appointments
- Generic industry trends with no {company}-specific data
- Any fact where board_relevance < 8 OR strategic_impact < 8

For source_trust: copy EXACTLY from the TRUST label in the raw context (HIGH TRUST / MEDIUM TRUST / LOW TRUST).
For financials: exact numbers if present. If absent — qualitative language only. NEVER invent figures.
For dates: facts from last 18 months only. Write "Undated" if no date found.

Raw Search Context:
{raw_context}"""
    return structured_llm.invoke(prompt)


def run_hard_gate_validation(facts: List[IntelligenceFact]) -> List[ValidatedFact]:
    """
    CRITICAL RULE #1 & #4: Programmatic hard gate — no LLM involvement.
    Confidence calculated from formula, not invented.
    Only facts passing ALL criteria proceed downstream.
    """
    verified = []
    for f in facts:
        # Hard reject on LLM scores below threshold
        if f.board_relevance < 8 or f.strategic_impact < 8:
            continue
        # Hard reject low trust sources
        trust_key = f.source_trust.split("(")[0].strip()
        if trust_key == "LOW TRUST":
            continue
        # Programmatic confidence — formula from spec
        confidence = calculate_confidence(f.source_trust, f.board_relevance, f.strategic_impact)
        # Hard reject below confidence threshold
        if confidence < 70:
            continue
        # Undated facts from unclassified sources get an extra penalty
        if "Undated" in f.date_signal and trust_key != "HIGH TRUST":
            continue
        verified.append(ValidatedFact(
            category=f.category,
            fact=f.fact,
            why_it_matters=f.why_it_matters,
            source_url=f.source_url,
            source_trust=f.source_trust,
            date_signal=f.date_signal,
            competitor_context=f.competitor_context,
            board_relevance=f.board_relevance,
            strategic_impact=f.strategic_impact,
            confidence=confidence,
        ))
    return verified


def run_competitor_intel(company: str, raw_context: str) -> CompetitorReport:
    """Competitor Intelligence — uses 8b model, structured extraction task."""
    structured_llm = llm_fast.with_structured_output(CompetitorReport)
    prompt = f"""You are a Competitive Intelligence Specialist. Your output goes directly to the CEO.

From the search context identify up to 3 NAMED competitors for {company}.
Name them specifically (e.g. "Swiggy", not "a competitor").

For each competitor:
- State the exact threat with data where available
- Identify where {company} still has an advantage
- Give a hyper-specific counter-move naming specific markets, product lines, or channels

FORBIDDEN counter-moves: "improve innovation", "focus on customers", "optimize operations", 
"review strategy", "increase efficiency", "enhance marketing"

Raw Search Context:
{raw_context}"""
    return structured_llm.invoke(prompt)


def run_signal_detector(company: str, verified_facts: List[ValidatedFact]) -> SignalReport:
    """
    CRITICAL RULE #2: Signals generated ONLY from validated facts.
    Uses 8b model — classification task, not deep reasoning.
    """
    structured_llm = llm_fast.with_structured_output(SignalReport)
    fact_text = "\n".join([
        f"[{f.category} | {f.source_trust} | {f.date_signal}] {f.fact} | Why it matters: {f.why_it_matters}"
        for f in verified_facts
    ])
    prompt = f"""You are a Strategic Signal Detector for {company}.
Identify inflection points from ONLY the validated facts below.

URGENCY:
- IMMEDIATE: action required within 30 days
- 90-DAY: decision required this quarter
- 6-MONTH: on strategic roadmap
- WATCH: monitor, no action yet

Validated Facts:
{fact_text}"""
    return structured_llm.invoke(prompt)


def run_strategist(company: str, verified_facts: List[ValidatedFact],
                   signals: List[StrategicSignal], competitors: List[CompetitorIntel]) -> CEOBrief:
    """
    CRITICAL RULE #3 & #4. Uses 70b for board-level reasoning quality.
    """
    structured_llm = llm_heavy.with_structured_output(CEOBrief)

    if not verified_facts:
        fact_text = (
            f"INTELLIGENCE FAILURE: No high-trust verified data found for {company}. "
            "Recommend halting discretionary capital allocation until primary-source data "
            "(investor relations, earnings call, regulatory filing) is obtained."
        )
    else:
        fact_text = "\n".join([
            f"[{f.category} | Confidence {f.confidence}%] {f.fact} | {f.why_it_matters}"
            for f in verified_facts
        ])

    signal_text = "\n".join([
        f"[{s.signal_type} | {s.urgency}] {s.signal}"
        for s in signals
    ]) if signals else "No signals detected."

    competitor_text = "\n".join([
        f"[{c.threat_type}] {c.competitor_name}: {c.threat_summary} | Edge: {c.advantage_summary}"
        for c in competitors
    ]) if competitors else "No competitor data found."

    prompt = f"""You are a McKinsey Senior Partner presenting to the Board of {company}.
This is a board memo — not an MBA essay, not a summary, not generic AI analysis.

MANDATORY CHAIN for every action:
Evidence → Implication → Competitor Context → Action → Expected Impact → Risk

RULES:
1. Every action must be traceable to a specific verified fact below.
2. FORBIDDEN actions: "improve innovation", "focus on customers", "optimize operations",
   "review strategy", "increase efficiency", "enhance marketing", "explore opportunities"
3. Actions must name specific markets, product lines, channels, or supply chain nodes.
4. ANTI-HALLUCINATION: Never invent dollar values, percentages, market share, or timelines.
   Use qualitative language when hard numbers are absent.
5. Timelines must be future-dated. Never reference past dates.
6. Provide EXACTLY 3 actions ranked by strategic impact (highest first).

Verified Evidence (ONLY these facts may be used):
{fact_text}

Strategic Signals:
{signal_text}

Competitor Intelligence:
{competitor_text}"""
    return structured_llm.invoke(prompt)


# ==========================================
# 5. STREAMLIT UI
# ==========================================
company = st.text_input("Target Company:", placeholder="e.g. Zomato, Reliance, Tesla, Nykaa...")

if st.button("Run Strategic Analysis", type="primary"):
    if not company:
        st.error("Please enter a company name.")
    else:
        with st.status(f"Compiling Board Intelligence on {company}...", expanded=True) as status:

            # STEP 1 — Search
            st.write("📡 Executing 8-vector competitive intelligence search...")
            raw_context = run_enhanced_search(company)
            if not raw_context:
                st.error("Search API failed to return data.")
                st.stop()

            time.sleep(3)

            # STEP 2 — Research Agent
            st.write("📊 Goldman Sachs Researcher — extracting strategic signals...")
            research_data = run_researcher(company, raw_context[:3000])

            # STEP 3 — Hard-Gate Validation (programmatic, no LLM call, no sleep needed)
            st.write("🔒 Hard-Gate Validation — programmatic confidence scoring...")
            verified_facts = run_hard_gate_validation(research_data.facts)
            st.write(f"   → {len(research_data.facts)} facts extracted · {len(verified_facts)} passed hard gate")

            if not verified_facts:
                st.warning("No facts passed the hard gate. Report will flag intelligence failure.")

            time.sleep(4)

            # STEP 4 — Competitor Intelligence (independent of validation, uses raw context)
            st.write("🎯 Competitor Intelligence — mapping named threats and advantages...")
            competitor_data = run_competitor_intel(company, raw_context[:2000])

            time.sleep(4)

            # STEP 5 — Signal Detector (runs ONLY on verified_facts)
            st.write("🔭 Signal Detector — identifying inflection points from validated facts only...")
            signal_data = run_signal_detector(company, verified_facts)

            time.sleep(4)

            # STEP 6 — Strategist (receives only verified_facts)
            st.write("📋 McKinsey Strategist — synthesizing board brief from verified evidence...")
            final_brief = run_strategist(
                company,
                verified_facts,
                signal_data.signals,
                competitor_data.competitors
            )

            status.update(label="Analysis Complete", state="complete")

        # ==========================================
        # DISPLAY
        # ==========================================

        # --- Pipeline Transparency ---
        st.subheader("🛡️ Intelligence Pipeline & Verification Logs")
        total_extracted = len(research_data.facts)
        passed_gate = len(verified_facts)
        gate_rate = int((passed_gate / total_extracted * 100)) if total_extracted else 0

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Facts Extracted", total_extracted)
        col_m2.metric("Passed Hard Gate", passed_gate)
        col_m3.metric("Gate Pass Rate", f"{gate_rate}%")
        col_m4.metric("Signals Detected", len(signal_data.signals))

        with st.expander("View Pipeline Detail: Facts, Rejections, Signals, Competitor Intel"):

            st.markdown("**✅ Verified Facts (passed all gates)**")
            if verified_facts:
                for vf in verified_facts:
                    st.success(
                        f"**[{vf.category} | {vf.confidence}% confidence | {vf.source_trust} | {vf.date_signal}]**\n\n"
                        f"{vf.fact}\n\n*Why it matters: {vf.why_it_matters}*"
                    )
            else:
                st.warning("No facts passed the hard gate.")

            st.markdown("**❌ Rejected Facts (failed hard gate)**")
            rejected = [
                f for f in research_data.facts
                if not any(vf.fact == f.fact for vf in verified_facts)
            ]
            for rf in rejected:
                conf = calculate_confidence(rf.source_trust, rf.board_relevance, rf.strategic_impact)
                reasons = []
                if rf.board_relevance < 8:
                    reasons.append(f"board_relevance={rf.board_relevance} (min 8)")
                if rf.strategic_impact < 8:
                    reasons.append(f"strategic_impact={rf.strategic_impact} (min 8)")
                if "LOW TRUST" in rf.source_trust:
                    reasons.append("LOW TRUST source")
                if conf < 70:
                    reasons.append(f"confidence={conf}% (min 70%)")
                if "Undated" in rf.date_signal and "HIGH TRUST" not in rf.source_trust:
                    reasons.append("Undated + non-HIGH-TRUST source")
                st.error(
                    f"**[{rf.category} | {conf}% confidence]** {rf.fact}\n\n"
                    f"*Rejected: {' · '.join(reasons) if reasons else 'Failed gate criteria'}*"
                )

            if signal_data.signals:
                st.divider()
                st.markdown("**🔭 Strategic Signals (from validated facts only)**")
                for sig in signal_data.signals:
                    icon = "🔴" if sig.urgency == "IMMEDIATE" else "🟡" if sig.urgency == "90-DAY" else "🟢"
                    st.info(f"{icon} **[{sig.signal_type} | {sig.urgency}]** {sig.signal}\n\n*Evidence: {sig.evidence_fact}*")

            if competitor_data.competitors:
                st.divider()
                st.markdown("**🎯 Competitor Intelligence**")
                for c in competitor_data.competitors:
                    st.warning(f"**[{c.threat_type}] {c.competitor_name}:** {c.threat_summary}")

        # --- Executive Brief Header ---
        st.divider()
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.header(f"Board-Level Strategic Brief — {company.upper()}")
        with col2:
            st.metric("Health Score", f"{final_brief.company_health_score}/100")
        with col3:
            st.metric("Report Confidence", f"{final_brief.report_confidence}%")

        # --- Board Message ---
        st.markdown("### 📢 Board Message")
        with st.container(border=True):
            st.markdown(f"*{final_brief.board_message}*")

        # --- Strategic Narrative ---
        st.markdown("### The Strategic Narrative")
        with st.container(border=True):
            st.markdown(f"**📉 What Changed:** {final_brief.narrative_what_changed}")
            st.markdown(f"**⏳ Why Now (Catalyst):** {final_brief.narrative_why_now}")
            st.markdown(f"**🎯 Primary Move:** {final_brief.narrative_primary_move}")

        # --- Opportunity / Risk / Do Not Do ---
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.markdown("**🚀 Biggest Opportunity**")
                st.success(final_brief.biggest_opportunity)
        with c2:
            with st.container(border=True):
                st.markdown("**⚠️ Biggest Risk**")
                st.error(final_brief.biggest_risk)
        with c3:
            with st.container(border=True):
                st.markdown("**🚫 Do NOT Do**")
                st.warning(final_brief.do_not_do)

        # --- Competitor Benchmarks ---
        if competitor_data.competitors:
            st.markdown("### 🏆 Competitor Benchmarks")
            for c in competitor_data.competitors:
                with st.container(border=True):
                    st.markdown(f"#### ⚔️ {c.competitor_name} — {c.threat_type}")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Their Threat**")
                        st.error(c.threat_summary)
                    with col_b:
                        st.markdown(f"**{company}'s Edge**")
                        st.success(c.advantage_summary)
                    st.markdown(f"**Counter-Move:** {c.recommended_response}")

        # --- Prioritized Actions ---
        st.markdown("### Prioritized Strategic Directives")
        for i, action in enumerate(final_brief.prioritized_actions, 1):
            icon = "🔴" if action.framework == "STOP" else "🟢" if action.framework == "START" else "🔥"
            with st.container(border=True):
                st.markdown(f"#### #{i} {icon} **{action.framework}**: {action.action}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**1. Evidence**")
                    st.info(f"*{action.evidence}*")
                    st.markdown("**2. Implication (So What?)**")
                    st.warning(action.implication)
                    st.markdown("**3. Competitor Context**")
                    st.caption(action.competitor_context)
                with c2:
                    st.markdown("**4. Timeline**")
                    st.write(f"📅 {action.timeline}")
                    st.markdown("**5. Expected Impact**")
                    st.success(action.expected_impact)
                    st.markdown("**6. Risk**")
                    st.error(action.risk)
                    st.caption(f"Action Confidence: {action.confidence}/100")

        # --- Export ---
        st.divider()
        export_data = {
            "company": company,
            "health_score": final_brief.company_health_score,
            "report_confidence": final_brief.report_confidence,
            "pipeline_stats": {
                "facts_extracted": total_extracted,
                "passed_hard_gate": passed_gate,
                "gate_pass_rate_pct": gate_rate,
                "signals_detected": len(signal_data.signals),
            },
            "verified_facts": [vf.model_dump() for vf in verified_facts],
            "board_brief": final_brief.model_dump(),
            "signals": [s.model_dump() for s in signal_data.signals],
            "competitor_intel": [c.model_dump() for c in competitor_data.competitors],
        }
        st.download_button(
            "Download Full Intelligence Package (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"{company}_board_brief.json",
            mime="application/json"
        )