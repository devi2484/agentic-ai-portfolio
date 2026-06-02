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

llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.3-70b-versatile", temperature=0.1)

st.set_page_config(page_title="AI Intelligence Engine", page_icon="♟️", layout="wide")
st.title("♟️ Strategic Intelligence Engine")
st.markdown("**Consulting-Grade Pipeline** · Signal Detection · Competitor Intelligence · Board-Level Briefings")
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

# Facts that are strategically useless — reject on sight
FORBIDDEN_FACT_KEYWORDS = [
    "founded in", "was established", "headquartered in", "ceo of",
    "launched in 20", "won award", "named best", "partnership announced",
    "hired", "appointed", "joined as", "celebrates", "anniversary"
]

def evaluate_trust(url: str) -> tuple[str, int]:
    """Returns (trust_label, trust_penalty). Higher penalty = lower confidence."""
    domain = urlparse(url).netloc.lower().replace("www.", "")
    if any(ht in domain for ht in HIGH_TRUST_DOMAINS):
        return "HIGH TRUST (Tier 1 Financial/Primary Source)", 0
    if any(mt in domain for mt in MEDIUM_TRUST_DOMAINS):
        return "MEDIUM TRUST (Industry Publication)", 15
    if any(lt in domain for lt in LOW_TRUST_DOMAINS):
        return "LOW TRUST (UGC/Social — Heavy Penalty)", 40
    return "MEDIUM TRUST (Unclassified Publication)", 20

def run_enhanced_search(company: str) -> str:
    """
    8-query search targeting the highest-signal strategic data.
    Covers: financials, competitive benchmarks, capital allocation,
    regulatory risks, AI strategy, and supply chain pressure points.
    """
    queries = [
        f"{company} revenue profit margin Q1 Q2 2025",
        f"{company} market share vs competitors 2025",
        f"{company} capital allocation fundraise acquisition 2025",
        f"{company} supply chain logistics cost pressure 2025",
        f"{company} strategic pivot new market expansion 2025",
        f"{company} pricing power customer acquisition cost 2025",
        f"{company} regulatory risk compliance challenge 2025",
        f"{company} AI technology investment strategy 2025",
    ]

    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=3, timelimit="y"):
                    url = r.get("href", "")
                    trust_label, _ = evaluate_trust(url)
                    results.append(
                        f"SOURCE: {url}\n"
                        f"TRUST RATING: {trust_label}\n"
                        f"CONTENT: {r.get('title', '')} — {r.get('body', '')}\n"
                        f"{'-' * 50}"
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
            "One specific, verifiable fact from the source text. "
            "Must include concrete numbers or dates where present. "
            "REJECT: founding dates, awards, product launches >18 months old, "
            "executive biographies, company descriptions, social media activity."
        )
    )
    why_it_matters: str = Field(
        description=(
            "Why would a CEO care if this fact disappeared tomorrow? "
            "If the answer is 'they wouldn't', do not include this fact."
        )
    )
    board_relevance: int = Field(
        description="1-10. ONLY include if >= 8. 10 = existential threat or massive opportunity."
    )
    strategic_impact: int = Field(
        description="1-10. ONLY include if >= 8. Measures direct effect on competitive position."
    )
    source_url: str = Field(description="The exact source URL.")
    source_trust: str = Field(description="Trust level from the raw context header.")
    date_signal: str = Field(
        description=(
            "The date or time reference found in the source (e.g., 'Q1 2025', 'March 2025'). "
            "If no date is present, write 'Undated — treat with caution'."
        )
    )
    competitor_context: str = Field(
        description=(
            "How does this compare to a named direct competitor? "
            "If genuinely no competitor data exists, write 'No benchmark available'."
        )
    )

class ResearchReport(BaseModel):
    company: str
    facts: List[IntelligenceFact] = Field(
        description=(
            "Return EXACTLY 6 facts covering one each of: "
            "Profitability, Growth, Competitive Threat, Competitive Advantage, "
            "Capital Allocation, Strategic Shift. "
            "If a category has no qualifying data, state that explicitly as the fact."
        )
    )

# --- Signal Detector ---
class StrategicSignal(BaseModel):
    signal_type: Literal[
        "Emerging Threat", "Emerging Opportunity", "Strategic Inflection",
        "Capital Shift", "Competitive Surprise", "Moat Erosion", "Moat Strengthening"
    ]
    signal: str = Field(description="The specific signal observed in the data.")
    urgency: Literal["IMMEDIATE", "90-DAY", "6-MONTH", "WATCH"]
    evidence_fact: str = Field(description="The exact fact from research that triggered this signal.")

class SignalReport(BaseModel):
    signals: List[StrategicSignal]

# --- Competitor Intelligence ---
class CompetitorIntel(BaseModel):
    competitor_name: str
    threat_type: Literal[
        "Fastest Growing", "Largest Threat", "Weakening Moat",
        "Strengthening Moat", "Competitive Surprise", "Most Likely Future Threat"
    ]
    threat_summary: str = Field(description="What specific move or metric makes them a threat?")
    advantage_summary: str = Field(description="Where does the target company have an edge over this competitor?")
    recommended_response: str = Field(
        description=(
            "Hyper-specific counter-move. "
            "Forbidden: 'improve innovation', 'focus on customers', 'optimize operations'."
        )
    )

class CompetitorReport(BaseModel):
    competitors: List[CompetitorIntel]

# --- Validation Agent ---
class FactValidation(BaseModel):
    original_fact: str
    keep: bool = Field(
        description=(
            "True ONLY if: (1) explicitly supported by raw search text, "
            "(2) source is HIGH or MEDIUM TRUST, "
            "(3) board_relevance >= 8, (4) not older than 18 months."
        )
    )
    confidence: int = Field(
        description=(
            "0-100. Start at 100. Deduct: 40 for LOW TRUST source, "
            "20 for undated fact, 15 for estimation not backed by data, "
            "10 for MEDIUM TRUST source."
        )
    )
    reason: str = Field(description="Exact reason for keep=True or keep=False.")

class ValidationReport(BaseModel):
    validations: List[FactValidation]

# --- Strategic Action ---
class StrategicAction(BaseModel):
    framework: Literal["STOP", "START", "DOUBLE DOWN"]
    evidence: str = Field(description="The specific verified fact driving this recommendation.")
    implication: str = Field(
        description="The 'So What?' — why this fact fundamentally changes competitive dynamics."
    )
    competitor_context: str = Field(
        description="How a named competitor is positioned on this dimension."
    )
    action: str = Field(
        description=(
            "Hyper-specific operational directive. "
            "Must name specific markets, product lines, channels, or supply chain nodes. "
            "FORBIDDEN: 'improve innovation', 'focus on customers', 'optimize operations', "
            "'review strategy', 'increase efficiency', 'enhance marketing'."
        )
    )
    expected_impact: str = Field(
        description=(
            "Use qualitative language if no hard numbers exist in evidence. "
            "FORBIDDEN: inventing dollar values, percentages, or timelines not in the evidence."
        )
    )
    risk: str = Field(description="The primary risk if this action is taken or not taken.")
    timeline: str = Field(description="e.g., 90 Days | 6 Months | Q3 2025. Must be future-dated.")
    confidence: int = Field(description="1-100 based on evidence quality and source trust.")

# --- Board Brief ---
class CEOBrief(BaseModel):
    company_health_score: int = Field(
        description="0-100 composite score based on profitability, growth trajectory, and competitive position."
    )
    report_confidence: int = Field(description="0-100 based on overall data quality and source trust.")
    narrative_what_changed: str = Field(
        description="What fundamental shift occurred recently in their market, unit economics, or competitive position?"
    )
    narrative_why_now: str = Field(
        description="What is the specific catalyst demanding immediate action?"
    )
    narrative_primary_move: str = Field(
        description="The single most important strategic pivot — hyper-specific, not generic."
    )
    biggest_opportunity: str = Field(description="The highest-upside move available based on evidence.")
    biggest_risk: str = Field(description="The most dangerous threat if left unaddressed.")
    do_not_do: str = Field(
        description="The most tempting but strategically wrong move given current evidence."
    )
    board_message: str = Field(
        description=(
            "A 3-sentence executive summary written as if presenting to the Board of Directors. "
            "Must convey urgency, evidence-backed insight, and a clear call to action."
        )
    )
    prioritized_actions: List[StrategicAction] = Field(
        description="Exactly 3 actions ranked by strategic impact. No more, no fewer."
    )

# ==========================================
# 4. CORE AGENTS
# ==========================================

def run_researcher(company: str, raw_search_context: str) -> ResearchReport:
    """
    Goldman Sachs Research Analyst persona.
    Extracts only board-relevant strategic signals — never trivia.
    """
    structured_llm = llm.with_structured_output(ResearchReport)
    prompt = f"""
You are a Goldman Sachs Research Analyst preparing a fact pack for a Managing Director.
Your job is NOT to summarize. Your job is to find STRATEGIC SIGNALS for {company}.

GOLDEN RULE: Before accepting any fact, ask: "If this fact disappeared tomorrow, would the CEO care?"
If NO — reject it immediately.

MANDATORY OUTPUT: Return EXACTLY 6 facts, one per category:
1. Profitability — revenue, margins, EBITDA, unit economics
2. Growth — GMV, user growth, market expansion, revenue trajectory
3. Competitive Threat — a named competitor gaining ground on {company}
4. Competitive Advantage — where {company} is winning vs competitors
5. Capital Allocation — fundraising, capex, acquisitions, burn rate, dividends
6. Strategic Shift — pivot, new market entry, AI investment, regulatory response

STRICT REJECTION LIST — NEVER include:
- Company founding or history
- Product launches older than 18 months
- Awards, rankings, PR announcements
- Executive biographies or appointments
- Wikipedia-style company descriptions
- Social media activity or follower counts
- Generic industry trends with no {company}-specific data
- Any fact where board_relevance < 8 OR strategic_impact < 8

For competitive benchmarking: always name the specific competitor (e.g., "{company} vs Swiggy").
For financials: include exact numbers if present. If absent, use qualitative language — NEVER invent figures.
For dates: only include facts from the last 18 months. Flag undated facts explicitly.

Raw Search Context:
{raw_search_context}
"""
    return structured_llm.invoke(prompt)


def run_signal_detector(company: str, facts: List[IntelligenceFact]) -> SignalReport:
    """
    Identifies strategic inflection points from the research facts.
    Acts as a filter between research and validation.
    """
    structured_llm = llm.with_structured_output(SignalReport)
    fact_text = "\n".join([
        f"[{f.category}] {f.fact} | Why it matters: {f.why_it_matters} | Competitor: {f.competitor_context}"
        for f in facts
    ])
    prompt = f"""
You are a Strategic Signal Detector for {company}.
Your job is to identify inflection points — moments where the competitive landscape is shifting.

From the research facts below, identify the top strategic signals.

SIGNAL TYPES you must look for:
- Emerging Threat: a competitor or market force that could materially harm {company}
- Emerging Opportunity: an underexploited position {company} could capture
- Strategic Inflection: a pivot point where the business model is changing
- Capital Shift: a major reallocation of resources signalling strategic intent
- Competitive Surprise: an unexpected competitive move that changes the rules
- Moat Erosion: evidence that {company}'s core advantage is weakening
- Moat Strengthening: evidence that {company}'s competitive position is widening

URGENCY SCALE:
- IMMEDIATE: requires action within 30 days
- 90-DAY: requires a decision within one quarter
- 6-MONTH: on the strategic roadmap
- WATCH: monitor but no action yet

Research Facts:
{fact_text}
"""
    return structured_llm.invoke(prompt)


def run_competitor_intel(company: str, raw_search_context: str) -> CompetitorReport:
    """
    Dedicated Competitor Intelligence Agent.
    Identifies named competitors and maps threats/advantages.
    """
    structured_llm = llm.with_structured_output(CompetitorReport)
    prompt = f"""
You are a Competitive Intelligence Specialist analyzing {company}'s competitive landscape.
Your output goes directly to the CEO. Vague analysis is unacceptable.

From the search context, identify up to 3 named competitors and classify them:

THREAT TYPES:
- Fastest Growing: which competitor has the highest momentum right now?
- Largest Threat: which competitor poses the most material risk to {company}?
- Weakening Moat: where is {company} losing its edge to a competitor?
- Strengthening Moat: where is {company} widening its lead?
- Competitive Surprise: an unexpected move from any competitor
- Most Likely Future Threat: who is positioning to attack {company} in 12-18 months?

For each competitor:
- Name them specifically (e.g., "Swiggy", "Zepto", not "a competitor")
- State the exact threat with data if available
- Identify where {company} still has an advantage
- Give a hyper-specific recommended counter-move

FORBIDDEN recommended responses: "improve innovation", "focus on customers", 
"optimize operations", "review strategy", "increase efficiency"

Raw Search Context:
{raw_search_context}
"""
    return structured_llm.invoke(prompt)


def run_validator(facts: List[IntelligenceFact], raw_search_context: str) -> ValidationReport:
    """
    BCG Compliance Auditor persona.
    5-stage validation: recency, source quality, strategic relevance, factual support, board relevance.
    """
    structured_llm = llm.with_structured_output(ValidationReport)

    # Pre-filter: only pass facts that meet the minimum bar
    candidate_facts = [f for f in facts if f.board_relevance >= 8 and f.strategic_impact >= 8]
    fact_strings = "\n".join([
        f"FACT: {f.fact}\n"
        f"Category: {f.category} | Trust: {f.source_trust} | Date: {f.date_signal}\n"
        f"Board Relevance: {f.board_relevance}/10 | Strategic Impact: {f.strategic_impact}/10\n"
        f"Why It Matters: {f.why_it_matters}"
        for f in candidate_facts
    ])

    prompt = f"""
You are a BCG Senior Partner running a compliance audit on intelligence facts.
Your job is to protect the CEO from acting on bad data.

Run each fact through this 5-stage filter:
1. RECENCY CHECK: Is there a date signal? Is it within 18 months? (Fail if undated and unverifiable)
2. SOURCE QUALITY: What is the trust level? Deduct 40 points for LOW TRUST, 20 for MEDIUM TRUST.
3. STRATEGIC RELEVANCE: Does this fact directly affect capital allocation or competitive position?
4. FACTUAL SUPPORT: Is this fact explicitly present in the raw context, or is it inferred/invented?
5. BOARD RELEVANCE: Would a board member ask a follow-up question about this fact?

CONFIDENCE SCORING (start at 100, deduct):
- LOW TRUST source: -40
- MEDIUM TRUST source: -10
- Undated fact: -20
- Estimation not backed by data: -15
- Fact not directly found in raw context: -30

Set keep=False if confidence < 70 OR if the fact fails stages 1, 3, or 4.

Facts to audit:
{fact_strings}

Raw Context (ground truth):
{raw_search_context}
"""
    return structured_llm.invoke(prompt)


def run_strategist(company: str, validated_facts: List[FactValidation],
                   signals: List[StrategicSignal], competitors: List[CompetitorIntel]) -> CEOBrief:
    """
    McKinsey Strategy Partner persona.
    Drafts a board-level brief following the Evidence → Implication → Competitor Context → Action chain.
    """
    structured_llm = llm.with_structured_output(CEOBrief)

    verified_facts = [f.original_fact for f in validated_facts if f.keep and f.confidence >= 70]
    signal_text = "\n".join([
        f"[{s.signal_type} | {s.urgency}] {s.signal} (Evidence: {s.evidence_fact})"
        for s in signals
    ])
    competitor_text = "\n".join([
        f"[{c.threat_type}] {c.competitor_name}: {c.threat_summary} | {company} edge: {c.advantage_summary}"
        for c in competitors
    ])

    if not verified_facts:
        verified_facts = [
            f"INTELLIGENCE FAILURE: No verified, high-trust data found for {company}'s "
            f"recent unit economics or competitive position. "
            f"Recommend halting discretionary capital allocation until primary-source data "
            f"(investor relations, earnings call, regulatory filing) is obtained."
        ]

    prompt = f"""
You are a McKinsey Senior Partner presenting to the Board of {company}.
This is a board memo — not an MBA essay, not a Wikipedia summary, not a generic AI analysis.

MANDATORY CHAIN for every action:
Evidence → Implication → Competitor Context → Action → Expected Impact → Risk

RULES:
1. Every action must be traceable to a specific verified fact.
2. FORBIDDEN actions: "improve innovation", "focus on customers", "optimize operations",
   "review strategy", "increase efficiency", "enhance marketing", "explore opportunities".
3. All actions must name specific: markets, product lines, channels, supply chain nodes, or geographies.
4. ANTI-HALLUCINATION: Never invent dollar values, percentages, or timelines not present in evidence.
   Use qualitative language when hard numbers are absent.
5. Timeline must be future-dated. Reject any recommendation referencing past dates.
6. Provide EXACTLY 3 actions ranked by strategic impact (highest first).

BOARD BRIEF SECTIONS REQUIRED:
- company_health_score: composite 0-100 score on profitability + growth + competitive position
- report_confidence: based on source quality and fact verification rate
- narrative_what_changed: the specific market or unit economics shift, with evidence
- narrative_why_now: the catalyst forcing action now, not in 6 months
- narrative_primary_move: the single most important pivot — hyper-specific
- biggest_opportunity: highest-upside move with evidence
- biggest_risk: most dangerous unaddressed threat
- do_not_do: the most tempting but strategically wrong move
- board_message: 3-sentence executive summary with urgency, evidence, and call to action

Verified Evidence:
{verified_facts}

Strategic Signals:
{signal_text if signal_text else "No signals detected — treat as data gap."}

Competitor Intelligence:
{competitor_text if competitor_text else "No named competitor data found — treat as intelligence gap."}
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
        with st.status(f"Compiling Board Intelligence on {company}...", expanded=True) as status:

            st.write("📡 Executing 8-vector competitive search...")
            raw_context = run_enhanced_search(company)

            if not raw_context:
                st.error("Search API failed to return data.")
                st.stop()

            st.write("📊 Goldman Sachs Researcher — extracting strategic signals...")
            research_data = run_researcher(company, raw_context)

            st.write("🔭 Signal Detector — identifying inflection points...")
            signal_data = run_signal_detector(company, research_data.facts)

            st.write("🎯 Competitor Intelligence — mapping threats and advantages...")
            competitor_data = run_competitor_intel(company, raw_context)

            st.write("⚖️ BCG Auditor — 5-stage fact validation...")
            validation_data = run_validator(research_data.facts, raw_context)

            st.write("📋 McKinsey Strategist — synthesizing board brief...")
            final_brief = run_strategist(
                company,
                validation_data.validations,
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
        passed_audit = sum(1 for f in validation_data.validations if f.keep and f.confidence >= 70)
        st.caption(f"Facts Extracted: {total_extracted} → Passed 5-Stage Audit: {passed_audit} → Signals Detected: {len(signal_data.signals)}")

        with st.expander("View Fact Audit, Signals & Competitor Intel"):
            st.markdown("**Fact Audit**")
            for fv in validation_data.validations:
                if fv.keep and fv.confidence >= 70:
                    st.success(f"**[{fv.confidence}% Confidence]** {fv.original_fact}")
                else:
                    st.error(f"**Rejected ({fv.confidence}%):** {fv.original_fact}\n*{fv.reason}*")

            if signal_data.signals:
                st.divider()
                st.markdown("**Strategic Signals**")
                for sig in signal_data.signals:
                    urgency_color = "🔴" if sig.urgency == "IMMEDIATE" else "🟡" if sig.urgency == "90-DAY" else "🟢"
                    st.info(f"{urgency_color} **[{sig.signal_type} | {sig.urgency}]** {sig.signal}")

            if competitor_data.competitors:
                st.divider()
                st.markdown("**Competitor Intelligence**")
                for c in competitor_data.competitors:
                    st.warning(f"**[{c.threat_type}] {c.competitor_name}:** {c.threat_summary}")

        # --- Executive Brief Header ---
        st.divider()
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.header(f"Board-Level Strategic Brief — {company.upper()}")
        with col2:
            st.metric(label="Health Score", value=f"{final_brief.company_health_score}/100")
        with col3:
            st.metric(label="Report Confidence", value=f"{final_brief.report_confidence}%")

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

        # --- Opportunities, Risks, Do Not Do ---
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
                    st.markdown(f"**Recommended Counter-Move:** {c.recommended_response}")

        # --- Prioritized Actions ---
        st.markdown("### Prioritized Strategic Directives")
        for i, action in enumerate(final_brief.prioritized_actions, 1):
            color = "🔴" if action.framework == "STOP" else "🟢" if action.framework == "START" else "🔥"
            with st.container(border=True):
                st.markdown(f"#### #{i} {color} **{action.framework}**: {action.action}")

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
            "board_brief": final_brief.model_dump(),
            "signals": [s.model_dump() for s in signal_data.signals],
            "competitor_intel": [c.model_dump() for c in competitor_data.competitors],
            "fact_audit": [f.model_dump() for f in validation_data.validations],
        }
        import json
        st.download_button(
            "Download Full Intelligence Package (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"{company}_board_brief.json",
            mime="application/json"
        )