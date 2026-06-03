import os
import json
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from urllib.parse import urlparse

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY") or st.secrets.get("GROQ_KEY", "")

# Single model — plain invoke, no tool-call schema (avoids all 400 errors)
llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.1-8b-instant", temperature=0.1)

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
TRUST_SCORE_MAP = {"HIGH TRUST": 10, "MEDIUM TRUST": 6, "LOW TRUST": 2}

def evaluate_trust(url: str) -> str:
    domain = urlparse(url).netloc.lower().replace("www.", "")
    if any(h in domain for h in HIGH_TRUST_DOMAINS):   return "HIGH TRUST"
    if any(m in domain for m in MEDIUM_TRUST_DOMAINS): return "MEDIUM TRUST"
    if any(l in domain for l in LOW_TRUST_DOMAINS):    return "LOW TRUST"
    return "MEDIUM TRUST"

def calculate_confidence(trust_label: str, board_relevance: int, strategic_impact: int) -> int:
    trust_score = TRUST_SCORE_MAP.get(trust_label.strip(), 5)
    raw = (trust_score * 0.4) + (board_relevance * 0.3) + (strategic_impact * 0.3)
    return int((raw / 10) * 100)

def run_enhanced_search(company: str) -> str:
    queries = [
        f"{company} revenue profit margin 2025",
        f"{company} market share competitors 2025",
        f"{company} capital allocation acquisition strategic pivot 2025",
        f"{company} regulatory risk supply chain 2025",
        f"{company} AI investment pricing power 2025",
    ]
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=2, timelimit="y"):
                    url = r.get("href", "")
                    results.append(
                        f"SOURCE: {url}\nTRUST: {evaluate_trust(url)}\n"
                        f"CONTENT: {r.get('title','')} — {r.get('body','')}\n{'-'*40}"
                    )
    except Exception as e:
        st.error(f"Search error: {e}")
    return "\n".join(results)

# ==========================================
# 3. JSON INVOKE HELPER
# Bypasses with_structured_output entirely — no Groq tool-call validator = no 400 errors
# ==========================================
def invoke_json(prompt: str) -> dict:
    """Call LLM, strip markdown fences, parse JSON. Returns dict or raises."""
    messages = [
        SystemMessage(content="You are a precise JSON-only responder. Output ONLY valid JSON. No markdown, no explanation, no code fences."),
        HumanMessage(content=prompt)
    ]
    resp = llm.invoke(messages)
    text = resp.content.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip().rstrip("```").strip()
    return json.loads(text)

# ==========================================
# 4. PYDANTIC MODELS (for internal use / display only — not sent to Groq)
# ==========================================
class IntelligenceFact(BaseModel):
    category: str
    fact: str
    why_it_matters: str
    board_relevance: int
    strategic_impact: int
    source_url: str
    source_trust: str
    date_signal: str = "Undated"
    competitor_context: str = "No benchmark available"

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
    confidence: int

class StrategicSignal(BaseModel):
    signal_type: str
    signal: str
    urgency: str
    evidence_fact: str

class CompetitorIntel(BaseModel):
    competitor_name: str
    threat_type: str
    threat_summary: str
    advantage_summary: str
    recommended_response: str

class StrategicAction(BaseModel):
    framework: str
    evidence: str
    implication: str
    competitor_context: str
    action: str
    expected_impact: str
    risk: str
    timeline: str

class CEOBrief(BaseModel):
    company_health_score: int
    report_confidence: int
    narrative_what_changed: str
    narrative_why_now: str
    narrative_primary_move: str
    biggest_opportunity: str
    biggest_risk: str
    do_not_do: str
    board_message: str
    prioritized_actions: List[StrategicAction]

# ==========================================
# 5. PIPELINE AGENTS
# ==========================================

FACT_CATEGORIES = ["Profitability", "Growth", "Competitive Threat",
                   "Competitive Advantage", "Capital Allocation", "Strategic Shift"]

def run_researcher(company: str, raw_context: str) -> List[IntelligenceFact]:
    prompt = f"""You are a Goldman Sachs Research Analyst. Extract strategic intelligence for {company}.

GOLDEN RULE: "If this fact disappeared tomorrow, would the board care?" If NO — reject it.

Return a JSON object with this exact structure:
{{
  "facts": [
    {{
      "category": "Profitability",
      "fact": "specific fact with numbers",
      "why_it_matters": "board-level reason",
      "board_relevance": 9,
      "strategic_impact": 9,
      "source_url": "https://...",
      "source_trust": "HIGH TRUST or MEDIUM TRUST or LOW TRUST",
      "date_signal": "Q1 2025 or Undated",
      "competitor_context": "vs CompetitorName or No benchmark available"
    }}
  ]
}}

Return EXACTLY 6 facts — one per category: {", ".join(FACT_CATEGORIES)}

HARD REJECT: founding dates, awards, executive bios, social media, product launches >18 months old.
Only include facts where board_relevance >= 8 AND strategic_impact >= 8.
source_trust must be copied EXACTLY from the TRUST label in the context.
NEVER invent numbers. Use qualitative language if no hard data exists.

Raw Search Context:
{raw_context}"""

    try:
        data = invoke_json(prompt)
        facts = []
        for f in data.get("facts", []):
            try:
                facts.append(IntelligenceFact(**f))
            except Exception:
                continue
        return facts
    except Exception as e:
        st.warning(f"Researcher parse error: {e}")
        return []


def run_hard_gate_validation(facts: List[IntelligenceFact]) -> List[ValidatedFact]:
    verified = []
    for f in facts:
        if f.board_relevance < 8 or f.strategic_impact < 8:
            continue
        if "LOW TRUST" in f.source_trust:
            continue
        confidence = calculate_confidence(f.source_trust, f.board_relevance, f.strategic_impact)
        if confidence < 70:
            continue
        if f.date_signal == "Undated" and "HIGH TRUST" not in f.source_trust:
            continue
        verified.append(ValidatedFact(
            category=f.category, fact=f.fact, why_it_matters=f.why_it_matters,
            source_url=f.source_url, source_trust=f.source_trust,
            date_signal=f.date_signal, competitor_context=f.competitor_context,
            board_relevance=f.board_relevance, strategic_impact=f.strategic_impact,
            confidence=confidence,
        ))
    return verified


def run_competitor_intel(company: str, raw_context: str) -> List[CompetitorIntel]:
    prompt = f"""You are a Competitive Intelligence Specialist analysing {company}. Output goes to the CEO.

Return a JSON object:
{{
  "competitors": [
    {{
      "competitor_name": "exact name e.g. Swiggy",
      "threat_type": "one of: Fastest Growing, Largest Threat, Weakening Moat, Strengthening Moat, Competitive Surprise, Most Likely Future Threat",
      "threat_summary": "specific threat with data if available",
      "advantage_summary": "where {company} still leads",
      "recommended_response": "specific counter-move naming markets or product lines"
    }}
  ]
}}

Identify up to 3 NAMED competitors only. Use real company names.
FORBIDDEN responses: improve innovation, focus on customers, optimize operations, review strategy.

Raw Search Context:
{raw_context}"""

    try:
        data = invoke_json(prompt)
        comps = []
        for c in data.get("competitors", []):
            try:
                comps.append(CompetitorIntel(**c))
            except Exception:
                continue
        return comps
    except Exception as e:
        st.warning(f"Competitor intel parse error: {e}")
        return []


def run_signal_detector(company: str, verified_facts: List[ValidatedFact]) -> List[StrategicSignal]:
    if not verified_facts:
        return []

    fact_text = "\n".join([
        f"[{f.category} | {f.source_trust} | {f.date_signal}] {f.fact}"
        for f in verified_facts
    ])

    prompt = f"""You are a Strategic Signal Detector for {company}.
Identify inflection points from the validated facts below.

Return a JSON object:
{{
  "signals": [
    {{
      "signal_type": "one of: Emerging Threat, Emerging Opportunity, Strategic Inflection, Capital Shift, Competitive Surprise, Moat Erosion, Moat Strengthening, Regulatory Risk, Margin Compression, Pricing Pressure",
      "signal": "specific inflection point",
      "urgency": "one of: IMMEDIATE, 90-DAY, 6-MONTH, WATCH",
      "evidence_fact": "the exact fact that triggered this"
    }}
  ]
}}

Validated Facts (use ONLY these):
{fact_text}"""

    try:
        data = invoke_json(prompt)
        signals = []
        for s in data.get("signals", []):
            try:
                signals.append(StrategicSignal(**s))
            except Exception:
                continue
        return signals
    except Exception as e:
        st.warning(f"Signal detector parse error: {e}")
        return []


def run_strategist(company: str, verified_facts: List[ValidatedFact],
                   signals: List[StrategicSignal], competitors: List[CompetitorIntel]) -> Optional[CEOBrief]:

    fact_text = "\n".join([
        f"[{f.category} | {f.confidence}% confidence] {f.fact} | {f.why_it_matters}"
        for f in verified_facts
    ]) if verified_facts else f"INTELLIGENCE FAILURE: No verified data for {company}."

    signal_text = "\n".join([f"[{s.signal_type}|{s.urgency}] {s.signal}" for s in signals]) or "None"
    competitor_text = "\n".join([f"[{c.threat_type}] {c.competitor_name}: {c.threat_summary}" for c in competitors]) or "None"

    prompt = f"""You are a McKinsey Senior Partner presenting to the Board of {company}.

Return a JSON object with this EXACT structure — no extra fields, no missing fields:
{{
  "company_health_score": 75,
  "report_confidence": 80,
  "narrative_what_changed": "specific recent shift with evidence",
  "narrative_why_now": "specific catalyst for immediate action",
  "narrative_primary_move": "hyper-specific single most important pivot",
  "biggest_opportunity": "highest-upside move from evidence",
  "biggest_risk": "most dangerous unaddressed threat",
  "do_not_do": "most tempting but wrong move",
  "board_message": "3-sentence: urgency + evidence + call to action",
  "prioritized_actions": [
    {{
      "framework": "STOP or START or DOUBLE DOWN",
      "evidence": "exact verified fact",
      "implication": "so what — why this changes competitive dynamics",
      "competitor_context": "how a named competitor is positioned",
      "action": "specific directive naming markets/products/channels",
      "expected_impact": "qualitative impact — no invented numbers",
      "risk": "primary risk if taken or ignored",
      "timeline": "90 Days or 6 Months or Q3 2025"
    }}
  ]
}}

RULES:
- EXACTLY 3 prioritized_actions, ranked by strategic impact
- NEVER invent dollar values, percentages, or market share
- Actions must name specific markets, product lines, or channels
- FORBIDDEN actions: improve innovation, focus on customers, optimize operations, review strategy

Verified Evidence:
{fact_text}

Signals:
{signal_text}

Competitor Intelligence:
{competitor_text}"""

    try:
        data = invoke_json(prompt)
        actions = []
        for a in data.get("prioritized_actions", []):
            try:
                actions.append(StrategicAction(**a))
            except Exception:
                continue
        data["prioritized_actions"] = actions
        return CEOBrief(**data)
    except Exception as e:
        st.error(f"Strategist parse error: {e}")
        return None


# ==========================================
# 6. STREAMLIT UI
# ==========================================
company = st.text_input("Target Company:", placeholder="e.g. Zomato, Reliance, Tesla, Nykaa...")

if st.button("Run Strategic Analysis", type="primary"):
    if not company:
        st.error("Please enter a company name.")
    else:
        with st.status(f"Compiling Board Intelligence on {company}...", expanded=True) as status:

            st.write("📡 Executing competitive intelligence search...")
            raw_context = run_enhanced_search(company)
            if not raw_context:
                st.error("Search returned no data.")
                st.stop()

            time.sleep(3)
            st.write("📊 Researcher — extracting strategic signals...")
            raw_facts = run_researcher(company, raw_context[:3000])

            st.write("🔒 Hard-Gate Validation — programmatic confidence scoring...")
            verified_facts = run_hard_gate_validation(raw_facts)
            st.write(f"   → {len(raw_facts)} extracted · {len(verified_facts)} passed gate")

            time.sleep(4)
            st.write("🎯 Competitor Intelligence...")
            competitors = run_competitor_intel(company, raw_context[:2000])

            time.sleep(4)
            st.write("🔭 Signal Detector (validated facts only)...")
            signals = run_signal_detector(company, verified_facts)

            time.sleep(4)
            st.write("📋 Strategist — synthesizing board brief...")
            final_brief = run_strategist(company, verified_facts, signals, competitors)

            status.update(label="Analysis Complete", state="complete")

        if not final_brief:
            st.error("Strategist failed to produce a brief. Try again.")
            st.stop()

        # --- Pipeline Stats ---
        st.subheader("🛡️ Intelligence Pipeline")
        total = len(raw_facts)
        passed = len(verified_facts)
        rate = int(passed / total * 100) if total else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Facts Extracted", total)
        c2.metric("Passed Hard Gate", passed)
        c3.metric("Gate Pass Rate", f"{rate}%")
        c4.metric("Signals", len(signals))

        with st.expander("View Pipeline Detail"):
            st.markdown("**✅ Verified Facts**")
            for vf in verified_facts:
                st.success(f"**[{vf.category} | {vf.confidence}% | {vf.source_trust} | {vf.date_signal}]**\n\n{vf.fact}\n\n*{vf.why_it_matters}*")

            st.markdown("**❌ Rejected Facts**")
            rejected = [f for f in raw_facts if not any(vf.fact == f.fact for vf in verified_facts)]
            for rf in rejected:
                conf = calculate_confidence(rf.source_trust, rf.board_relevance, rf.strategic_impact)
                reasons = []
                if rf.board_relevance < 8:    reasons.append(f"board_relevance={rf.board_relevance}")
                if rf.strategic_impact < 8:   reasons.append(f"strategic_impact={rf.strategic_impact}")
                if "LOW TRUST" in rf.source_trust: reasons.append("LOW TRUST source")
                if conf < 70:                  reasons.append(f"confidence={conf}%")
                st.error(f"**[{rf.category} | {conf}%]** {rf.fact}\n\n*Rejected: {' · '.join(reasons) or 'gate criteria'}*")

            if signals:
                st.divider()
                st.markdown("**🔭 Signals**")
                for s in signals:
                    icon = "🔴" if s.urgency == "IMMEDIATE" else "🟡" if s.urgency == "90-DAY" else "🟢"
                    st.info(f"{icon} **[{s.signal_type} | {s.urgency}]** {s.signal}\n\n*Evidence: {s.evidence_fact}*")

            if competitors:
                st.divider()
                st.markdown("**🎯 Competitor Intel**")
                for c in competitors:
                    st.warning(f"**[{c.threat_type}] {c.competitor_name}:** {c.threat_summary}")

        # --- Board Brief ---
        st.divider()
        h1, h2, h3 = st.columns([3, 1, 1])
        with h1: st.header(f"Board-Level Strategic Brief — {company.upper()}")
        with h2: st.metric("Health Score", f"{final_brief.company_health_score}/100")
        with h3: st.metric("Report Confidence", f"{final_brief.report_confidence}%")

        st.markdown("### 📢 Board Message")
        with st.container(border=True):
            st.markdown(f"*{final_brief.board_message}*")

        st.markdown("### The Strategic Narrative")
        with st.container(border=True):
            st.markdown(f"**📉 What Changed:** {final_brief.narrative_what_changed}")
            st.markdown(f"**⏳ Why Now:** {final_brief.narrative_why_now}")
            st.markdown(f"**🎯 Primary Move:** {final_brief.narrative_primary_move}")

        o1, o2, o3 = st.columns(3)
        with o1:
            with st.container(border=True):
                st.markdown("**🚀 Biggest Opportunity**")
                st.success(final_brief.biggest_opportunity)
        with o2:
            with st.container(border=True):
                st.markdown("**⚠️ Biggest Risk**")
                st.error(final_brief.biggest_risk)
        with o3:
            with st.container(border=True):
                st.markdown("**🚫 Do NOT Do**")
                st.warning(final_brief.do_not_do)

        if competitors:
            st.markdown("### 🏆 Competitor Benchmarks")
            for c in competitors:
                with st.container(border=True):
                    st.markdown(f"#### ⚔️ {c.competitor_name} — {c.threat_type}")
                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown("**Their Threat**"); st.error(c.threat_summary)
                    with cb:
                        st.markdown(f"**{company}'s Edge**"); st.success(c.advantage_summary)
                    st.markdown(f"**Counter-Move:** {c.recommended_response}")

        st.markdown("### Prioritized Strategic Directives")
        for i, action in enumerate(final_brief.prioritized_actions, 1):
            icon = "🔴" if action.framework == "STOP" else "🟢" if action.framework == "START" else "🔥"
            with st.container(border=True):
                st.markdown(f"#### #{i} {icon} **{action.framework}**: {action.action}")
                a1, a2 = st.columns(2)
                with a1:
                    st.markdown("**1. Evidence**");         st.info(f"*{action.evidence}*")
                    st.markdown("**2. Implication**");      st.warning(action.implication)
                    st.markdown("**3. Competitor Context**"); st.caption(action.competitor_context)
                with a2:
                    st.markdown("**4. Timeline**");         st.write(f"📅 {action.timeline}")
                    st.markdown("**5. Expected Impact**");  st.success(action.expected_impact)
                    st.markdown("**6. Risk**");             st.error(action.risk)

        st.divider()
        export = {
            "company": company,
            "pipeline": {"extracted": total, "passed": passed, "rate_pct": rate, "signals": len(signals)},
            "verified_facts": [vf.model_dump() for vf in verified_facts],
            "signals": [s.model_dump() for s in signals],
            "competitor_intel": [c.model_dump() for c in competitors],
            "board_brief": final_brief.model_dump(),
        }
        st.download_button(
            "Download Full Intelligence Package (JSON)",
            data=json.dumps(export, indent=2),
            file_name=f"{company}_board_brief.json",
            mime="application/json"
        )