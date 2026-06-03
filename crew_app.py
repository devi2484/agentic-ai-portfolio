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
llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.1-8b-instant", temperature=0.1)

st.set_page_config(page_title="Strategic Intelligence Engine", page_icon="♟️", layout="wide")
st.title("♟️ Strategic Intelligence Engine")
st.markdown("**Board-Grade Pipeline** · Entity Resolution · Root Cause Analysis · Hard-Gate Validation · Signal Detection")
st.divider()

# ==========================================
# 2. TRUST SCORING
# ==========================================
HIGH_TRUST_DOMAINS = [
    "reuters.com","bloomberg.com","cnbc.com","wsj.com","ft.com","sec.gov",
    "moneycontrol.com","economictimes.indiatimes.com","livemint.com",
    "businessstandard.com","thehindubusinessline.com","financialexpress.com",
    "bseindia.com","nseindia.com","sebi.gov.in","rbi.org.in",
]
MEDIUM_TRUST_DOMAINS = [
    "techcrunch.com","forbes.com","inc42.com","entrackr.com",
    "yourstory.com","themorningcontext.com","restofworld.org","fortune.com",
]
LOW_TRUST_DOMAINS = [
    "linkedin.com","reddit.com","quora.com","wikipedia.org",
    "medium.com","twitter.com","x.com","substack.com",
]
TRUST_SCORE_MAP = {"HIGH TRUST": 10, "MEDIUM TRUST": 6, "LOW TRUST": 2}

def evaluate_trust(url: str) -> str:
    domain = urlparse(url).netloc.lower().replace("www.", "")
    if any(h in domain for h in HIGH_TRUST_DOMAINS):   return "HIGH TRUST"
    if any(m in domain for m in MEDIUM_TRUST_DOMAINS): return "MEDIUM TRUST"
    if any(l in domain for l in LOW_TRUST_DOMAINS):    return "LOW TRUST"
    return "MEDIUM TRUST"

# ==========================================
# 3. DETERMINISTIC SCORING
# All scores are formula-based — never LLM-invented
# ==========================================
def calculate_confidence(trust_label: str, board_relevance: int, strategic_impact: int) -> int:
    """Formula: (trust*0.4) + (board_relevance*0.3) + (strategic_impact*0.3), scaled 0-100."""
    trust_score = TRUST_SCORE_MAP.get(trust_label.strip(), 5)
    raw = (trust_score * 0.4) + (board_relevance * 0.3) + (strategic_impact * 0.3)
    return int((raw / 10) * 100)

def calculate_health_score(verified_facts: list, signals: list, competitors: list) -> int:
    """
    Deterministic company health score.
    Base: 50. Adjusted by fact quality, signal urgency, competitor threats.
    """
    score = 50
    for f in verified_facts:
        if f.category == "Profitability":
            score += 5 if f.strategic_impact >= 9 else 2
        if f.category == "Growth":
            score += 4 if f.strategic_impact >= 9 else 1
        if f.category == "Competitive Advantage":
            score += 3
    for s in signals:
        if s.urgency == "IMMEDIATE":
            score -= 8
        elif s.urgency == "90-DAY":
            score -= 4
        if "Threat" in s.signal_type or "Erosion" in s.signal_type or "Compression" in s.signal_type:
            score -= 3
        if "Opportunity" in s.signal_type or "Strengthening" in s.signal_type:
            score += 3
    for c in competitors:
        if "Largest Threat" in c.threat_type or "Fastest Growing" in c.threat_type:
            score -= 5
    return max(0, min(100, score))

def calculate_report_confidence(verified_facts: list, total_facts: int) -> int:
    """Deterministic report confidence based on gate pass rate and average fact confidence."""
    if not verified_facts or total_facts == 0:
        return 20
    gate_rate = len(verified_facts) / total_facts
    avg_conf = sum(f.confidence for f in verified_facts) / len(verified_facts)
    return int((gate_rate * 0.4 + avg_conf / 100 * 0.6) * 100)

# ==========================================
# 4. SEARCH
# ==========================================
def run_enhanced_search(company: str) -> str:
    """5 targeted queries — max signal density per token."""
    queries = [
        f"{company} revenue profit margin earnings 2025",
        f"{company} market share competitor comparison 2025",
        f"{company} capital allocation acquisition fundraise 2025",
        f"{company} regulatory risk supply chain disruption 2025",
        f"{company} strategic pivot AI investment new market 2025",
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
# 5. JSON INVOKE — no tool-call schema, no 400 errors
# ==========================================
def invoke_json(prompt: str) -> dict:
    messages = [
        SystemMessage(content=(
            "You are a precise JSON-only responder. "
            "Output ONLY valid JSON. No markdown, no explanation, no code fences, no trailing commas."
        )),
        HumanMessage(content=prompt)
    ]
    resp = llm.invoke(messages)
    text = resp.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    text = text.strip().rstrip("```").strip()
    return json.loads(text)

# ==========================================
# 6. PYDANTIC MODELS — internal use only, never sent to Groq
# ==========================================
class EntityProfile(BaseModel):
    canonical_name: str
    industry: str
    sector: str
    business_model: str
    primary_market: str
    known_subsidiaries: str
    known_competitors: str
    contamination_warnings: str   # flags if search results may have mixed company data

class IntelligenceFact(BaseModel):
    category: str
    fact: str
    root_cause: str           # NEW: WHY did this happen?
    business_driver: str      # NEW: what underlying driver caused it?
    strategic_implication: str # NEW: what does this mean for the business?
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
    root_cause: str
    business_driver: str
    strategic_implication: str
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
    root_cause: str           # NEW: signal must be grounded in a root cause
    evidence_fact: str

class CompetitorIntel(BaseModel):
    competitor_name: str
    threat_type: str
    threat_summary: str
    root_cause_of_threat: str  # NEW: why are they gaining ground?
    advantage_summary: str
    recommended_response: str

class StrategicAction(BaseModel):
    framework: str
    evidence: str
    root_cause: str            # NEW: action grounded in root cause
    implication: str
    competitor_context: str
    action: str
    expected_impact: str
    risk: str
    timeline: str

class CEOBrief(BaseModel):
    company_health_score: int
    report_confidence: int
    entity_context: str        # NEW: confirms which company was analysed
    narrative_what_changed: str
    narrative_root_cause: str  # NEW: WHY did it change?
    narrative_why_now: str
    narrative_primary_move: str
    biggest_opportunity: str
    biggest_risk: str
    most_important_competitor: str  # NEW
    key_decision: str               # NEW: the one decision leadership must make
    do_not_do: str
    board_message: str
    prioritized_actions: List[StrategicAction]

# ==========================================
# 7. PIPELINE
#
# Search
# ↓
# Entity Resolution       ← NEW: prevents cross-company contamination
# ↓
# Researcher              ← enriched: root cause + business driver + implication
# ↓
# Hard-Gate Validation    ← programmatic, no LLM
# ↓  [only verified_facts proceed]
# Root Cause Analyser     ← NEW: deepens causal reasoning before signals
# ↓
# Competitor Intelligence ← grounded in root causes
# ↓
# Signal Detector         ← runs ONLY on validated + root-cause-enriched facts
# ↓
# Strategist              ← full Evidence→RootCause→Implication→Action chain
# ==========================================

FACT_CATEGORIES = [
    "Profitability", "Growth", "Competitive Threat",
    "Competitive Advantage", "Capital Allocation", "Strategic Shift"
]

# --- AGENT 1: Entity Resolution ---
def run_entity_resolution(company: str, raw_context: str) -> EntityProfile:
    prompt = f"""You are an Entity Resolution Specialist. Analyse the search results to identify exactly which company is being researched.

Return a JSON object:
{{
  "canonical_name": "official company name e.g. FSN E-Commerce Ventures Ltd (Nykaa)",
  "industry": "e.g. Beauty E-Commerce",
  "sector": "e.g. Consumer Discretionary",
  "business_model": "e.g. Inventory-led B2C e-commerce + owned brands",
  "primary_market": "e.g. India",
  "known_subsidiaries": "e.g. Nykaa Fashion, Nykaa Pro",
  "known_competitors": "e.g. Purplle, Reliance Beauty, Myntra Beauty",
  "contamination_warnings": "e.g. None detected OR: Some results may refer to [other company] — flag for researcher"
}}

Company queried: {company}

Search Context (first 1500 chars):
{raw_context[:1500]}"""
    try:
        data = invoke_json(prompt)
        return EntityProfile(**data)
    except Exception:
        return EntityProfile(
            canonical_name=company, industry="Unknown", sector="Unknown",
            business_model="Unknown", primary_market="Unknown",
            known_subsidiaries="Unknown", known_competitors="Unknown",
            contamination_warnings="Entity resolution failed — verify facts manually"
        )


# --- AGENT 2: Researcher (enriched with root cause) ---
def run_researcher(company: str, entity: EntityProfile, raw_context: str) -> List[IntelligenceFact]:
    prompt = f"""You are a Goldman Sachs Research Analyst. Extract decision-relevant intelligence for {entity.canonical_name}.

Company context:
- Industry: {entity.industry} | Sector: {entity.sector}
- Business model: {entity.business_model}
- Primary market: {entity.primary_market}
- Known competitors: {entity.known_competitors}
- Contamination warning: {entity.contamination_warnings}

CONTAMINATION GUARD: Only extract facts that are explicitly about {entity.canonical_name} or its subsidiaries ({entity.known_subsidiaries}). Reject any fact that appears to be about a different company.

GOLDEN RULE: "If this fact disappeared tomorrow, would the board care?" If NO — reject it.

Return a JSON object:
{{
  "facts": [
    {{
      "category": "one of: {', '.join(FACT_CATEGORIES)}",
      "fact": "specific verifiable fact with numbers/dates where present",
      "root_cause": "WHY did this happen? What is the underlying cause?",
      "business_driver": "what business mechanism drove this outcome?",
      "strategic_implication": "what does this mean for the company's competitive position?",
      "why_it_matters": "why would the board care if this disappeared?",
      "board_relevance": 9,
      "strategic_impact": 9,
      "source_url": "https://...",
      "source_trust": "HIGH TRUST or MEDIUM TRUST or LOW TRUST",
      "date_signal": "Q1 2025 or Undated",
      "competitor_context": "vs [NamedCompetitor] or No benchmark available"
    }}
  ]
}}

Return EXACTLY 6 facts — one per category: {', '.join(FACT_CATEGORIES)}

HARD REJECT LIST:
- Founding dates, company history, awards, PR, executive bios, social media
- Product launches older than 18 months
- Facts about companies other than {entity.canonical_name}
- Generic industry trends with no company-specific data
- Any fact where board_relevance < 8 OR strategic_impact < 8

NEVER invent numbers. Qualitative language only when no hard data exists.
source_trust: copy EXACTLY from the TRUST label in context (HIGH TRUST / MEDIUM TRUST / LOW TRUST).

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
        st.warning(f"Researcher error: {e}")
        return []


# --- GATE: Hard Validation (programmatic — no LLM) ---
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
            category=f.category, fact=f.fact,
            root_cause=f.root_cause, business_driver=f.business_driver,
            strategic_implication=f.strategic_implication,
            why_it_matters=f.why_it_matters,
            source_url=f.source_url, source_trust=f.source_trust,
            date_signal=f.date_signal, competitor_context=f.competitor_context,
            board_relevance=f.board_relevance, strategic_impact=f.strategic_impact,
            confidence=confidence,
        ))
    return verified


# --- AGENT 3: Competitor Intelligence (grounded in root causes) ---
def run_competitor_intel(company: str, entity: EntityProfile, raw_context: str) -> List[CompetitorIntel]:
    prompt = f"""You are a Competitive Intelligence Specialist analysing {entity.canonical_name}.

Known competitors: {entity.known_competitors}
Industry: {entity.industry} | Market: {entity.primary_market}

Identify up to 3 NAMED competitors. Use real company names from the known competitors list or from the search context.

Return a JSON object:
{{
  "competitors": [
    {{
      "competitor_name": "exact company name",
      "threat_type": "one of: Fastest Growing, Largest Threat, Weakening Moat, Strengthening Moat, Competitive Surprise, Most Likely Future Threat",
      "threat_summary": "specific threat with data",
      "root_cause_of_threat": "WHY are they gaining ground? What is driving their momentum?",
      "advantage_summary": "where {entity.canonical_name} still leads",
      "recommended_response": "specific counter-move naming markets, product lines, or channels"
    }}
  ]
}}

FORBIDDEN responses: improve innovation, focus on customers, optimize operations, review strategy, increase efficiency.

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
        st.warning(f"Competitor intel error: {e}")
        return []


# --- AGENT 4: Signal Detector (validated facts only, root-cause aware) ---
def run_signal_detector(company: str, verified_facts: List[ValidatedFact]) -> List[StrategicSignal]:
    if not verified_facts:
        return []
    fact_text = "\n".join([
        f"[{f.category}] FACT: {f.fact} | ROOT CAUSE: {f.root_cause} | IMPLICATION: {f.strategic_implication}"
        for f in verified_facts
    ])
    prompt = f"""You are a Strategic Signal Detector for {company}.
Identify inflection points from ONLY the validated facts below. Every signal must reference a root cause.

Return a JSON object:
{{
  "signals": [
    {{
      "signal_type": "one of: Emerging Threat, Emerging Opportunity, Strategic Inflection, Capital Shift, Competitive Surprise, Moat Erosion, Moat Strengthening, Regulatory Risk, Margin Compression, Pricing Pressure, Technology Disruption",
      "signal": "specific inflection point — what is changing?",
      "urgency": "one of: IMMEDIATE, 90-DAY, 6-MONTH, WATCH",
      "root_cause": "the underlying driver that makes this a signal, not just a data point",
      "evidence_fact": "the exact validated fact that triggered this signal"
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
        st.warning(f"Signal detector error: {e}")
        return []


# --- AGENT 5: Strategist (full causal chain) ---
def run_strategist(
    company: str, entity: EntityProfile,
    verified_facts: List[ValidatedFact], signals: List[StrategicSignal],
    competitors: List[CompetitorIntel],
    health_score: int, report_confidence: int
) -> Optional[CEOBrief]:

    fact_text = "\n".join([
        f"[{f.category} | {f.confidence}%] {f.fact}\n  Root Cause: {f.root_cause}\n  Implication: {f.strategic_implication}"
        for f in verified_facts
    ]) if verified_facts else f"INTELLIGENCE FAILURE: No verified data found for {entity.canonical_name}."

    signal_text = "\n".join([
        f"[{s.signal_type}|{s.urgency}] {s.signal} | Root Cause: {s.root_cause}"
        for s in signals
    ]) or "No signals detected."

    competitor_text = "\n".join([
        f"[{c.threat_type}] {c.competitor_name}: {c.threat_summary} | Why they're gaining: {c.root_cause_of_threat}"
        for c in competitors
    ]) or "No competitor data."

    prompt = f"""You are a McKinsey Senior Partner presenting to the Board of {entity.canonical_name}.
Company: {entity.canonical_name} | Industry: {entity.industry} | Market: {entity.primary_market}

This is a board memo. Not an MBA essay. Not a summary. Executive intelligence only.

MANDATORY RECOMMENDATION CHAIN for every action:
Evidence → Root Cause → Strategic Implication → Competitive Context → Action → Expected Impact → Risk

Return a JSON object with EXACTLY this structure:
{{
  "company_health_score": {health_score},
  "report_confidence": {report_confidence},
  "entity_context": "confirm: analysing {entity.canonical_name}, {entity.industry}, {entity.primary_market}",
  "narrative_what_changed": "specific recent shift — with evidence",
  "narrative_root_cause": "WHY did this change happen? What is the underlying driver?",
  "narrative_why_now": "specific catalyst demanding action now, not in 6 months",
  "narrative_primary_move": "single most important strategic pivot — hyper-specific, names market/product/channel",
  "biggest_opportunity": "highest-upside move supported by evidence",
  "biggest_risk": "most dangerous unaddressed threat",
  "most_important_competitor": "the one competitor leadership must respond to and why",
  "key_decision": "the single decision the board must make in the next 90 days",
  "do_not_do": "most tempting but strategically wrong move given evidence",
  "board_message": "3 sentences: urgency + root-cause insight + call to action. No generics.",
  "prioritized_actions": [
    {{
      "framework": "STOP or START or DOUBLE DOWN",
      "evidence": "exact verified fact",
      "root_cause": "why this fact signals a need for action",
      "implication": "what changes competitively if this is ignored",
      "competitor_context": "how a named competitor is positioned here",
      "action": "specific directive — names market, product line, channel, or supply chain node",
      "expected_impact": "qualitative outcome — NEVER invent dollar values or percentages",
      "risk": "primary risk if action is taken or ignored",
      "timeline": "90 Days or 6 Months or Q3 2025 — must be future-dated"
    }}
  ]
}}

HARD RULES:
- EXACTLY 3 prioritized_actions, ranked highest strategic impact first
- NEVER invent dollar values, percentages, market share, or revenue figures
- Actions must name specific markets, products, channels, or nodes
- FORBIDDEN action language: improve innovation, focus on customers, optimize operations, review strategy, increase efficiency, enhance marketing, explore opportunities
- company_health_score: use exactly {health_score}
- report_confidence: use exactly {report_confidence}

Verified Evidence:
{fact_text}

Strategic Signals:
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
        data["company_health_score"] = health_score
        data["report_confidence"] = report_confidence
        return CEOBrief(**data)
    except Exception as e:
        st.error(f"Strategist error: {e}")
        return None


# ==========================================
# 8. STREAMLIT UI
# ==========================================
company = st.text_input("Target Company:", placeholder="e.g. Zomato, Reliance, Tesla, Nykaa...")

if st.button("Run Strategic Analysis", type="primary"):
    if not company:
        st.error("Please enter a company name.")
    else:
        with st.status(f"Compiling Board Intelligence on {company}...", expanded=True) as status:

            # STEP 1 — Search
            st.write("📡 Search — competitive intelligence queries...")
            raw_context = run_enhanced_search(company)
            if not raw_context:
                st.error("Search returned no data.")
                st.stop()

            time.sleep(3)

            # STEP 2 — Entity Resolution
            st.write("🔍 Entity Resolution — identifying company, industry, contamination risks...")
            entity = run_entity_resolution(company, raw_context)
            st.write(f"   → {entity.canonical_name} | {entity.industry} | {entity.primary_market}")
            if "failed" not in entity.contamination_warnings.lower() and entity.contamination_warnings != "None detected":
                st.warning(f"⚠️ Contamination warning: {entity.contamination_warnings}")

            time.sleep(3)

            # STEP 3 — Research (with entity context + root cause)
            st.write("📊 Researcher — extracting strategic facts with root cause analysis...")
            raw_facts = run_researcher(company, entity, raw_context[:3000])

            # STEP 4 — Hard Gate (programmatic)
            st.write("🔒 Hard-Gate Validation — deterministic confidence scoring...")
            verified_facts = run_hard_gate_validation(raw_facts)
            st.write(f"   → {len(raw_facts)} extracted · {len(verified_facts)} passed gate")

            time.sleep(4)

            # STEP 5 — Competitor Intel
            st.write("🎯 Competitor Intelligence — root-cause analysis of threats...")
            competitors = run_competitor_intel(company, entity, raw_context[:2000])

            time.sleep(4)

            # STEP 6 — Signal Detector (validated facts only)
            st.write("🔭 Signal Detector — inflection points from verified facts only...")
            signals = run_signal_detector(company, verified_facts)

            # STEP 7 — Deterministic scores before strategist
            health_score = calculate_health_score(verified_facts, signals, competitors)
            report_confidence = calculate_report_confidence(verified_facts, len(raw_facts))

            time.sleep(4)

            # STEP 8 — Strategist
            st.write("📋 Strategist — board brief with full causal chain...")
            final_brief = run_strategist(
                company, entity, verified_facts, signals, competitors,
                health_score, report_confidence
            )

            status.update(label="Analysis Complete", state="complete")

        if not final_brief:
            st.error("Strategist failed. Try again.")
            st.stop()

        # ==========================================
        # DISPLAY
        # ==========================================

        # Pipeline Stats
        st.subheader("🛡️ Intelligence Pipeline")
        total = len(raw_facts)
        passed = len(verified_facts)
        rate = int(passed / total * 100) if total else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Facts Extracted", total)
        m2.metric("Passed Hard Gate", passed)
        m3.metric("Gate Pass Rate", f"{rate}%")
        m4.metric("Signals", len(signals))
        m5.metric("Competitors Mapped", len(competitors))

        with st.expander("🔬 Full Pipeline Detail — Facts, Rejections, Root Causes, Signals"):

            # Entity Profile
            st.markdown("**🏢 Entity Profile**")
            with st.container(border=True):
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown(f"**Company:** {entity.canonical_name}")
                    st.markdown(f"**Industry:** {entity.industry} | **Sector:** {entity.sector}")
                    st.markdown(f"**Business Model:** {entity.business_model}")
                with ec2:
                    st.markdown(f"**Primary Market:** {entity.primary_market}")
                    st.markdown(f"**Subsidiaries:** {entity.known_subsidiaries}")
                    st.markdown(f"**Known Competitors:** {entity.known_competitors}")
                if entity.contamination_warnings and entity.contamination_warnings != "None detected":
                    st.warning(f"⚠️ {entity.contamination_warnings}")

            # Verified Facts
            st.divider()
            st.markdown("**✅ Verified Facts (passed all gates)**")
            for vf in verified_facts:
                with st.container(border=True):
                    st.success(f"**[{vf.category} | {vf.confidence}% | {vf.source_trust} | {vf.date_signal}]** {vf.fact}")
                    st.markdown(f"🔍 **Root Cause:** {vf.root_cause}")
                    st.markdown(f"⚙️ **Business Driver:** {vf.business_driver}")
                    st.markdown(f"🎯 **Strategic Implication:** {vf.strategic_implication}")
                    st.caption(f"Competitor context: {vf.competitor_context} | Source: {vf.source_url}")

            # Rejected Facts
            st.divider()
            st.markdown("**❌ Rejected Facts**")
            rejected = [f for f in raw_facts if not any(vf.fact == f.fact for vf in verified_facts)]
            for rf in rejected:
                conf = calculate_confidence(rf.source_trust, rf.board_relevance, rf.strategic_impact)
                reasons = []
                if rf.board_relevance < 8:         reasons.append(f"board_relevance={rf.board_relevance}")
                if rf.strategic_impact < 8:         reasons.append(f"strategic_impact={rf.strategic_impact}")
                if "LOW TRUST" in rf.source_trust:  reasons.append("LOW TRUST source")
                if conf < 70:                        reasons.append(f"confidence={conf}%")
                if rf.date_signal == "Undated":      reasons.append("Undated + non-HIGH-TRUST")
                st.error(f"**[{rf.category} | {conf}%]** {rf.fact}\n\n*Rejected: {' · '.join(reasons) or 'gate criteria'}*")

            # Signals
            if signals:
                st.divider()
                st.markdown("**🔭 Strategic Signals (from validated facts only)**")
                for s in signals:
                    icon = "🔴" if s.urgency == "IMMEDIATE" else "🟡" if s.urgency == "90-DAY" else "🟢"
                    with st.container(border=True):
                        st.info(f"{icon} **[{s.signal_type} | {s.urgency}]** {s.signal}")
                        st.markdown(f"🔍 **Root Cause:** {s.root_cause}")
                        st.caption(f"Evidence: {s.evidence_fact}")

            # Competitor Intel
            if competitors:
                st.divider()
                st.markdown("**🎯 Competitor Intelligence**")
                for c in competitors:
                    st.warning(
                        f"**[{c.threat_type}] {c.competitor_name}:** {c.threat_summary}\n\n"
                        f"🔍 *Why they're gaining:* {c.root_cause_of_threat}"
                    )

        # Board Brief
        st.divider()
        h1, h2, h3 = st.columns([3, 1, 1])
        with h1: st.header(f"Board-Level Strategic Brief — {entity.canonical_name.upper()}")
        with h2: st.metric("Health Score", f"{final_brief.company_health_score}/100")
        with h3: st.metric("Report Confidence", f"{final_brief.report_confidence}%")

        st.caption(f"Entity confirmed: {final_brief.entity_context}")

        # Board Message
        st.markdown("### 📢 Board Message")
        with st.container(border=True):
            st.markdown(f"*{final_brief.board_message}*")

        # Key Decision callout
        st.markdown("### ⚡ Key Decision for Leadership")
        with st.container(border=True):
            st.error(f"**{final_brief.key_decision}**")

        # Strategic Narrative
        st.markdown("### The Strategic Narrative")
        with st.container(border=True):
            st.markdown(f"**📉 What Changed:** {final_brief.narrative_what_changed}")
            st.markdown(f"**🔍 Root Cause:** {final_brief.narrative_root_cause}")
            st.markdown(f"**⏳ Why Now (Catalyst):** {final_brief.narrative_why_now}")
            st.markdown(f"**🎯 Primary Move:** {final_brief.narrative_primary_move}")

        # Opp / Risk / Do Not Do
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

        # Competitor Benchmarks
        if competitors:
            st.markdown("### 🏆 Competitor Benchmarks")
            st.info(f"**Most Important Competitor:** {final_brief.most_important_competitor}")
            for c in competitors:
                with st.container(border=True):
                    st.markdown(f"#### ⚔️ {c.competitor_name} — {c.threat_type}")
                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown("**Their Threat**")
                        st.error(c.threat_summary)
                        st.caption(f"Why gaining: {c.root_cause_of_threat}")
                    with cb:
                        st.markdown(f"**{entity.canonical_name}'s Edge**")
                        st.success(c.advantage_summary)
                    st.markdown(f"**Counter-Move:** {c.recommended_response}")

        # Prioritized Actions
        st.markdown("### Prioritized Strategic Directives")
        for i, action in enumerate(final_brief.prioritized_actions, 1):
            icon = "🔴" if action.framework == "STOP" else "🟢" if action.framework == "START" else "🔥"
            with st.container(border=True):
                st.markdown(f"#### #{i} {icon} **{action.framework}**: {action.action}")
                a1, a2 = st.columns(2)
                with a1:
                    st.markdown("**1. Evidence**");              st.info(f"*{action.evidence}*")
                    st.markdown("**2. Root Cause**");            st.markdown(f"🔍 {action.root_cause}")
                    st.markdown("**3. Implication**");           st.warning(action.implication)
                    st.markdown("**4. Competitor Context**");    st.caption(action.competitor_context)
                with a2:
                    st.markdown("**5. Timeline**");              st.write(f"📅 {action.timeline}")
                    st.markdown("**6. Expected Impact**");       st.success(action.expected_impact)
                    st.markdown("**7. Risk**");                  st.error(action.risk)

        # Export
        st.divider()
        export = {
            "company": company,
            "entity_profile": entity.model_dump(),
            "pipeline": {"extracted": total, "passed": passed, "rate_pct": rate,
                         "signals": len(signals), "competitors": len(competitors)},
            "verified_facts": [vf.model_dump() for vf in verified_facts],
            "signals": [s.model_dump() for s in signals],
            "competitor_intel": [c.model_dump() for c in competitors],
            "board_brief": final_brief.model_dump(),
        }
        st.download_button(
            "Download Full Intelligence Package (JSON)",
            data=json.dumps(export, indent=2),
            file_name=f"{company.replace(' ','_')}_board_brief.json",
            mime="application/json"
        )