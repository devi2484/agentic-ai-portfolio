import os
import json
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from urllib.parse import urlparse

# ==========================================
# 1. SETUP
# ==========================================
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY") or st.secrets.get("GROQ_KEY", "")
llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.1-8b-instant", temperature=0.1)

st.set_page_config(page_title="Strategic Intelligence Engine", page_icon="♟️", layout="wide")
st.title("♟️ Strategic Intelligence Engine")
st.markdown("**Decision-Support Platform** · Evidence-Backed · Root Cause Reasoning · Options Analysis")
st.divider()

# ==========================================
# 2. TRUST & SCORING
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

MIN_VERIFIED_FACTS   = 2
MIN_REPORT_CONFIDENCE = 50
ENTITY_CONFIDENCE_THRESHOLD = 60

def evaluate_trust(url: str) -> str:
    domain = urlparse(url).netloc.lower().replace("www.", "")
    if any(h in domain for h in HIGH_TRUST_DOMAINS):   return "HIGH TRUST"
    if any(m in domain for m in MEDIUM_TRUST_DOMAINS): return "MEDIUM TRUST"
    if any(l in domain for l in LOW_TRUST_DOMAINS):    return "LOW TRUST"
    return "MEDIUM TRUST"

# ==========================================
# 3. DETERMINISTIC SCORING
# ==========================================
def calculate_confidence(trust_label: str, board_relevance: int, strategic_impact: int) -> int:
    trust_score = TRUST_SCORE_MAP.get(trust_label.strip(), 5)
    raw = (trust_score * 0.4) + (board_relevance * 0.3) + (strategic_impact * 0.3)
    return int((raw / 10) * 100)

def calculate_entity_confidence(entity) -> tuple[int, str]:
    score = 100
    reasons = []
    if entity.industry == "Unknown":
        score -= 20; reasons.append("industry unknown")
    if entity.sector == "Unknown":
        score -= 10; reasons.append("sector unknown")
    if entity.business_model == "Unknown":
        score -= 15; reasons.append("business model unknown")
    if entity.primary_market == "Unknown":
        score -= 10; reasons.append("primary market unknown")
    if entity.known_competitors == "Unknown":
        score -= 10; reasons.append("competitors unknown")
    contamination = entity.contamination_warnings.lower()
    if "failed" in contamination:
        score -= 25; reasons.append("entity resolution failed")
    elif "none" not in contamination and contamination != "":
        score -= 20; reasons.append(f"contamination risk: {entity.contamination_warnings}")
    explanation = f"Entity confidence {score}%"
    if reasons:
        explanation += f" — Issues: {', '.join(reasons)}"
    return max(0, score), explanation

def calculate_health_score(verified_facts: list, signals: list, competitors: list) -> int:
    score = 50
    for f in verified_facts:
        if f.category == "Profitability":   score += 5 if f.strategic_impact >= 9 else 2
        if f.category == "Growth":          score += 4 if f.strategic_impact >= 9 else 1
        if f.category == "Competitive Advantage": score += 3
        if f.category == "Competitive Threat":    score -= 2
    for s in signals:
        if s.urgency == "IMMEDIATE":  score -= 8
        elif s.urgency == "90-DAY":   score -= 4
        if any(t in s.signal_type for t in ["Threat","Erosion","Compression","Regulatory","Disruption"]): score -= 3
        if any(t in s.signal_type for t in ["Opportunity","Strengthening","Inflection"]): score += 3
    for c in competitors:
        if any(t in c.threat_type for t in ["Largest Threat","Fastest Growing"]): score -= 5
    return max(0, min(100, score))

def calculate_report_confidence(verified_facts: list, total_facts: int) -> int:
    if not verified_facts or total_facts == 0:
        return 15
    gate_rate = len(verified_facts) / total_facts
    avg_conf  = sum(f.confidence for f in verified_facts) / len(verified_facts)
    return int((gate_rate * 0.4 + avg_conf / 100 * 0.6) * 100)

def get_evidence_sufficiency(verified_facts: list, report_confidence: int) -> tuple[bool, str]:
    if len(verified_facts) < MIN_VERIFIED_FACTS:
        return False, f"Only {len(verified_facts)} fact(s) passed validation. Recommendations will be speculative."
    if report_confidence < MIN_REPORT_CONFIDENCE:
        return False, f"Report confidence {report_confidence}% is below threshold. Evidence quality is low."
    return True, "Evidence sufficient for reliable recommendations."

# ==========================================
# 4. SEARCH
# ==========================================
def run_enhanced_search(company: str) -> str:
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
# 5. JSON INVOKE
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
# 6. PYDANTIC MODELS (Updated for Strategy Layer)
# ==========================================
class EntityProfile(BaseModel):
    canonical_name: str
    industry: str
    sector: str
    business_model: str
    primary_market: str
    known_subsidiaries: str
    known_competitors: str
    contamination_warnings: str

class IntelligenceFact(BaseModel):
    category: str
    fact: str
    root_cause: str
    business_driver: str
    strategic_implication: str
    is_structural: str
    why_it_matters: str
    decision_relevance: str
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
    is_structural: str
    why_it_matters: str
    decision_relevance: str
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
    root_cause: str
    implication: str
    evidence_fact: str
    is_restatement: bool = False

class CompetitorIntel(BaseModel):
    competitor_name: str
    threat_type: str
    threat_summary: str
    root_cause_of_threat: str
    capability_driving_advantage: str
    structural_or_temporary: str
    advantage_summary: str
    recommended_response: str

# NEW STRATEGY LAYER MODELS
class RootCauseAnalysis(BaseModel):
    fact: str
    root_cause: str
    temporary_or_structural: str
    confidence: str

class StrategicTheme(BaseModel):
    theme: str
    supporting_facts: List[str]
    why_it_matters: str

class CompetitiveImplication(BaseModel):
    theme: str
    competitive_implication: str
    winning_capability: str
    risk_capability: str

class StrategicOption(BaseModel):
    option_name: str
    description: str
    benefits: str
    risks: str
    resource_requirements: str
    strategic_fit: str

class OptionsAnalysis(BaseModel):
    theme: str
    options: List[StrategicOption]

class RecommendedAction(BaseModel):
    action: str
    why_this_action: str
    why_now: str
    why_not_alternatives: str
    supporting_evidence: List[str]
    supporting_theme: str
    expected_benefit: str
    key_risk: str

class ChiefStrategyBrief(BaseModel):
    company_health_score: int
    report_confidence: int
    evidence_sufficiency: str
    entity_context: str
    root_causes: List[RootCauseAnalysis]
    strategic_themes: List[StrategicTheme]
    competitive_implications: List[CompetitiveImplication]
    options_considered: List[OptionsAnalysis]
    recommended_actions: List[RecommendedAction]

# ==========================================
# 7. PIPELINE AGENTS
# ==========================================
FACT_CATEGORIES = [
    "Profitability", "Growth", "Competitive Threat",
    "Competitive Advantage", "Capital Allocation", "Strategic Shift"
]

def run_entity_resolution(company: str, raw_context: str) -> EntityProfile:
    prompt = f"""You are an Entity Resolution Specialist. Identify exactly which company the search results describe.

Return a JSON object:
{{
  "canonical_name": "official registered name e.g. FSN E-Commerce Ventures Ltd (Nykaa)",
  "industry": "specific industry",
  "sector": "sector",
  "business_model": "how it makes money",
  "primary_market": "main geography",
  "known_subsidiaries": "e.g. Nykaa Fashion — or Unknown",
  "known_competitors": "e.g. Purplle — or Unknown",
  "contamination_warnings": "None detected — OR describe results about a different entity"
}}

Company queried: {company}
Search context: {raw_context[:1500]}"""
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

def run_researcher(company: str, entity: EntityProfile, raw_context: str,
                   evidence_warning: str) -> List[IntelligenceFact]:
    prompt = f"""You are a Goldman Sachs Research Analyst extracting decision-relevant intelligence for {entity.canonical_name}.

Entity context:
- Industry: {entity.industry} | Sector: {entity.sector} | Market: {entity.primary_market}
- Business model: {entity.business_model}
- Known competitors: {entity.known_competitors}

PRIMARY FILTER — Before accepting any fact, answer: "Could this information change a decision?" If NO — reject it.

Return a JSON object:
{{
  "facts": [
    {{
      "category": "one of: {', '.join(FACT_CATEGORIES)}",
      "fact": "specific verifiable fact with numbers/dates where present",
      "root_cause": "WHY did this happen? Underlying cause, not description.",
      "business_driver": "what business mechanism produced this outcome?",
      "strategic_implication": "what does this mean for competitive position?",
      "is_structural": "Structural (permanent) or Temporary (blip) — explain briefly",
      "why_it_matters": "why would this change a board decision?",
      "decision_relevance": "what specific decision could this change?",
      "board_relevance": 9,
      "strategic_impact": 9,
      "source_url": "https://...",
      "source_trust": "HIGH TRUST or MEDIUM TRUST or LOW TRUST",
      "date_signal": "Q1 2025 or Undated",
      "competitor_context": "vs [NamedCompetitor] or No benchmark available"
    }}
  ]
}}

Return EXACTLY 6 facts — one per category.
HARD REJECT: Founding dates, awards, PR, old launches, generic trends.
{f"DATA WARNING: {evidence_warning}" if evidence_warning else ""}

Raw Context:
{raw_context}"""
    try:
        data = invoke_json(prompt)
        facts = []
        for f in data.get("facts", []):
            try: facts.append(IntelligenceFact(**f))
            except Exception: continue
        return facts
    except Exception as e:
        st.warning(f"Researcher error: {e}")
        return []

def run_hard_gate_validation(facts: List[IntelligenceFact]) -> List[ValidatedFact]:
    verified = []
    for f in facts:
        if f.board_relevance < 8 or f.strategic_impact < 8: continue
        if "LOW TRUST" in f.source_trust: continue
        confidence = calculate_confidence(f.source_trust, f.board_relevance, f.strategic_impact)
        if confidence < 70: continue
        if f.date_signal == "Undated" and "HIGH TRUST" not in f.source_trust: continue
        verified.append(ValidatedFact(
            category=f.category, fact=f.fact, root_cause=f.root_cause, 
            business_driver=f.business_driver, strategic_implication=f.strategic_implication,
            is_structural=f.is_structural, why_it_matters=f.why_it_matters, 
            decision_relevance=f.decision_relevance, source_url=f.source_url, 
            source_trust=f.source_trust, date_signal=f.date_signal, 
            competitor_context=f.competitor_context, board_relevance=f.board_relevance, 
            strategic_impact=f.strategic_impact, confidence=confidence,
        ))
    return verified

def run_competitor_intel(company: str, entity: EntityProfile, raw_context: str, evidence_sufficient: bool) -> List[CompetitorIntel]:
    prompt = f"""You are a Competitive Intelligence Specialist analysing {entity.canonical_name}.
Known competitors: {entity.known_competitors}

Explain WHY competitors are succeeding or struggling. Focus on capabilities.
Return a JSON object:
{{
  "competitors": [
    {{
      "competitor_name": "exact company name",
      "threat_type": "one of: Fastest Growing, Largest Threat, Weakening Moat, Strengthening Moat, Competitive Surprise, Most Likely Future Threat",
      "threat_summary": "what specific move signals threat",
      "root_cause_of_threat": "WHY are they gaining? Underlying capability",
      "capability_driving_advantage": "tech/distribution/cost/brand/network",
      "structural_or_temporary": "structural shift or temporary surge?",
      "advantage_summary": "where {entity.canonical_name} still leads",
      "recommended_response": "specific counter-move"
    }}
  ]
}}
Raw Context:
{raw_context}"""
    try:
        data = invoke_json(prompt)
        comps = []
        for c in data.get("competitors", []):
            try: comps.append(CompetitorIntel(**c))
            except Exception: continue
        return comps
    except Exception as e:
        st.warning(f"Competitor intel error: {e}")
        return []

def run_signal_detector(company: str, verified_facts: List[ValidatedFact], evidence_sufficient: bool) -> List[StrategicSignal]:
    if not verified_facts: return []
    fact_text = "\n".join([f"[{f.category}] FACT: {f.fact}\nRoot Cause: {f.root_cause}\nImplication: {f.strategic_implication}" for f in verified_facts])

    prompt = f"""You are a Strategic Signal Detector for {company}.
CRITICAL RULE: A signal must explain WHY a fact matters — not restate the fact.
Return a JSON object:
{{
  "signals": [
    {{
      "signal_type": "one of: Emerging Threat, Emerging Opportunity, Strategic Inflection, Capital Shift, Competitive Surprise, Moat Erosion, Moat Strengthening, Regulatory Risk, Margin Compression, Pricing Pressure, Technology Disruption",
      "signal": "what is changing — the inflection, not the fact",
      "urgency": "IMMEDIATE, 90-DAY, 6-MONTH, WATCH",
      "root_cause": "underlying driver",
      "implication": "specific decision affected",
      "evidence_fact": "the validated fact that triggered this",
      "is_restatement": false
    }}
  ]
}}
Validated Facts:
{fact_text}"""
    try:
        data = invoke_json(prompt)
        signals = []
        for s in data.get("signals", []):
            try:
                sig = StrategicSignal(**s)
                if not sig.is_restatement: signals.append(sig)
            except Exception: continue
        return signals
    except Exception as e:
        st.warning(f"Signal detector error: {e}")
        return []


# --- AGENT 5: The Strategy Layer (CSO Level) ---
def run_strategist(
    company: str, entity: EntityProfile, verified_facts: List[ValidatedFact], 
    signals: List[StrategicSignal], competitors: List[CompetitorIntel],
    health_score: int, report_confidence: int, evidence_sufficient: bool, sufficiency_message: str
) -> Optional[ChiefStrategyBrief]:

    fact_text = "\n".join([f"- {f.fact} (Confidence: {f.confidence}%)" for f in verified_facts]) if verified_facts else "INSUFFICIENT EVIDENCE."
    signal_text = "\n".join([f"- {s.signal} ({s.urgency})" for s in signals]) or "No validated signals."
    competitor_text = "\n".join([f"- {c.competitor_name}: {c.threat_summary}" for c in competitors]) or "No competitor data."

    sufficiency_instruction = f'  "evidence_sufficiency": "SPECULATIVE — {sufficiency_message}",' if not evidence_sufficient else f'  "evidence_sufficiency": "Sufficient — {sufficiency_message}",'

    prompt = f"""You are a Fortune 500 Chief Strategy Officer, former McKinsey Senior Partner, and Board Advisor.
Your job is NOT to summarize facts. Your job is to transform verified intelligence into executive decisions.

You will receive:
1. Verified facts
2. Strategic signals
3. Competitor intelligence

Use ONLY this information. Never invent facts. Never invent metrics.

==================================================
CORE PRINCIPLE
==================================================
Facts are not strategy. Signals are not strategy.
Strategy begins when you determine WHY something happened, WHY it matters, and WHAT management should do.

==================================================
STEP 1 — ROOT CAUSE ANALYSIS
==================================================
For every verified fact determine: What caused it? Temporary or structural? Which capability created it?

==================================================
STEP 2 — STRATEGIC THEME EXTRACTION
==================================================
Group facts into themes (e.g., Margin Expansion, Moat Erosion, Capital Allocation Shift).
Themes must emerge from evidence.

==================================================
STEP 3 — COMPETITIVE IMPLICATIONS
==================================================
For each theme determine: Which competitor benefits/is threatened? Which capability becomes winning/risk?

==================================================
STEP 4 — OPTIONS ANALYSIS
==================================================
Before creating recommendations generate options (A, B, C). Evaluate benefits, risks, resource requirements, and fit. Evaluate alternatives first.

==================================================
STEP 5 — RECOMMENDATION GENERATION
==================================================
Generate recommendations only after evaluating alternatives.
Every recommendation must answer: Why this action? Why now? Why not the alternatives?

FORBIDDEN RECOMMENDATIONS: Improve innovation, Improve efficiency, Focus on customers, Increase marketing.
If the recommendation could be given to any random company without modification: Reject it.

==================================================
JSON OUTPUT FORMAT (STRICT)
==================================================
Escape all line breaks as '\\n'. Do NOT escape single quotes (\\'). 

{{
  "company_health_score": {health_score},
  "report_confidence": {report_confidence},
{sufficiency_instruction}
  "entity_context": "{entity.canonical_name} | {entity.industry} | {entity.primary_market}",
  "root_causes": [
    {{
      "fact": "",
      "root_cause": "",
      "temporary_or_structural": "",
      "confidence": ""
    }}
  ],
  "strategic_themes": [
    {{
      "theme": "",
      "supporting_facts": [],
      "why_it_matters": ""
    }}
  ],
  "competitive_implications": [
    {{
      "theme": "",
      "competitive_implication": "",
      "winning_capability": "",
      "risk_capability": ""
    }}
  ],
  "options_considered": [
    {{
      "theme": "",
      "options": [
        {{
          "option_name": "",
          "description": "",
          "benefits": "",
          "risks": "",
          "resource_requirements": "",
          "strategic_fit": ""
        }}
      ]
    }}
  ],
  "recommended_actions": [
    {{
      "action": "",
      "why_this_action": "",
      "why_now": "",
      "why_not_alternatives": "",
      "supporting_evidence": [],
      "supporting_theme": "",
      "expected_benefit": "",
      "key_risk": ""
    }}
  ]
}}

Verified Evidence:
{fact_text}

Strategic Signals:
{signal_text}

Competitor Intelligence:
{competitor_text}"""

    try:
        data = invoke_json(prompt)
        return ChiefStrategyBrief(**data)
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
        with st.status(f"Compiling Decision Intelligence on {company}...", expanded=True) as status:

            st.write("📡 Search...")
            raw_context = run_enhanced_search(company)
            if not raw_context:
                st.error("Search returned no data.")
                st.stop()
            time.sleep(2)

            st.write("🔍 Entity Resolution...")
            entity = run_entity_resolution(company, raw_context)
            entity_conf, entity_conf_msg = calculate_entity_confidence(entity)
            if entity_conf < ENTITY_CONFIDENCE_THRESHOLD:
                st.warning(f"⚠️ **Low Entity Confidence** — {entity_conf_msg}")
            time.sleep(2)

            st.write("📊 Researcher — extracting facts & structural drivers...")
            raw_facts = run_researcher(company, entity, raw_context[:3000], "")

            st.write("🔒 Hard-Gate Validation...")
            verified_facts = run_hard_gate_validation(raw_facts)

            report_confidence_prelim = calculate_report_confidence(verified_facts, len(raw_facts))
            evidence_sufficient, sufficiency_message = get_evidence_sufficiency(verified_facts, report_confidence_prelim)
            if not evidence_sufficient:
                st.warning(f"⚠️ **Evidence Warning:** {sufficiency_message}")
            time.sleep(2)

            st.write("🎯 Competitor Intelligence...")
            competitors = run_competitor_intel(company, entity, raw_context[:2000], evidence_sufficient)
            time.sleep(2)

            st.write("🔭 Signal Detector...")
            signals = run_signal_detector(company, verified_facts, evidence_sufficient)

            health_score      = calculate_health_score(verified_facts, signals, competitors)
            report_confidence = calculate_report_confidence(verified_facts, len(raw_facts))
            time.sleep(2)

            st.write("📋 Chief Strategy Officer — Options Analysis & Root Cause Synthesis...")
            final_brief = run_strategist(
                company, entity, verified_facts, signals, competitors,
                health_score, report_confidence, evidence_sufficient, sufficiency_message
            )

            status.update(label="Analysis Complete", state="complete")

        if not final_brief:
            st.error("Strategist failed. Try again.")
            st.stop()

        # ==========================================
        # DISPLAY: PIPELINE METRICS
        # ==========================================
        if not evidence_sufficient:
            st.error(f"⚠️ **SPECULATIVE REPORT** — {sufficiency_message}\n\nVerify against primary sources before acting.")
        else:
            st.success("✅ Evidence sufficient for reliable recommendations.")

        total  = len(raw_facts)
        passed = len(verified_facts)
        rate   = int(passed / total * 100) if total else 0

        st.subheader("🛡️ Intelligence Pipeline Metrics")
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Facts Extracted", total)
        m2.metric("Passed Hard Gate", passed)
        m3.metric("Gate Pass Rate", f"{rate}%")
        m4.metric("Signals", len(signals))
        m5.metric("Competitors", len(competitors))
        m6.metric("Entity Confidence", f"{entity_conf}%")

        # ==========================================
        # DISPLAY: STRATEGY LAYER (NEW FORMAT)
        # ==========================================
        st.divider()
        h1, h2, h3 = st.columns([3, 1, 1])
        with h1: st.header(f"Strategy & Execution Plan — {entity.canonical_name.upper()}")
        with h2: st.metric("Health Score", f"{final_brief.company_health_score}/100")
        with h3: st.metric("Report Confidence", f"{final_brief.report_confidence}%")
        st.caption(f"Entity Context: {final_brief.entity_context}")

        # 1. ROOT CAUSE ANALYSIS
        st.markdown("### 1. Root Cause Analysis")
        for rc in final_brief.root_causes:
            with st.container(border=True):
                st.markdown(f"**Fact:** {rc.fact}")
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"**Root Cause:**\n{rc.root_cause}")
                col2.markdown(f"**Driver Type:**\n{rc.temporary_or_structural}")
                col3.markdown(f"**Confidence:**\n{rc.confidence}")

        # 2. STRATEGIC THEMES
        st.markdown("### 2. Strategic Themes")
        for theme in final_brief.strategic_themes:
            with st.container(border=True):
                st.subheader(f"📌 {theme.theme}")
                st.markdown(f"**Why it matters:** {theme.why_it_matters}")
                with st.expander("Supporting Facts"):
                    for fact in theme.supporting_facts:
                        st.markdown(f"- {fact}")

        # 3. COMPETITIVE IMPLICATIONS
        if final_brief.competitive_implications:
            st.markdown("### 3. Competitive Implications")
            for imp in final_brief.competitive_implications:
                with st.container(border=True):
                    st.markdown(f"**Theme:** {imp.theme}")
                    st.warning(f"**Implication:** {imp.competitive_implication}")
                    c1, c2 = st.columns(2)
                    c1.success(f"**Winning Capability:** {imp.winning_capability}")
                    c2.error(f"**Risk Capability:** {imp.risk_capability}")

        # 4. OPTIONS ANALYSIS
        st.markdown("### 4. Options Analysis")
        for opt_group in final_brief.options_considered:
            st.markdown(f"#### Evaluating: {opt_group.theme}")
            tabs = st.tabs([opt.option_name for opt in opt_group.options])
            for idx, tab in enumerate(tabs):
                with tab:
                    opt = opt_group.options[idx]
                    st.markdown(f"**Description:** {opt.description}")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.success(f"**Benefits:** {opt.benefits}")
                        st.info(f"**Strategic Fit:** {opt.strategic_fit}")
                    with c2:
                        st.error(f"**Risks:** {opt.risks}")
                        st.warning(f"**Resource Requirements:** {opt.resource_requirements}")

        # 5. RECOMMENDED ACTIONS
        st.markdown("### 5. Final Recommendations")
        for i, action in enumerate(final_brief.recommended_actions, 1):
            with st.container(border=True):
                st.subheader(f"Recommendation #{i}: {action.action}")
                
                # The "Why" Matrix
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.info("**Why This Action?**")
                    st.markdown(action.why_this_action)
                with c2:
                    st.warning("**Why Now?**")
                    st.markdown(action.why_now)
                with c3:
                    st.error("**Why Not Alternatives?**")
                    st.markdown(action.why_not_alternatives)
                
                st.divider()
                
                # The "Impact" Matrix
                c4, c5 = st.columns(2)
                with c4:
                    st.markdown(f"**Expected Benefit:** {action.expected_benefit}")
                    st.markdown(f"**Key Risk:** {action.key_risk}")
                with c5:
                    st.markdown(f"**Supporting Theme:** {action.supporting_theme}")
                    with st.expander("View Supporting Evidence"):
                        for ev in action.supporting_evidence:
                            st.markdown(f"- {ev}")

        # Export
        st.divider()
        export = {
            "company": company,
            "entity_profile": entity.model_dump(),
            "pipeline_stats": {"extracted": total, "passed": passed, "rate_pct": rate},
            "verified_facts": [vf.model_dump() for vf in verified_facts],
            "signals": [s.model_dump() for s in signals],
            "competitor_intel": [c.model_dump() for c in competitors],
            "strategy_brief": final_brief.model_dump(),
        }
        st.download_button(
            "Download Full Strategy Package (JSON)",
            data=json.dumps(export, indent=2),
            file_name=f"{company.replace(' ','_')}_strategy_brief.json",
            mime="application/json"
        )