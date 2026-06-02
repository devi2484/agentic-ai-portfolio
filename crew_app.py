import os
import json
import streamlit as st
import feedparser

from dotenv import load_dotenv
from ddgs import DDGS

from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GROQ_KEY = (
    os.getenv("GROQ_KEY")
    or st.secrets.get("GROQ_KEY", "")
)

if not GROQ_KEY:
    st.error("Missing GROQ_KEY")
    st.stop()

# Fast + Cheap model
fast_llm = ChatGroq(
    api_key=GROQ_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

# Better strategy model
smart_llm = ChatGroq(
    api_key=GROQ_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)

# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="AI Market Intelligence Crew",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Market Intelligence Crew")

st.markdown("""
### Multi-Agent Strategic Intelligence System

**Pipeline**

Research → Validation → Fact Filter → Competitive Intelligence → Strategic Analysis → Challenger Review → CEO Brief
""")

st.divider()

# =========================================================
# SEARCH
# =========================================================

def ddg_search(query):

    results = []

    try:

        with DDGS() as ddgs:

            for r in ddgs.text(
                query,
                max_results=10
            ):

                title = r.get("title", "")
                body = r.get("body", "")

                results.append(
                    f"TITLE: {title}\nSUMMARY: {body}"
                )

    except Exception as e:

        return f"DDGS ERROR: {e}"

    return "\n\n".join(results)


def google_news_search(query):

    try:

        rss_url = (
            "https://news.google.com/rss/search?q="
            + query.replace(" ", "+")
        )

        feed = feedparser.parse(rss_url)

        results = []

        for entry in feed.entries[:10]:

            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")

            results.append(
                f"TITLE: {title}\nSUMMARY: {summary}"
            )

        return "\n\n".join(results)

    except Exception as e:

        return f"GOOGLE NEWS ERROR: {e}"


def combined_search(company):

    ddg = ddg_search(
        company + " financial results business strategy acquisitions"
    )

    news = google_news_search(company)

    return f"""

DUCKDUCKGO RESULTS
==================

{ddg}

GOOGLE NEWS RESULTS
===================

{news}

"""


# =========================================================
# HELPERS
# =========================================================

def safe_json_parse(text):

    try:
        return json.loads(text)

    except Exception:

        # Try extracting JSON block

        try:

            start = text.find("[")
            end = text.rfind("]") + 1

            cleaned = text[start:end]

            return json.loads(cleaned)

        except Exception:

            return []


def confidence_score(validated):

    total = len(validated)

    if total == 0:
        return 0

    verified = len([
        x for x in validated
        if x.get("status") == "VERIFIED"
    ])

    return round((verified / total) * 100)


def filter_verified_facts(validated):

    good = []

    for fact in validated:

        if fact.get("status") in [
            "VERIFIED",
            "LIKELY"
        ]:

            good.append(fact)

    return good


# =========================================================
# RESEARCH AGENT
# =========================================================

def run_researcher(company):

    context = combined_search(company)

    response = fast_llm.invoke([

        SystemMessage(content="""
You are a Goldman Sachs research analyst.

Return ONLY valid JSON.

Format:

[
  {
    "fact":"...",
    "source":"...",
    "date":"...",
    "confidence":"HIGH"
  }
]

Rules:

- Exactly 6 facts.
- Must contain numbers or dates.
- Must be business relevant.
- Must come ONLY from search results.
- Prefer facts appearing in multiple sources.
- No markdown.
- No commentary.
"""),

        HumanMessage(content=context)

    ])

    return safe_json_parse(response.content)


# =========================================================
# VALIDATION AGENT
# =========================================================

def run_validator(facts):

    response = fast_llm.invoke([

        SystemMessage(content="""
You are a forensic fact checker.

Return ONLY valid JSON.

Format:

[
 {
   "fact":"...",
   "status":"VERIFIED",
   "confidence":"HIGH",
   "reason":"..."
 }
]

Rules:

- VERIFIED = strongly supported.
- LIKELY = probably correct.
- QUESTIONABLE = weak or unsupported.
- Be strict.
- No markdown.
"""),

        HumanMessage(content=json.dumps(facts))

    ])

    return safe_json_parse(response.content)


# =========================================================
# COMPETITIVE INTELLIGENCE
# =========================================================

def run_competitive_intel(company, facts):

    response = smart_llm.invoke([

        SystemMessage(content="""
You are a market intelligence expert.

Analyze:

1. Main competitors
2. Who is winning
3. Why they are winning
4. Which moat is strongest
5. Which moat is weakening
6. Strategic pressure points

Avoid generic statements.
"""),

        HumanMessage(content=f"""
COMPANY:
{company}

FACTS:
{json.dumps(facts, indent=2)}
""")

    ])

    return response.content


# =========================================================
# STRATEGIC ANALYSIS
# =========================================================

def run_analysis(
    company,
    facts,
    competitive_intel
):

    response = smart_llm.invoke([

        SystemMessage(content="""
You are a BCG Senior Partner.

Analyze:

1. What changed?
2. Growth drivers
3. Strategic risks
4. Capital allocation implications
5. Future bets
6. Competitive positioning

Rules:

- Avoid generic business advice.
- Use evidence.
- Be specific.
- Think like a real strategist.
"""),

        HumanMessage(content=f"""
COMPANY:
{company}

FACTS:
{json.dumps(facts, indent=2)}

COMPETITIVE INTELLIGENCE:
{competitive_intel}
""")

    ])

    return response.content


# =========================================================
# CHALLENGER AGENT
# =========================================================

def run_challenger(company, analysis):

    response = smart_llm.invoke([

        SystemMessage(content="""
You are a Bain Partner.

Challenge the analysis.

Identify:

1. Weak assumptions
2. Missing evidence
3. Risks
4. Alternative interpretations
5. Recommendations likely to fail

Be skeptical.
"""),

        HumanMessage(content=f"""
COMPANY:
{company}

ANALYSIS:
{analysis}
""")

    ])

    return response.content


# =========================================================
# CEO BRIEF AGENT
# =========================================================

def run_ceo_brief(
    company,
    facts,
    competitive_intel,
    analysis,
    challenge
):

    response = smart_llm.invoke([

        SystemMessage(content="""
You are a Fortune 500 CEO.

You have:
- 12 months
- $10B capital
- shareholder pressure

Create a strategic CEO briefing.

Output:

1. COMPANY HEALTH
2. WHAT CHANGED
3. TOP STRATEGIC RISKS
4. BIGGEST OPPORTUNITIES
5. STOP
6. START
7. DOUBLE DOWN
8. KEY STRATEGIC MOVES
9. CEO TAKEAWAY

Rules:

- Generic advice is forbidden.
- Every recommendation must contain:
  Evidence
  Action
  Expected Impact
  Risk
  Timeline
- Think like an elite operator.
"""),

        HumanMessage(content=f"""
COMPANY:
{company}

FACTS:
{json.dumps(facts, indent=2)}

COMPETITIVE INTEL:
{competitive_intel}

ANALYSIS:
{analysis}

CHALLENGER REVIEW:
{challenge}
""")

    ])

    return response.content


# =========================================================
# UI
# =========================================================

company = st.text_input(
    "Company Name",
    placeholder="Tesla, Nvidia, Zomato, Reliance..."
)

if st.button(
    "Deploy Intelligence Crew",
    type="primary"
):

    if not company:

        st.error("Enter a company name.")

    else:

        with st.status(
            f"Analyzing {company}...",
            expanded=True
        ) as status:

            # Research

            st.write(
                "🔍 Research agent gathering intelligence..."
            )

            research = run_researcher(company)

            # Validation

            st.write(
                "✅ Validation agent checking facts..."
            )

            validation = run_validator(research)

            # Filter

            verified_facts = filter_verified_facts(
                validation
            )

            # Confidence

            score = confidence_score(validation)

            # Competitive Intel

            st.write(
                "⚔️ Competitive intelligence agent running..."
            )

            intel = run_competitive_intel(
                company,
                verified_facts
            )

            # Analysis

            st.write(
                "📊 BCG strategy analysis..."
            )

            analysis = run_analysis(
                company,
                verified_facts,
                intel
            )

            # Challenger

            st.write(
                "🧠 Challenger agent stress-testing strategy..."
            )

            challenge = run_challenger(
                company,
                analysis
            )

            # CEO Brief

            st.write(
                "🎯 CEO briefing agent preparing report..."
            )

            brief = run_ceo_brief(
                company,
                verified_facts,
                intel,
                analysis,
                challenge
            )

            status.update(
                label="Mission Complete",
                state="complete"
            )

        # =================================================
        # DASHBOARD
        # =================================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Report Confidence",
                f"{score}%"
            )

        with col2:

            st.metric(
                "Verified Facts",
                len(verified_facts)
            )

        st.divider()

        # =================================================
        # FINAL BRIEF
        # =================================================

        st.subheader(
            f"CEO Strategic Brief — {company}"
        )

        st.markdown(brief)

        # =================================================
        # EXPANDERS
        # =================================================

        with st.expander("Research Facts"):

            st.json(research)

        with st.expander("Fact Validation"):

            st.json(validation)

        with st.expander("Verified Facts Only"):

            st.json(verified_facts)

        with st.expander("Competitive Intelligence"):

            st.write(intel)

        with st.expander("Strategic Analysis"):

            st.write(analysis)

        with st.expander("Challenger Review"):

            st.write(challenge)

        # =================================================
        # DOWNLOAD
        # =================================================

        full_report = f"""
CEO STRATEGIC BRIEF
===================

{brief}

RESEARCH FACTS
==============

{json.dumps(research, indent=2)}

VALIDATION
==========

{json.dumps(validation, indent=2)}

VERIFIED FACTS
==============

{json.dumps(verified_facts, indent=2)}

COMPETITIVE INTELLIGENCE
========================

{intel}

STRATEGIC ANALYSIS
==================

{analysis}

CHALLENGER REVIEW
=================

{challenge}
"""

        st.download_button(
            "Download Full Report",
            data=full_report,
            file_name=f"{company}_ceo_strategy_report.txt",
            mime="text/plain"
        )

st.divider()

st.caption(
    "AI Market Intelligence Crew · Multi-Agent Strategic Intelligence System"
)