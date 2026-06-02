import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from ddgs import DDGS
from dotenv import load_dotenv

# ------------------------
# CONFIG
# ------------------------

load_dotenv()

GROQ_KEY = os.getenv("GROQ_KEY") or st.secrets.get("GROQ_KEY", "")

llm = ChatGroq(
    api_key=GROQ_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

st.set_page_config(
    page_title="AI Market Intelligence Crew",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Market Intelligence Crew")
st.markdown(
    "**4 AI Agents** → Goldman Sachs Researcher → Fact Checker → BCG Analyst → McKinsey Strategist"
)

st.divider()

# ------------------------
# SEARCH
# ------------------------

def search(query):

    results = []

    try:
        with DDGS() as ddgs:

            for r in ddgs.text(query, max_results=10):

                title = r.get("title", "")
                body = r.get("body", "")

                results.append(
                    f"TITLE: {title}\nSUMMARY: {body}"
                )

    except Exception as e:
        return f"Search Error: {e}"

    return "\n\n".join(results)


# ------------------------
# RESEARCH AGENT
# ------------------------

def run_researcher(company):

    news = search(f"{company} latest business news")
    finance = search(f"{company} financial results earnings")
    strategy = search(f"{company} strategy acquisitions expansion")

    research_context = f"""
NEWS:
{news}

FINANCIALS:
{finance}

STRATEGY:
{strategy}
"""

    response = llm.invoke([
        SystemMessage(content="""
You are a Goldman Sachs research analyst.

Your task:

Generate exactly 6 high-value facts.

Rules:

- Must contain numbers or dates.
- Must be business relevant.
- Must be based ONLY on provided search results.
- Do not invent facts.

Format:

FACT:
SOURCE:
DATE:
CONFIDENCE: High/Medium/Low

Repeat 6 times.
"""),
        HumanMessage(content=research_context)
    ])

    return response.content


# ------------------------
# FACT VALIDATOR
# ------------------------

def run_validator(research):

    response = llm.invoke([
        SystemMessage(content="""
You are a forensic fact checker.

Review every fact.

For each fact provide:

FACT:
STATUS: VERIFIED / LIKELY / QUESTIONABLE
CONFIDENCE: HIGH / MEDIUM / LOW
REASON:

Be strict.
"""),
        HumanMessage(content=research)
    ])

    return response.content


# ------------------------
# ANALYST
# ------------------------

def run_analyst(company, validated_facts):

    response = llm.invoke([
        SystemMessage(content="""
You are a BCG Partner.

Rules:

- Use ONLY facts marked VERIFIED or LIKELY.
- Ignore QUESTIONABLE facts.
- Apply Porter's Five Forces.

Output:

1. Market Position
2. Competitive Threats
3. Opportunities
4. Risks
5. Strategic Insights

Support conclusions with facts.
"""),
        HumanMessage(content=f"""
Company: {company}

Validated Facts:

{validated_facts}
""")
    ])

    return response.content


# ------------------------
# STRATEGIST
# ------------------------

def run_strategist(company, research, validation, analysis):

    response = llm.invoke([
        SystemMessage(content="""
You are a McKinsey Senior Partner.

Create a board-level executive briefing.

Output:

EXECUTIVE SUMMARY

THREAT LEVEL:
(Low / Medium / High)

TOP 5 INSIGHTS

KEY RISKS

KEY OPPORTUNITIES

RECOMMENDATIONS

90-DAY ACTION PLAN

90-180 DAY ACTION PLAN

CEO TAKEAWAY

Be concise but strategic.
"""),
        HumanMessage(content=f"""
COMPANY:
{company}

RESEARCH:
{research}

VALIDATION:
{validation}

ANALYSIS:
{analysis}
""")
    ])

    return response.content


# ------------------------
# UI
# ------------------------

company = st.text_input(
    "Company Name",
    placeholder="Tesla, Zomato, Nykaa, Reliance..."
)

if st.button("Deploy Crew", type="primary"):

    if not company:

        st.error("Please enter a company name.")

    else:

        with st.status(
            f"Analysing {company}...",
            expanded=True
        ) as status:

            st.write("🔍 Research agent gathering intelligence...")
            research = run_researcher(company)

            st.write("✅ Fact-checking intelligence...")
            validation = run_validator(research)

            st.write("📊 BCG analyst reviewing market...")
            analysis = run_analyst(company, validation)

            st.write("🎯 McKinsey strategist building board brief...")
            brief = run_strategist(
                company,
                research,
                validation,
                analysis
            )

            status.update(
                label="Mission Complete",
                state="complete"
            )

        st.subheader(f"Executive Brief — {company}")

        st.markdown(brief)

        with st.expander("Research Output"):
            st.write(research)

        with st.expander("Fact Validation"):
            st.write(validation)

        with st.expander("Strategic Analysis"):
            st.write(analysis)

        full_report = f"""
RESEARCH
========
{research}

VALIDATION
==========
{validation}

ANALYSIS
========
{analysis}

BRIEF
=====
{brief}
"""

        st.download_button(
            "Download Report",
            data=full_report,
            file_name=f"{company}_strategy_report.txt",
            mime="text/plain"
        )

st.divider()
st.caption("4-Agent Market Intelligence Crew")