import streamlit as st
import sys
import os

st.set_page_config(
    page_title="AI Market Intelligence Crew",
    page_icon="🔍", layout="centered"
)

st.title("🔍 AI Market Intelligence Crew")
st.markdown("""
**3 specialised AI agents** working as a consulting team:
Researcher (Goldman Sachs) → Analyst (BCG) → Strategist (McKinsey)
""")
st.divider()

# Import your crew functions
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from ddgs import DDGS

GROQ_KEY = "paste_your_groq_key_here"
llm = ChatGroq(
    api_key=GROQ_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)

def search(query):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=4):
            results.append(r["title"] + ": " + r["body"])
    return "\n".join(results)

def run_researcher(company):
    news1 = search(company + " business news 2025")
    news2 = search(company + " financial results strategy")
    response = llm.invoke([
        SystemMessage(content="You are a Goldman Sachs research analyst. Be specific with numbers and dates. List 6 key facts."),
        HumanMessage(content=f"Research {company}:\n{news1}\n{news2}")
    ])
    return response.content

def run_analyst(company, research):
    response = llm.invoke([
        SystemMessage(content="You are a BCG consultant. Use Porter's Five Forces. Find non-obvious insights."),
        HumanMessage(content=f"Analyse {company}:\n{research[:600]}")
    ])
    return response.content

def run_strategist(company, research, analysis):
    response = llm.invoke([
        SystemMessage(content="You are a McKinsey Partner. Write an executive brief. Sections: THREAT LEVEL, KEY INSIGHTS, RECOMMENDATIONS, 90-DAY PLAN."),
        HumanMessage(content=f"Brief for {company}:\nResearch: {research[:400]}\nAnalysis: {analysis[:400]}")
    ])
    return response.content

# UI
company = st.text_input("Enter company name:", placeholder="e.g. Zomato, Reliance, HDFC...")

if st.button("🚀 Deploy Crew", type="primary"):
    if not company:
        st.error("Please enter a company name.")
    else:
        with st.status(f"Crew working on {company}...", expanded=True) as status:
            st.write("🔎 Researcher searching for intelligence...")
            research = run_researcher(company)
            st.write("✅ Research complete.")

            st.write("📊 Analyst identifying strategic patterns...")
            analysis = run_analyst(company, research)
            st.write("✅ Analysis complete.")

            st.write("✍️ Strategist writing board brief...")
            brief = run_strategist(company, research, analysis)
            st.write("✅ Brief complete.")

            status.update(label="Crew mission complete!", state="complete")

        st.divider()
        with st.expander("📋 Raw Research (Goldman Sachs Analyst)"):
            st.write(research)

        with st.expander("📈 Strategic Analysis (BCG Consultant)"):
            st.write(analysis)

        st.subheader("📊 Strategic Brief — " + company.upper())
        st.markdown(brief)

        full = f"RESEARCH:\n{research}\n\nANALYSIS:\n{analysis}\n\nBRIEF:\n{brief}"
        st.download_button(
            "⬇️ Download Full Report",
            data=full,
            file_name=company + "_crew_report.txt",
            mime="text/plain"
        )

st.divider()
st.caption("3-Agent AI Crew · Goldman Sachs → BCG → McKinsey ")