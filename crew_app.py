import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY") or st.secrets.get("GROQ_KEY","")

llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.1-8b-instant", temperature=0.3)

st.set_page_config(page_title="AI Market Intelligence Crew", page_icon="🔍")
st.title("🔍 AI Market Intelligence Crew")
st.markdown("**3 AI agents** — Goldman Sachs Researcher → BCG Analyst → McKinsey Strategist")
st.divider()

def search(query):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=4):
            results.append(r["title"] + ": " + r["body"])
    return "\n".join(results)

def run_researcher(company):
    n1 = search(company + " business news 2025")
    n2 = search(company + " financial results strategy")
    r = llm.invoke([
        SystemMessage(content="You are a Goldman Sachs research analyst. List 6 specific facts with numbers and dates."),
        HumanMessage(content=f"Research {company}:\n{n1}\n{n2}")
    ])
    return r.content

def run_analyst(company, research):
    r = llm.invoke([
        SystemMessage(content="You are a BCG consultant. Use Porter's Five Forces. Find strategic insights."),
        HumanMessage(content=f"Analyse {company}:\n{research[:600]}")
    ])
    return r.content

def run_strategist(company, research, analysis):
    r = llm.invoke([
        SystemMessage(content="You are a McKinsey Partner. Write executive brief: THREAT LEVEL, KEY INSIGHTS, RECOMMENDATIONS, 90-DAY PLAN."),
        HumanMessage(content=f"Brief for {company}:\nResearch: {research[:400]}\nAnalysis: {analysis[:400]}")
    ])
    return r.content

company = st.text_input("Company name:", placeholder="e.g. Zomato, Reliance, HDFC...")

if st.button("Deploy Crew", type="primary"):
    if not company:
        st.error("Please enter a company name.")
    else:
        with st.status(f"Crew analysing {company}...", expanded=True) as status:
            st.write("Researcher gathering intelligence...")
            research = run_researcher(company)
            st.write("Analyst identifying patterns...")
            analysis = run_analyst(company, research)
            st.write("Strategist writing board brief...")
            brief = run_strategist(company, research, analysis)
            status.update(label="Mission complete!", state="complete")

        with st.expander("Raw Research (Goldman Sachs)"):
            st.write(research)
        with st.expander("Strategic Analysis (BCG)"):
            st.write(analysis)
        st.subheader("Strategic Brief — " + company.upper())
        st.markdown(brief)
        full = f"RESEARCH:\n{research}\n\nANALYSIS:\n{analysis}\n\nBRIEF:\n{brief}"
        st.download_button("Download Report", data=full, file_name=company+"_crew.txt", mime="text/plain")

st.divider()
st.caption("3-Agent AI Crew · Portfolio Project")