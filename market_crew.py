import os
import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY")

llm = ChatGroq(
    api_key=GROQ_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)

def search(query):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(r["title"] + ": " + r["body"])
    return "\n".join(results)

def agent_researcher(company):
    print("\n" + "="*50)
    print("AGENT 1: RESEARCHER (Goldman Sachs)")
    print("="*50)
    news1 = search(company + " latest business news 2025")
    news2 = search(company + " financial results strategy 2025")
    news3 = search(company + " competitors market share 2025")
    response = llm.invoke([
        SystemMessage(content="""You are a Senior Research Analyst at Goldman Sachs.
Be specific with numbers and dates. Produce a research report with:
- 8 specific facts with numbers and dates
- Key financial metrics
- 3 most important recent strategic moves"""),
        HumanMessage(content=f"Research {company}:\nSOURCE 1:\n{news1}\nSOURCE 2:\n{news2}\nSOURCE 3:\n{news3}")
    ])
    print(response.content)
    return response.content

def agent_analyst(company, research):
    print("\n" + "="*50)
    print("AGENT 2: ANALYST (BCG)")
    print("="*50)
    response = llm.invoke([
        SystemMessage(content="""You are a Strategic Analyst at BCG.
Use Porter's Five Forces and SWOT. Produce:
COMPETITIVE POSITION, TOP 3 RISKS, TOP 3 OPPORTUNITIES,
COMPETITOR COMPARISON, KEY STRATEGIC INSIGHT"""),
        HumanMessage(content=f"Analyse {company} using this research:\n\n{research[:800]}")
    ])
    print(response.content)
    return response.content

def agent_strategist(company, research, analysis):
    print("\n" + "="*50)
    print("AGENT 3: STRATEGIST (McKinsey)")
    print("="*50)
    response = llm.invoke([
        SystemMessage(content="""You are a former McKinsey Partner.
Write a boardroom-ready strategic brief with exactly these sections:
EXECUTIVE SUMMARY, THREAT LEVEL, KEY STRATEGIC INSIGHTS,
COMPETITIVE IMPLICATIONS, RECOMMENDED RESPONSE, 90-DAY ACTION PLAN"""),
        HumanMessage(content=f"Brief for {company}:\nResearch: {research[:400]}\nAnalysis: {analysis[:400]}")
    ])
    print(response.content)
    return response.content

def run_crew(company):
    print("\n" + "="*55)
    print(f"3-AGENT CREW: {company.upper()}")
    print("="*55)

    research = agent_researcher(company)
    analysis = agent_analyst(company, research)
    final_brief = agent_strategist(company, research, analysis)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = company + "_crew_report_" + timestamp + ".txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"3-AGENT CREW REPORT: {company.upper()}\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("="*55 + "\n\n")
        f.write("AGENT 1 - RESEARCH:\n" + research + "\n\n")
        f.write("AGENT 2 - ANALYSIS:\n" + analysis + "\n\n")
        f.write("AGENT 3 - STRATEGIC BRIEF:\n" + final_brief)

    print(f"\nCrew complete. Report saved as: {filename}")

company = input("\nEnter company for full crew analysis: ")
run_crew(company)