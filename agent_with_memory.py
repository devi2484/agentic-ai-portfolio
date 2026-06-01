import os
import json
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

MEMORY_FILE = "agent_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"analyses": [], "companies_researched": []}

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)
    print("Memory saved.")

def web_search(query):
    print("Searching: " + query)
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=4):
            results.append(r["title"] + ": " + r["body"])
    return "\n".join(results)

def run_agent(company):
    memory = load_memory()

    if company in memory["companies_researched"]:
        print(f"\nI already researched {company} before. Loading memory and finding updates...")

    memory_context = ""
    if memory["analyses"]:
        memory_context = "\n\nPREVIOUS RESEARCH IN MEMORY:\n"
        for a in memory["analyses"][-3:]:
            memory_context += f"- {a['company']} ({a['date']}): {a['summary'][:200]}\n"

    news = web_search(company + " business strategy 2025")
    news2 = web_search(company + " latest news financial results")

    messages = [
        SystemMessage(content="""You are a senior business analyst with memory of past research.
Structure your response: COMPANY OVERVIEW, KEY RECENT MOVES,
STRATEGIC ASSESSMENT, COMPARISON TO PAST RESEARCH, RECOMMENDATION."""),
        HumanMessage(content=f"Analyse {company}.\n\nCurrent research:\n{news}\n{news2}\n{memory_context}")
    ]

    response = llm.invoke(messages)
    analysis = response.content

    print("\n=== ANALYSIS: " + company.upper() + " ===")
    print(analysis)

    memory["analyses"].append({
        "company": company,
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "summary": analysis[:500]
    })
    if company not in memory["companies_researched"]:
        memory["companies_researched"].append(company)

    save_memory(memory)
    print(f"\nCompanies in memory: {', '.join(memory['companies_researched'])}")

company = input("\nWhich company to analyse? ")
run_agent(company)