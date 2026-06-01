import os
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

def web_search(query):
    print("Searching: " + query)
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=4):
            results.append(r["title"] + ": " + r["body"])
    return "\n".join(results)

print("Step 1: Searching web...")
news = web_search("Zomato business strategy 2025")

print("Step 2: Sending to AI...")
response = llm.invoke([
    SystemMessage(content="You are a business analyst. Analyse this news and give 3 key strategic insights."),
    HumanMessage(content=news)
])

print("\nAI RESPONSE:")
print(response.content)

with open("agent_output.txt", "w", encoding="utf-8") as f:
    f.write(response.content)
print("\nSaved to agent_output.txt")