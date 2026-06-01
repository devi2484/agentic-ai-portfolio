import os
import json
import math
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

def tool_web_search(query):
    print(f"  [TOOL: web_search] {query}")
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(r["title"] + ": " + r["body"])
    return "\n".join(results[:5])

def tool_calculator(expression):
    print(f"  [TOOL: calculator] {expression}")
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def tool_market_compare(companies_str):
    print(f"  [TOOL: market_compare] {companies_str}")
    companies = [c.strip() for c in companies_str.split(",")]
    results = {}
    for company in companies:
        with DDGS() as ddgs:
            news = list(ddgs.text(company + " market share revenue 2025", max_results=2))
            results[company] = news[0]["body"] if news else "No data found"
    return json.dumps(results, indent=2, ensure_ascii=False)

def tool_save_report(content):
    print(f"  [TOOL: save_report] Saving...")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"report_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Report saved as {filename}"

TOOLS = {
    "web_search": tool_web_search,
    "calculator": tool_calculator,
    "market_compare": tool_market_compare,
    "save_report": tool_save_report
}

TOOL_DESCRIPTIONS = """
Available tools:
1. web_search(query) - search web for current information
2. calculator(expression) - calculate numbers
3. market_compare(companies) - compare companies e.g. Zomato, Swiggy
4. save_report(content) - save final report to file
"""

def run_smart_agent(task):
    print("\n" + "="*50)
    print("SMART MULTI-TOOL AGENT")
    print("="*50)
    print(f"Task: {task}\n")

    plan_response = llm.invoke([
        SystemMessage(content=f"""You are a business analyst agent.
Plan which tools to use. Respond in this exact format:
PLAN: [your plan]
TOOL1: tool_name | input
TOOL2: tool_name | input
TOOL3: tool_name | input

Available tools:
{TOOL_DESCRIPTIONS}"""),
        HumanMessage(content=f"Task: {task}")
    ])

    plan = plan_response.content
    print("AGENT PLAN:\n" + plan + "\n")

    tool_results = {}
    for line in plan.split("\n"):
        if line.startswith("TOOL") and "|" in line:
            parts = line.split(":", 1)[1].strip().split("|")
            if len(parts) == 2:
                tool_name = parts[0].strip()
                tool_input = parts[1].strip()
                if tool_name in TOOLS:
                    result = TOOLS[tool_name](tool_input)
                    tool_results[tool_name] = result
                    print(f"  Done: {tool_name}\n")

    results_text = "\n\n".join([f"{k.upper()}:\n{v}" for k, v in tool_results.items()])

    final_response = llm.invoke([
        SystemMessage(content="You are a senior McKinsey analyst. Synthesise all tool results into a structured strategic report with: EXECUTIVE SUMMARY, KEY FINDINGS, STRATEGIC ANALYSIS, RECOMMENDATIONS, 30-DAY ACTIONS."),
        HumanMessage(content=f"Task: {task}\n\nTool results:\n{results_text}")
    ])

    final_answer = final_response.content
    print("\n" + "="*50)
    print("FINAL STRATEGIC REPORT")
    print("="*50)
    print(final_answer)
    tool_save_report(f"Task: {task}\n\n{final_answer}")
    return final_answer

task = input("\nEnter your analysis task: ")
run_smart_agent(task)