import os
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY")

llm = ChatGroq(
    api_key=GROQ_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.1
)

class SupplyState(TypedDict):
    supplier: str
    normal_days: int
    current_days: int
    change_pct: float
    anomaly: bool
    severity: str
    ai_analysis: str
    resolution: str

def node_monitor(state: SupplyState):
    change = ((state["current_days"] - state["normal_days"]) / state["normal_days"]) * 100
    anomaly = change > 25
    print(f"\nMONITOR: {state['supplier']}")
    print(f"Normal: {state['normal_days']}d | Current: {state['current_days']}d | Change: {change:.1f}%")
    print(f"Anomaly: {'YES' if anomaly else 'NO'}")
    return {"change_pct": round(change, 1), "anomaly": anomaly}

def node_classify(state: SupplyState):
    pct = state["change_pct"]
    if pct > 150:   severity = "CRITICAL"
    elif pct > 75:  severity = "HIGH"
    elif pct > 25:  severity = "MEDIUM"
    else:           severity = "LOW"
    print(f"CLASSIFY: {severity}")
    return {"severity": severity}

def node_analyse(state: SupplyState):
    print(f"ANALYSE: AI diagnosing...")
    prompt = f"""Supply chain anomaly:
Supplier: {state['supplier']}
Normal: {state['normal_days']} days | Current: {state['current_days']} days
Change: {state['change_pct']}% | Severity: {state['severity']}

Give exactly 3 lines:
1. Most likely cause
2. Business impact if unresolved
3. Immediate action required"""
    response = llm.invoke(prompt)
    return {"ai_analysis": response.content}

def node_auto_resolve(state: SupplyState):
    resolution = f"AUTO-RESOLVED: Activated backup supplier. Rerouted {state['supplier']} orders to secondary vendor. Monitoring every 6 hours."
    print(f"AUTO-RESOLVED: {state['supplier']}")
    return {"resolution": resolution}

def node_escalate(state: SupplyState):
    resolution = f"ESCALATED: {state['severity']} anomaly at {state['supplier']}. Lead time up {state['change_pct']}%. Procurement Director notified. Emergency review scheduled."
    print(f"ESCALATED: {state['supplier']}")
    return {"resolution": resolution}

def route_after_monitor(state):
    if state["anomaly"]:
        return "classify"
    print(f"Normal — no action needed.")
    return END

def route_after_analyse(state):
    if state["severity"] in ["CRITICAL", "HIGH"]:
        return "escalate"
    return "auto_resolve"

graph = StateGraph(SupplyState)
graph.add_node("monitor", node_monitor)
graph.add_node("classify", node_classify)
graph.add_node("analyse", node_analyse)
graph.add_node("auto_resolve", node_auto_resolve)
graph.add_node("escalate", node_escalate)
graph.set_entry_point("monitor")
graph.add_conditional_edges("monitor", route_after_monitor, {"classify": "classify", END: END})
graph.add_edge("classify", "analyse")
graph.add_conditional_edges("analyse", route_after_analyse, {"escalate": "escalate", "auto_resolve": "auto_resolve"})
graph.add_edge("auto_resolve", END)
graph.add_edge("escalate", END)
app = graph.compile()

scenarios = [
    {"supplier": "TechParts Ltd", "normal_days": 14, "current_days": 16},
    {"supplier": "PackCo India",  "normal_days": 7,  "current_days": 13},
    {"supplier": "RawMat Corp",   "normal_days": 10, "current_days": 26},
    {"supplier": "ChipSupply",    "normal_days": 21, "current_days": 58},
]

print("\n" + "="*55)
print("SUPPLY CHAIN ANOMALY DETECTION AGENT")
print("="*55)

for s in scenarios:
    state = {**s, "change_pct":0.0, "anomaly":False, "severity":"", "ai_analysis":"", "resolution":""}
    result = app.invoke(state)
    if result.get("ai_analysis"):
        print(f"\nAI DIAGNOSIS:\n{result['ai_analysis']}")
    if result.get("resolution"):
        print(f"RESOLUTION: {result['resolution']}")
    print("-"*45)

with open("supply_chain_report.txt", "w", encoding="utf-8") as f:
    f.write("SUPPLY CHAIN ANOMALY REPORT\n" + "="*45 + "\n\n")
    for s in scenarios:
        f.write(f"Supplier: {s['supplier']}\n\n")
print("Report saved to supply_chain_report.txt")
marketing_debate_agent.py
Portfolio Project 3 — debate agents
import os
import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY")

llm = ChatGroq(
    api_key=GROQ_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.5
)

CAMPAIGN_DATA = """
QUARTERLY MARKETING PERFORMANCE:
Channel          Spend(Rs)  Conversions  Revenue(Rs)  ROAS
Instagram Ads    200000     850          720000       3.6x
Google Search    150000     920          920000       6.1x
YouTube Ads       80000     210          126000       1.6x
Email Marketing   20000     480          480000      24.0x
Influencer Mktg  150000     340          272000       1.8x
TOTAL            600000    2800         2518000       4.2x
Next quarter budget: Rs 700000
"""

def agent_debate(role, persona, task, extra=""):
    response = llm.invoke([
        SystemMessage(content=f"You are the {role}.\n{persona}\nUse actual numbers. Maximum 200 words."),
        HumanMessage(content=f"{task}\n\nData:\n{CAMPAIGN_DATA}\n{extra}")
    ])
    return response.content

def run_debate():
    print("\n" + "="*55)
    print("MARKETING BUDGET DEBATE")
    print("="*55)
    print(CAMPAIGN_DATA)

    print("\n--- ROUND 1: DATA ANALYST ---")
    proposal = agent_debate(
        "Head of Marketing Analytics",
        "You optimise purely on ROAS. Every rupee goes to highest performing channels.",
        "Propose exact budget allocation for Rs 700000. Justify each with ROAS data."
    )
    print(proposal)

    print("\n--- ROUND 2: BRAND STRATEGIST ---")
    challenge = agent_debate(
        "Brand Strategy Director",
        "You believe ROAS optimisation destroys long-term brand value. You champion awareness and customer lifetime value.",
        f"Challenge this proposal: {proposal[:300]}\nWhat critical mistakes is it making?", ""
    )
    print(challenge)

    print("\n--- ROUND 3: CFO ---")
    cfo = agent_debate(
        "Chief Financial Officer",
        "You protect financial risk. You question ROAS sustainability.",
        f"Analyst: {proposal[:200]}\nBrand: {challenge[:200]}\nAre these projections sustainable? What are the risks?", ""
    )
    print(cfo)

    print("\n--- FINAL: CMO DECISION ---")
    final = agent_debate(
        "Chief Marketing Officer",
        "You balance data and brand. You make the final call.",
        f"Analyst: {proposal[:200]}\nBrand: {challenge[:200]}\nCFO: {cfo[:200]}\n\nMake the FINAL budget decision for Rs 700000.\nFormat: FINAL DECISION, CHANNEL ALLOCATIONS, TRADE-OFFS, EXPECTED OUTCOME, SUCCESS METRICS", ""
    )
    print(final)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"marketing_debate_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("MARKETING BUDGET DEBATE TRANSCRIPT\n" + "="*55 + "\n\n")
        f.write(f"DATA:\n{CAMPAIGN_DATA}\n\n")
        f.write(f"ANALYST:\n{proposal}\n\n")
        f.write(f"BRAND:\n{challenge}\n\n")
        f.write(f"CFO:\n{cfo}\n\n")
        f.write(f"CMO DECISION:\n{final}")
    print(f"\nSaved to {filename}")

run_debate()