from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import datetime

GROQ_KEY = "your_groq_key_here"

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

Next quarter budget available: Rs 700000 (+Rs 100000)
Goal: Maximise revenue while building brand awareness
"""

def agent_debate(role, persona, task, extra=""):
    response = llm.invoke([
        SystemMessage(content=f"""You are the {role}.
{persona}
Use actual numbers from the data.
Be direct and specific. Maximum 200 words."""),
        HumanMessage(content=f"{task}\n\nData:\n{CAMPAIGN_DATA}\n{extra}")
    ])
    return response.content

def run_debate():
    print("\n" + "="*55)
    print("MARKETING BUDGET DEBATE — Q2 PLANNING")
    print("="*55)
    print(CAMPAIGN_DATA)

    # ROUND 1: Data Analyst proposes allocation
    print("\n" + "─"*55)
    print("ROUND 1: DATA ANALYST PROPOSES")
    print("─"*55)
    proposal = agent_debate(
        "Head of Marketing Analytics",
        """You optimise purely on data and ROI.
        You believe every rupee should go to highest ROAS channels.
        You distrust brand spending without measurable returns.""",
        """Propose exact budget allocation for Rs 700000 next quarter.
        Give specific amounts per channel.
        Justify each with ROAS data."""
    )
    print(proposal)

    # ROUND 2: Brand Strategist challenges
    print("\n" + "─"*55)
    print("ROUND 2: BRAND STRATEGIST CHALLENGES")
    print("─"*55)
    challenge = agent_debate(
        "Brand Strategy Director",
        """You believe short-term ROAS optimisation destroys long-term brand value.
        You champion brand awareness, reach, and customer lifetime value.
        You are concerned about over-reliance on paid channels.""",
        f"""The Data Analyst proposed this budget: {proposal[:400]}
        Challenge this proposal. What critical mistakes is it making?
        What will happen to the brand in 12 months if we follow this plan?""",
        ""
    )
    print(challenge)

    # ROUND 3: CFO stress-tests both
    print("\n" + "─"*55)
    print("ROUND 3: CFO STRESS-TESTS")
    print("─"*55)
    cfo = agent_debate(
        "Chief Financial Officer",
        """You protect the company from financial risk.
        You question whether ROAS is sustainable.
        You want conservative projections and risk mitigation.""",
        f"""Analyst proposal: {proposal[:300]}
        Brand concerns: {challenge[:300]}
        Are these ROAS numbers sustainable? What are the financial risks?
        What budget guardrails should we put in place?""",
        ""
    )
    print(cfo)

    # ROUND 4: CMO final decision
    print("\n" + "─"*55)
    print("FINAL DECISION: CHIEF MARKETING OFFICER")
    print("─"*55)
    final = agent_debate(
        "Chief Marketing Officer",
        """You balance data-driven performance with brand building.
        You are accountable to the CEO for both revenue AND brand health.
        You make the final call after considering all perspectives.""",
        f"""Having heard all perspectives:
ANALYST PROPOSAL: {proposal[:250]}
BRAND CONCERNS: {challenge[:250]}
CFO RISK FLAGS: {cfo[:250]}

Make the FINAL budget decision for Rs 700000.
Format your response as:
FINAL DECISION: (one sentence)
CHANNEL ALLOCATIONS: (exact amounts)
KEY TRADE-OFFS ACCEPTED: (what you chose not to do and why)
EXPECTED OUTCOME: (revenue and brand impact in 90 days)
SUCCESS METRICS: (how we measure if this worked)""",
        ""
    )
    print(final)

    # Save full debate
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"marketing_debate_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("MARKETING BUDGET DEBATE TRANSCRIPT\n")
        f.write("="*55 + "\n\n")
        f.write(f"DATA:\n{CAMPAIGN_DATA}\n\n")
        f.write(f"ANALYST PROPOSAL:\n{proposal}\n\n")
        f.write(f"BRAND CHALLENGE:\n{challenge}\n\n")
        f.write(f"CFO REVIEW:\n{cfo}\n\n")
        f.write(f"CMO FINAL DECISION:\n{final}")

    print(f"\nFull debate saved to {filename}")
    print("\n" + "="*55)
    print("Portfolio Project 3 COMPLETE")
    print("="*55)

run_debate()