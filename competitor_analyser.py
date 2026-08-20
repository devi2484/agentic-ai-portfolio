import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_KEY"))

def analyse_competitor(company_name, signals):
    signals_text = ""
    for signal in signals:
        signals_text = signals_text + "- " + signal + "\n"

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "You are a senior business analyst. Analyse competitor signals and give a clear strategic brief with: 1) Threat level 2) Key insight 3) Recommended action."
            },
            {
                "role": "user",
                "content": "Analyse these signals from " + company_name + ":\n\n" + signals_text
            }
        ]
    )
    return response.choices[0].message.content

result = analyse_competitor(
    "Zomato",
    [
        "Acquired Blinkit for quick commerce expansion",
        "Reported first quarterly profit",
        "Launched Zomato Gold loyalty programme",
        "Expanding into B2B food supply"
    ]
)
print(result)