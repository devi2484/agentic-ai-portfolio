import os
import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv(r"C:\Users\devan\OneDrive\Desktop\.env")
client = Groq(api_key=os.getenv("GROQ_KEY"))

print("=== AI Competitor Analyser ===\n")
company = input("Enter company name to analyse: ")

print("\nEnter competitor signals one by one.")
print("Type 'done' when finished.\n")

signals = []
while True:
    signal = input("Signal: ")
    if signal == "done":
        break
    signals.append(signal)

print("\nAnalysing " + company + "... please wait...\n")

signals_text = ""
for s in signals:
    signals_text = signals_text + "- " + s + "\n"

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    messages=[
        {
            "role": "system",
            "content": "You are a senior McKinsey business analyst. Structure your response with these exact sections: THREAT LEVEL, KEY STRATEGIC INSIGHT, RECOMMENDED ACTION, 30-DAY PRIORITY."
        },
        {
            "role": "user",
            "content": "Analyse these signals from " + company + ":\n\n" + signals_text
        }
    ]
)

analysis = response.choices[0].message.content

print("=== ANALYSIS: " + company.upper() + " ===\n")
print(analysis)
print("\n=== END OF REPORT ===")

import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = company + "_analysis_" + timestamp + ".txt"

with open(filename, "w", encoding="utf-8") as f:
    f.write("COMPETITOR ANALYSIS REPORT\n")
    f.write("Company: " + company + "\n")
    f.write("Date: " + timestamp + "\n")
    f.write("="*40 + "\n\n")
    f.write(analysis)

print("\nReport saved as: " + filename)