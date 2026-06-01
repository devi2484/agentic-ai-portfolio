import os
import requests
import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY")
SERPER_KEY = os.getenv("SERPER_KEY")

client = Groq(api_key=GROQ_KEY)

def search_company_news(company_name):
    print("Searching for latest news on " + company_name + "...")
    response = requests.post(
        "https://google.serper.dev/news",
        headers={"X-API-KEY": SERPER_KEY},
        json={"q": company_name + " business news 2025", "num": 8}
    )
    results = response.json()
    news_text = ""
    for item in results["news"]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        news_text = news_text + "- " + title + ": " + snippet + "\n"
    print("Found " + str(len(results["news"])) + " articles.")
    return news_text

def extract_signals(company_name, raw_news):
    print("AI extracting strategic signals...")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You are a business intelligence analyst. Extract the 5 most strategically important signals from news. Write each as one clear sentence starting with a verb."
            },
            {
                "role": "user",
                "content": "Extract top 5 signals from this news about " + company_name + ":\n\n" + raw_news
            }
        ]
    )
    return response.choices[0].message.content

def generate_report(company_name, signals):
    print("Writing strategic report...")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.4,
        messages=[
            {
                "role": "system",
                "content": "You are a senior McKinsey business analyst. Write a strategic brief with sections: THREAT LEVEL, KEY STRATEGIC INSIGHT, COMPETITIVE IMPLICATIONS, RECOMMENDED RESPONSE, 30-DAY PRIORITY ACTIONS."
            },
            {
                "role": "user",
                "content": "Write a strategic brief for " + company_name + " based on:\n\n" + signals
            }
        ]
    )
    return response.choices[0].message.content

def run_analysis(company_name):
    print("\n" + "="*45)
    print("ANALYSING: " + company_name.upper())
    print("="*45 + "\n")

    raw_news = search_company_news(company_name)
    signals = extract_signals(company_name, raw_news)

    print("\n--- EXTRACTED SIGNALS ---")
    print(signals)

    report = generate_report(company_name, signals)

    print("\n--- STRATEGIC REPORT ---")
    print(report)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = company_name + "_auto_analysis_" + timestamp + ".txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("AUTO-GENERATED COMPETITIVE INTELLIGENCE REPORT\n")
        f.write("Company: " + company_name + "\n")
        f.write("Generated: " + timestamp + "\n")
        f.write("="*45 + "\n\n")
        f.write("EXTRACTED SIGNALS:\n" + signals + "\n\n")
        f.write("STRATEGIC REPORT:\n" + report)

    print("\nReport saved as: " + filename)

company = input("\nEnter company name to analyse: ")
run_analysis(company)