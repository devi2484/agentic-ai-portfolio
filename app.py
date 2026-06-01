import os
import requests
import datetime
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY") or st.secrets.get("GROQ_KEY", "")
SERPER_KEY = os.getenv("SERPER_KEY") or st.secrets.get("SERPER_KEY", "")

client = Groq(api_key=GROQ_KEY)

st.set_page_config(page_title="AI Competitor Analyser", page_icon="🔍")
st.title("🔍 AI Competitor Intelligence Analyser")
st.write("Type any company name and get a strategic analysis in 30 seconds.")
st.divider()

company = st.text_input("Enter company name:", placeholder="e.g. Zomato, Swiggy...")

if st.button("Analyse Now", type="primary"):
    if not company:
        st.error("Please type a company name first.")
    else:
        with st.status("Running analysis...", expanded=True) as status:
            st.write("Searching Google News for " + company + "...")
            search_response = requests.post(
                "https://google.serper.dev/news",
                headers={"X-API-KEY": SERPER_KEY},
                json={"q": company + " business news 2025", "num": 8}
            )
            results = search_response.json()
            news_text = ""
            for item in results["news"]:
                news_text += "- " + item.get("title","") + ": " + item.get("snippet","") + "\n"
            st.write("News found. AI extracting signals...")

            signals_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0.2,
                messages=[
                    {"role":"system","content":"Extract the 5 most strategic signals from this news. One sentence each."},
                    {"role":"user","content":"News about " + company + ":\n\n" + news_text}
                ]
            )
            signals = signals_response.choices[0].message.content
            st.write("Signals extracted. Writing report...")

            report_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0.4,
                messages=[
                    {"role":"system","content":"You are a McKinsey analyst. Write a strategic brief with: THREAT LEVEL, KEY INSIGHT, COMPETITIVE IMPLICATIONS, RECOMMENDATIONS, 30-DAY ACTIONS."},
                    {"role":"user","content":"Brief for " + company + " based on:\n\n" + signals}
                ]
            )
            report = report_response.choices[0].message.content
            status.update(label="Analysis complete!", state="complete")

        st.divider()
        st.subheader("Extracted Signals")
        st.info(signals)
        st.divider()
        st.subheader("Strategic Report — " + company.upper())
        st.markdown(report)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        full_text = "SIGNALS:\n" + signals + "\n\nREPORT:\n" + report
        st.download_button(
            label="Download Report",
            data=full_text,
            file_name=company + "_analysis_" + timestamp + ".txt",
            mime="text/plain"
        )

st.divider()
st.caption("Built with Python · Groq · Serper · Streamlit")