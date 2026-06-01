import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_KEY"))

conversation_history = [
    {
        "role": "system",
        "content": "You are a senior business analyst. Be concise and strategic."
    }
]

print("AI Business Analyst ready. Type your question.")
print("Type 'quit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input == "quit":
        print("Goodbye!")
        break

    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        messages=conversation_history
    )

    ai_reply = response.choices[0].message.content

    conversation_history.append({
        "role": "assistant",
        "content": ai_reply
    })

    print("\nAI: " + ai_reply + "\n")