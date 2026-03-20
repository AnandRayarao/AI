import anthropic
import os
from dotenv import load_dotenv

SYSTEM_PROMPT = """You are a helpful healthcare assistant. 
You help users understand healthcare topics, medical terminology, 
insurance concepts, and general wellness information.
Always remind users to consult a real doctor for medical advice.
Keep answers clear and simple."""

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
def ask_claude(question):
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        #system=SYSTEM_PROMPT,        # system prompt added here
        messages=[
            {"role": "user", "content": question}
        ]
    )
    return message.content[0].text
print("Ask Claude anything. Type quit to exit.")
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    answer = ask_claude(question)
    print("Claude:", answer)