import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
def ask_claude(question):
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
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