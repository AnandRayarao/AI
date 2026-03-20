import anthropic
import os
from dotenv import load_dotenv
from pdf_reader import read_pdf
from chunker import split_into_chunks
from retriever import find_relevant_chunks

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

def ask_document(question, chunks):
    relevant_chunks = find_relevant_chunks(question, chunks)
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""You are a helpful healthcare assistant.
Answer the question based ONLY on the context below.
If the answer is not in the context, say 'I could not find that in the document.'

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def main():
    print("=" * 50)
    print("   Healthcare Document AI Assistant")
    print("   Powered by Claude + RAG")
    print("=" * 50)
    
    print("\nLoading document...")
    pdf_text = read_pdf("consumer_rights.pdf")
    chunks = split_into_chunks(pdf_text)
    print(f"Ready. {len(chunks)} chunks loaded from document.")
    print("\nAsk any question about the document.")
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            print("\nGoodbye!")
            break
        if not question.strip():
            continue
        print("\nClaude: ", end="")
        answer = ask_document(question, chunks)
        print(answer)
        print()


if __name__ == "__main__":
    main()