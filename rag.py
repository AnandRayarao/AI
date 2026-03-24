import anthropic
import os
from dotenv import load_dotenv
from pdf_reader import read_pdf
from chunker import split_into_chunks
from vector_store import add_documents, search
from multi_doc_store import load_all_documents, search_all



load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

def ask_document(question):
   
    results = search_all(question)
    
    # Build context with sources
    context = ""
    for r in results:
        context += f"From {r['source']}:\n{r['text']}\n\n"
    prompt = f"""You are a helpful healthcare assistant.
Answer the question based ONLY on the context below.
If the answer is not in the context say 'I could not find that in the document.'

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
    print("Loading all documents...")
    load_all_documents()
    print("Ready. Ask questions about your documents.")
    print("Type 'quit' to exit.")

    while True:
        question = input("\nYou: ")
        if question.lower() == "quit":
            break
        answer = ask_document(question)
        print(f"\nClaude: {answer}")


if __name__ == "__main__":
    main()