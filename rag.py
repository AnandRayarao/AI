import anthropic
import os
from dotenv import load_dotenv
from pdf_reader import read_pdf
from chunker import split_into_chunks
from embedder import get_embeddings, find_similar_chunks


load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

def ask_document(question, chunks, embeddings):
    """
    Find relevant chunks and ask Claude to answer
    based only on those chunks.
    """
    
    # Step 1 - find relevant chunks
    relevant_chunks = find_similar_chunks(question, chunks, embeddings)

    
    # Step 2 - join chunks into one context block
    context = "\n\n".join(relevant_chunks)
    
    # Step 3 - build prompt with context
    prompt = f"""You are a helpful assistant. 
Answer the question based ONLY on the context below.
If the answer is not in the context, say 'I could not find that in the document.'

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    # Step 4 - send to Claude
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return message.content[0].text


def main():
    print("Loading document...")
    pdf_text = read_pdf("AnandResume.pdf")
    chunks = split_into_chunks(pdf_text)
    embeddings = get_embeddings(chunks)  # add this line
    print(f"Ready. {len(chunks)} chunks loaded.")
    print("-" * 40)
    print("Ask questions about your HIPAA document.")
    print("Type 'quit' to exit.")

    while True:
        question = input("\nYou: ")
        if question.lower() == "quit":
            break
        answer = ask_document(question, chunks, embeddings)  # add embeddings here
        print(f"\nClaude: {answer}")


if __name__ == "__main__":
    main()