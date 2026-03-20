def find_relevant_chunks(question, chunks, top_n=3):
    """
    Find the most relevant chunks for a question
    using simple keyword matching.
    """
    question_words = question.lower().split()
    
    scored_chunks = []
    
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        
        # Count how many question words appear in this chunk
        score = 0
        for word in question_words:
            if word in chunk_lower:
                score += 1
        
        scored_chunks.append((score, i, chunk))
    
    # Sort by score highest first
    scored_chunks.sort(reverse=True)
    
    # Return top N most relevant chunks
    top_chunks = [chunk for score, i, chunk in scored_chunks[:top_n]]
    
    return top_chunks


# Test it
from pdf_reader import read_pdf
from chunker import split_into_chunks

pdf_text = read_pdf("consumer_rights.pdf")
chunks = split_into_chunks(pdf_text)

question = "What are my rights under HIPAA?"
relevant = find_relevant_chunks(question, chunks)

print(f"Question: {question}")
print(f"\nTop 3 most relevant chunks found:")
for i, chunk in enumerate(relevant):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk[:200])