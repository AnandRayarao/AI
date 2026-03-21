from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model
# This small model runs on your computer for free
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(chunks):
    """Turn a list of text chunks into embeddings"""
    print("Creating embeddings for all chunks...")
    embeddings = model.encode(chunks)
    print(f"Done. Created {len(embeddings)} embeddings.")
    return embeddings

def find_similar_chunks(question, chunks, embeddings, top_n=5):
    """
    Find most relevant chunks using meaning
    not just keyword matching
    """
    # Turn question into embedding
    question_embedding = model.encode([question])[0]

    # Compare question with every chunk
    scores = []
    for i, chunk_embedding in enumerate(embeddings):
        # Calculate similarity score
        similarity = np.dot(question_embedding, chunk_embedding) / (
            np.linalg.norm(question_embedding) *
            np.linalg.norm(chunk_embedding)
        )
        scores.append((similarity, i, chunks[i]))

    # Sort by similarity highest first
    scores.sort(reverse=True)

    # Return top 3 most similar chunks
    top_chunks = [chunk for score, i, chunk in scores[:top_n]]
    return top_chunks


# Test it
from pdf_reader import read_pdf
from chunker import split_into_chunks

print("Loading PDF...")
pdf_text = read_pdf("AnandResume.pdf")
chunks = split_into_chunks(pdf_text)

# Create embeddings for all chunks
embeddings = get_embeddings(chunks)

# Test with a question
question = "Can I see my medical records?"
print(f"\nQuestion: {question}")
print("\nTop 3 relevant chunks found:")

relevant = find_similar_chunks(question, chunks, embeddings)
for i, chunk in enumerate(relevant):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk[:200])