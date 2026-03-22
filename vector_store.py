import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create ChromaDB client — saves to disk automatically
client = chromadb.PersistentClient(path="./chroma_db")

def create_collection(collection_name="healthcare_docs"):
    """Create or get existing collection"""
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def add_documents(chunks, collection_name="healthcare_docs"):
    """Add chunks to vector database"""
    collection = create_collection(collection_name)
    
    # Check if already populated
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} chunks. Skipping embedding.")
        return collection
    
    print(f"Adding {len(chunks)} chunks to vector database...")
    
    # Create embeddings
    embeddings = model.encode(chunks).tolist()
    
    # Add to ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    print(f"Done. {collection.count()} chunks stored in database.")
    return collection

def search(question, collection_name="healthcare_docs", top_n=3):
    """Search for relevant chunks using question"""
    collection = create_collection(collection_name)
    
    # Embed the question
    question_embedding = model.encode([question]).tolist()
    
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_n
    )
    
    # Return just the text chunks
    return results['documents'][0]


# Test it
from pdf_reader import read_pdf
from chunker import split_into_chunks

print("Loading PDF...")
pdf_text = read_pdf("consumer_rights.pdf")
chunks = split_into_chunks(pdf_text)

# Add to vector database
add_documents(chunks)

# Search test
question = "What are my rights under HIPAA?"
print(f"\nQuestion: {question}")
results = search(question)

print("\nTop 3 relevant chunks:")
for i, chunk in enumerate(results):
    print(f"\n--- chiii {i+1} ---")
    print(chunk[:200])