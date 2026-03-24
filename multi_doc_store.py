import chromadb
from sentence_transformers import SentenceTransformer
from pdf_reader import read_pdf
from chunker import split_into_chunks
import os

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db_multi")

def get_collection():
    return client.get_or_create_collection(
        name="multi_docs",
        metadata={"hnsw:space": "cosine"}
    )

def load_all_documents(docs_folder="docs"):
    """Load all PDFs from a folder into ChromaDB"""
    collection = get_collection()

    # Get all PDF files
    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs: {pdf_files}")

    for pdf_file in pdf_files:
        file_path = os.path.join(docs_folder, pdf_file)

        # Check if already loaded
        existing = collection.get(where={"source": pdf_file})
        if len(existing['ids']) > 0:
            print(f"Skipping {pdf_file} — already in database")
            continue

        print(f"Loading {pdf_file}...")
        pdf_text = read_pdf(file_path)
        chunks = split_into_chunks(pdf_text)

        # Create embeddings
        embeddings = model.encode(chunks).tolist()

        # Add with source metadata
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{pdf_file}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"source": pdf_file} for _ in chunks]
        )
        print(f"Added {len(chunks)} chunks from {pdf_file}")

    print(f"\nTotal chunks in database: {collection.count()}")
    return collection

def search_all(question, top_n=3):
    """Search across all documents"""
    collection = get_collection()

    question_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_n,
        include=['documents', 'metadatas']
    )

    # Return chunks with their source
    chunks_with_source = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        chunks_with_source.append({
            "text": doc,
            "source": meta['source']
        })

    return chunks_with_source


# Test it
load_all_documents()

question = "What are my rights?"
print(f"\nQuestion: {question}")
results = search_all(question)

for i, result in enumerate(results):
    print(f"\n--- Result {i+1} from {result['source']} ---")
    print(result['text'][:200])