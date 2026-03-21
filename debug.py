from pdf_reader import read_pdf
from chunker import split_into_chunks

pdf_text = read_pdf("AnandResume.pdf")
chunks = split_into_chunks(pdf_text)

print(f"Total chunks: {len(chunks)}")
print("\n--- ALL CHUNKS ---")
for i, chunk in enumerate(chunks):
    print(f"\n=== CHUNK {i+1} ===")
    print(chunk)
    print("-" * 40)