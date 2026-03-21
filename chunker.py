def split_into_chunks(text, chunk_size=500, overlap=200):
    """
    Split text into chunks of roughly chunk_size characters.
    overlap means each chunk shares 50 characters with the next
    -- this prevents cutting a sentence in a bad place.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap with next chunk

    return chunks


# Test it with your PDF
from pdf_reader import read_pdf

pdf_text = read_pdf("consumer_rights.pdf")
chunks = split_into_chunks(pdf_text)

print(f"Total characters in PDF: {len(pdf_text)}")
print(f"Total chunks created: {len(chunks)}")
print(f"\n--- CHUNK 1 ---")
print(chunks[0])
print(f"\n--- CHUNK 2 ---")
print(chunks[1])
print(f"\n--- CHUNK 3 ---")
print(chunks[2])
