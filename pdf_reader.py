import PyPDF2

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)
        print(f"Total pages found: {total_pages}")
        text = ""
        
        for page_num in range(total_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
            print(f"Reading page {page_num + 1}...")
    
    return text


pdf_text = read_pdf("consumer_rights.pdf")

print("\n--- FIRST 500 CHARACTERS ---")
print(pdf_text[:500])
print(f"\n--- TOTAL CHARACTERS READ: {len(pdf_text)} ---")

