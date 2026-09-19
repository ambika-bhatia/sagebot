from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

if __name__ == "__main__":
    pdf_path = "data/Dynamic_Memory_Allocation_C.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    
    print(f"Total chunks created: {len(chunks)}")
    print("\n--- First chunk ---")
    print(chunks[0])
    print("\n--- Second chunk ---")
    print(chunks[1] if len(chunks) > 1 else "Only 1 chunk created")