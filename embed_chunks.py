from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

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
    
    print(f"Total chunks: {len(chunks)}")
    

    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings...")
    embeddings = model.encode(chunks)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Har chunk ek {embeddings.shape[1]}-dimensional vector ban gaya hai")
    print(f"\nFirst embedding ke pehle 10 numbers:")
    print(embeddings[0][:10])