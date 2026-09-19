from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import requests

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

def ask_ollama(prompt, model="llama3.2:3b"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

if __name__ == "__main__":
    pdf_path = "data/Dynamic_Memory_Allocation_C.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="notes")

    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])

    print("Storing chunks in ChromaDB...")
    embeddings = model.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks)
    print(f"Stored {len(chunks)} chunks!\n")

    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        query = input("Your question: ")
        if query.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        query_embedding = model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=2)
        retrieved_chunks = results['documents'][0]
        context = "\n\n".join(retrieved_chunks)

        prompt = f"""You are a helpful study assistant. Answer the question using ONLY the context provided below. If the answer is not in the context, say "Mujhe yeh information notes mein nahi mili."

Context:
{context}

Question: {query}

Answer:"""

        print("\nThinking...")
        answer = ask_ollama(prompt)
        print(f"\nAnswer: {answer}\n")
        print("-" * 50)