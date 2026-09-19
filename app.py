import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import requests

st.set_page_config(page_title="Study Notes Chatbot", page_icon="📚", layout="centered")

# --- Custom Study Theme CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #FAF6F0;
    }
    h1 {
        color: #3B2F2F;
        font-family: 'Georgia', serif;
    }
    .stCaption {
        color: #6B5B4E;
    }
    div[data-testid="stFileUploader"] {
        background-color: #FFF8F0;
        border: 2px dashed #D4A574;
        border-radius: 12px;
        padding: 1rem;
    }
    .stButton > button {
        background-color: #A0522D;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #8B4513;
        color: white;
    }
    div[data-testid="stChatMessage"] {
        background-color: #FFFDF9;
        border-radius: 12px;
        border: 1px solid #E8DCC8;
        padding: 0.5rem;
    }
    .stChatInput {
        border-radius: 12px;
    }
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
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
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# --- Header ---
st.title("📚 Study Notes Chatbot")
st.caption("Upload your notes as a PDF and ask questions — powered by a locally running Llama 3.2 model via Ollama.")

model = load_embedding_model()

if "collection" not in st.session_state:
    st.session_state.collection = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Upload section ---
st.subheader("📄 Upload Your Notes")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")

if uploaded_file is not None:
    if st.button("📖 Process Document"):
        with st.spinner("Reading and indexing your notes..."):
            text = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(text)

            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_or_create_collection(name="notes")

            existing = collection.get()
            if existing['ids']:
                collection.delete(ids=existing['ids'])

            embeddings = model.encode(chunks).tolist()
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            collection.add(ids=ids, embeddings=embeddings, documents=chunks)

            st.session_state.collection = collection
            st.session_state.messages = []
        st.success(f"✅ Document processed successfully — {len(chunks)} chunks created.")

# --- Chat section ---
if st.session_state.collection is not None:
    st.divider()
    st.subheader("💬 Ask Your Notes")

    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "📘"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

    query = st.chat_input("Ask a question about your document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.write(query)

        with st.chat_message("assistant", avatar="📘"):
            with st.spinner("Generating answer..."):
                query_embedding = model.encode([query]).tolist()
                results = st.session_state.collection.query(
                    query_embeddings=query_embedding, n_results=2
                )
                context = "\n\n".join(results['documents'][0])

                prompt = f"""You are a helpful study assistant. Answer the question using ONLY the context provided below. If the answer is not in the context, say "This information isn't available in the notes."

Context:
{context}

Question: {query}

Answer:"""
                answer = ask_ollama(prompt)
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👆 Upload a PDF and click 'Process Document' to get started.")