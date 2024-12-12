import os
import sqlite3
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="Personality-Based RAG", layout="wide")

# Database Initialization
def init_db():
    """Initialize the database and create necessary tables."""
    conn = sqlite3.connect("documents.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Call the database initialization function
init_db()

# Persistent Document Storage
def add_document_to_db(content):
    """Add a document's content to the database."""
    try:
        conn = sqlite3.connect("documents.db")
        c = conn.cursor()
        c.execute("INSERT INTO documents (content) VALUES (?)", (content,))
        conn.commit()
        conn.close()
    except sqlite3.OperationalError as e:
        st.error(f"Database error: {e}")

def fetch_documents_from_db():
    """Fetch all documents from the database."""
    conn = sqlite3.connect("documents.db")
    c = conn.cursor()
    c.execute("SELECT content FROM documents")
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

# File Upload Section
uploaded_file = st.sidebar.file_uploader(
    "Upload Documents (CSV or TXT):", type=["csv", "txt"]
)

def process_file(file):
    """Process uploaded files into usable data."""
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".txt"):
            return file.read().decode("utf-8").splitlines()
    except Exception as e:
        st.sidebar.error(f"Error processing file: {str(e)}")
        return None

# Load and Display File Data
if uploaded_file:
    data = process_file(uploaded_file)
    if data is not None:
        if isinstance(data, pd.DataFrame):
            st.write("Raw Data from File:", data.head())
            for doc in data.iloc[:, 0]:
                add_document_to_db(doc)
        else:
            st.write("Raw Text Data:", data[:10])
            for line in data:
                add_document_to_db(line)

# Embedding-based Vector Store Class
class EmbeddingVectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.docs = []
        self.embeddings = None

    def add_documents(self, documents):
        """Add and embed documents."""
        self.docs.extend(documents)
        new_embeddings = self.model.encode(documents, convert_to_tensor=True)
        self.embeddings = (
            new_embeddings
            if self.embeddings is None
            else torch.cat((self.embeddings, new_embeddings))
        )

    def search(self, query, top_k=5):
        """Retrieve top-k similar documents based on the query."""
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:top_k]
        return [(self.docs[i], float(similarities[i])) for i in top_indices]

# Response Generation
def generate_response(prompt):
    """Generate response using Llama 2 or similar model."""
    if 'llama_model' not in st.session_state:
        llama_model_name = "facebook/opt-1.3b"  # Adjust to your preferred model
        tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        llama_model = AutoModelForCausalLM.from_pretrained(
            llama_model_name, device_map="auto", torch_dtype=torch.float16
        )
        st.session_state.llama_model = llama_model
        st.session_state.tokenizer = tokenizer
    else:
        llama_model = st.session_state.llama_model
        tokenizer = st.session_state.tokenizer

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = llama_model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize Embedding Vector Store
vector_store = EmbeddingVectorStore()

# Load Existing Documents into Vector Store
documents = fetch_documents_from_db()
if documents:
    vector_store.add_documents(documents)

# App Structure
st.sidebar.title("Personality-Based RAG")

# Sidebar Personality Selector
personality_options = ["Thinking", "Feeling", "Logical", "Creative"]
personalities = st.sidebar.multiselect(
    "Select Personality Traits:", options=personality_options, default=["Thinking"]
)

# Chat History
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Query Handling
query = st.text_input("Ask your question:")
if query:
    retrieved_docs = vector_store.search(query)
    context = "\n".join([doc[0] for doc in retrieved_docs])

    # Display Retrieved Documents
    st.write("**Retrieved Documents:**")
    for doc, score in retrieved_docs:
        st.write(f"- {doc} (Score: {score:.2f})")

    # Generate Response
    prompt = f"Personality Traits: {', '.join(personalities)}\nContext: {context}\nQuestion: {query}\nAnswer:"
    response = generate_response(prompt)

    st.session_state.chat_history.append(("User", query))
    st.session_state.chat_history.append(("Assistant", response))

# Display Chat History
for role, message in st.session_state.chat_history:
    if role == "User":
        st.write(f"**User:** {message}")
    else:
        st.write(f"**Assistant:** {message}")

# Enhanced Features
if st.sidebar.checkbox("Enable Enhanced Features"):
    st.sidebar.info("Persistent storage and advanced embeddings are enabled.")

# Export Chat History
if st.sidebar.button("Export Chat History"):
    history_text = "\n".join([f"{role}: {msg}" for role, msg in st.session_state.chat_history])
    st.sidebar.download_button(
        label="Download History", data=history_text, file_name="chat_history.txt"
    )
