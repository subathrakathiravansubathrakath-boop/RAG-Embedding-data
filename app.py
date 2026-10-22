import os
import tempfile
import streamlit as st
import numpy as np
import faiss
from openai import OpenAI
from pypdf import PdfReader
from docx import Document

st.set_page_config(page_title="Chat With Your Documents", page_icon="📚", layout="wide")

st.title("📚 Chat With Your Documents")
st.caption("RAG: Document → Chunking → Embeddings → Vector DB → Retrieval → LLM → Answer")

api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) if os.environ.get("OPENAI_API_KEY") else None

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        data = uploaded_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(data)
            path = tmp.name
        doc = Document(path)
        os.unlink(path)
        return "\n".join(p.text for p in doc.paragraphs)
    if name.endswith(".txt") or name.endswith(".md"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    return ""

def chunk_text(text, chunk_size=900, overlap=150):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks

def embed_texts(texts):
    if not client:
        raise RuntimeError("Please enter your OpenAI API key in the sidebar.")
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors

def build_vector_store(chunks):
    vectors = embed_texts(chunks)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, vectors

def retrieve(query, chunks, index, top_k=4):
    q = embed_texts([query])
    scores, ids = index.search(q, min(top_k, len(chunks)))
    results = [(chunks[i], float(scores[0][rank])) for rank, i in enumerate(ids[0]) if i >= 0]
    return results

def answer_question(question, retrieved):
    context = "\n\n--- Retrieved chunk ---\n\n".join(
        f"[Similarity: {score:.3f}]\n{chunk}" for chunk, score in retrieved
    )
    system = """You are a document question-answering assistant.
Answer using only the retrieved context below. If the answer is not present,
say that the information was not found in the uploaded document.
Do not invent facts. Keep the answer clear and concise."""
    prompt = f"""Retrieved context:
{context}

Question:
{question}"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

uploaded = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "md", "docx"],
    help="Upload a PDF, TXT, Markdown, or DOCX file."
)

if uploaded:
    if not client:
        st.warning("Enter your OpenAI API key in the sidebar to process the document.")
    else:
        if st.button("🔎 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                text = extract_text(uploaded)
                if not text.strip():
                    st.error("No readable text was found in the document.")
                else:
                    chunks = chunk_text(text)
                    index, vectors = build_vector_store(chunks)
                    st.session_state["chunks"] = chunks
                    st.session_state["index"] = index
                    st.session_state["filename"] = uploaded.name
                    st.session_state["messages"] = []
                    st.success(f"Processed {uploaded.name}: {len(chunks)} chunks indexed.")

if "index" in st.session_state:
    st.info(f"📄 Active document: **{st.session_state['filename']}**")

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your document...")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant information..."):
                retrieved = retrieve(
                    question,
                    st.session_state["chunks"],
                    st.session_state["index"],
                    top_k=4,
                )
                answer = answer_question(question, retrieved)
                st.markdown(answer)

                with st.expander("🔍 Retrieved context"):
                    for i, (chunk, score) in enumerate(retrieved, 1):
                        st.markdown(f"**Chunk {i} — similarity {score:.3f}**")
                        st.write(chunk)

        st.session_state["messages"].append({"role": "assistant", "content": answer})
else:
    st.markdown("""
### How it works
1. Upload a document.
2. The document is split into smaller chunks.
3. Each chunk is converted into an embedding.
4. Embeddings are stored in a FAISS vector index.
5. Your question is embedded and compared using semantic similarity.
6. The most relevant chunks are sent to the LLM.
7. The LLM generates an answer grounded in the retrieved context.

**Supported files:** PDF, DOCX, TXT, Markdown.
""")
