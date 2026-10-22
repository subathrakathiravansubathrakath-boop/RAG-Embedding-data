# 📚 Chat With Your Documents — RAG Application

A beginner-friendly Retrieval-Augmented Generation (RAG) application built with Python, Streamlit, OpenAI embeddings, FAISS, and an OpenAI chat model.

## RAG Pipeline

**Document → Chunking → Embeddings → Vector DB → Retrieval → LLM → Answer**

## Features

- Upload PDF, DOCX, TXT, or Markdown documents
- Text extraction and chunking with overlap
- Semantic embeddings using `text-embedding-3-small`
- FAISS vector database/index for similarity search
- Top-k relevant chunk retrieval
- LLM answer generation grounded in retrieved context
- Chat history
- Retrieved-context inspection
- Basic hallucination reduction by instructing the LLM not to invent information

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter your OpenAI API key in the sidebar.

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. Open Streamlit Community Cloud.
4. Select the repository and `app.py`.
5. Deploy the application.

For a public deployment, prefer Streamlit Secrets instead of entering the API key directly.

## Suggested demo

Upload a PDF containing notes or study material and ask:
- "What is the main topic?"
- "Explain the key points."
- "What are the advantages mentioned?"
- "Give a short summary."

## Project learning outcomes

- Understand LLM hallucinations
- Understand RAG architecture
- Learn embeddings and semantic similarity
- Perform document processing and chunking
- Use a vector index for retrieval
- Connect retrieved context to an LLM
- Build a document-based Q&A application
