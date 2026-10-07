# Cloud RAG Document Assistant

An AI-powered document question-answering application that allows users to upload PDF documents and ask questions using Retrieval-Augmented Generation (RAG).

## 🚀 Live Demo

[Live Demo](https://cloud-rag-document-assistant-g5svttb3sldjfq8qjeh9tx.streamlit.app/)

## ✨ Features

- Upload PDF documents
- Extract and split document content
- Generate semantic embeddings
- Store embeddings in Pinecone
- Retrieve relevant document sections
- Generate grounded answers using Google Gemini
- Display relevant document sources and page references
- Isolate documents using Pinecone namespaces
- Secure API-key configuration using environment variables / Streamlit secrets

## 🏗️ How It Works

```text
PDF Upload
    ↓
Text Extraction
    ↓
Document Chunking
    ↓
Gemini Embeddings
    ↓
Pinecone Vector Database
    ↓
Semantic Retrieval
    ↓
Google Gemini
    ↓
Grounded Answer + Sources

🛠️ Tech Stack
- Python
- Streamlit
- LangChain
- Google Gemini
- Pinecone
- PDF processing
- Vector embeddings
⚙️ Run Locally
Clone the repository:
git clone https://github.com/241210027-max/cloud-rag-document-assistant.git
cd cloud-rag-document-assistant

Install dependencies:
pip install -r requirements.txt

Configure your API keys using environment variables or Streamlit secrets.
Run:
streamlit run app.py

🔐 Security
API keys are stored using environment variables or Streamlit secrets and are not committed to the repository.
👨‍💻 Author
Ayush Kumar Tripathi
B.Tech CSE — NIT Delhi
