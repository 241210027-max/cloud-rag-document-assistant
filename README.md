
# Cloud RAG Document Assistant

A cloud-based Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents and ask questions about their contents.

The application uses Gemini for embeddings and answer generation, Pinecone for vector storage and semantic retrieval, and Streamlit for the user interface.

## Features

- Upload PDF documents through a Streamlit interface
- Extract and split PDF content into smaller chunks
- Generate semantic embeddings using Gemini
- Store document embeddings in Pinecone
- Retrieve relevant document sections for each question
- Generate answers grounded in the uploaded document
- Display document sources and page references
- Isolate uploaded documents using unique Pinecone namespaces
- Secure API-key configuration through environment variables or Streamlit secrets

## Architecture

```text
                  ┌─────────────────┐
                  │   PDF Upload    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  PDF Extraction │
                  │  & Chunking     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Gemini Embedding│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    Pinecone     │
                  │  Vector Store   │
                  └────────┬────────┘
                           │
                    User Question
                           │
                           ▼
                  ┌─────────────────┐
                  │ Semantic Search │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Gemini LLM      │
                  │ Answer Generation│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Answer + Sources│
                  └─────────────────┘
```

## Tech Stack

### Frontend
- Streamlit

### Backend / RAG Pipeline
- Python
- LangChain
- Gemini API
- Pinecone

### Document Processing
- PyPDF
- Recursive Character Text Splitter

## Project Structure

```text
cloud-rag-document-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── rag/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── retrieval.py
│   └── generation.py
│
├── assets/
│   └── sample.pdf
│
└── tests/
```

## How It Works

### 1. Upload a document

The user uploads a PDF through the Streamlit interface.

### 2. Extract and split the document

The PDF is loaded and divided into smaller text chunks using LangChain text splitting utilities.

### 3. Generate embeddings

Each document chunk is converted into a numerical embedding using Gemini's embedding model.

### 4. Store embeddings

The embeddings are stored in Pinecone under a unique namespace for the uploaded document.

### 5. Retrieve relevant context

When the user asks a question, the application performs semantic similarity search against the document's vector namespace.

### 6. Generate the answer

The retrieved document context is passed to Gemini, which generates an answer based strictly on the retrieved content.

### 7. Display sources

The application displays the document source and relevant page references used during retrieval.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/241210027-max/cloud-rag-document-assistant.git
cd cloud-rag-document-assistant
```

### 2. Create a virtual environment

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

Do not commit the `.env` file.

The repository includes `.env.example` as a template.

### 5. Configure Pinecone

Create a Pinecone index named:

```text
document-assistant
```

The application uses a 768-dimensional Gemini embedding configuration.

### 6. Run the application

```bash
streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

## Usage

1. Upload a PDF.
2. Click **Process Document**.
3. Enter a question about the document.
4. Click **Ask AI**.
5. Review the generated answer and source references.

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | API key used for Gemini embeddings and answer generation |
| `PINECONE_API_KEY` | API key used to access Pinecone |

## Current Limitations

- The application currently processes one active document per session.
- Uploaded document namespaces are generated dynamically.
- Vector cleanup for previously processed documents is not currently automated.
- Authentication and persistent user accounts are not currently implemented.

## Future Improvements

- Add conversation history
- Add document management and deletion
- Add automated vector cleanup
- Add authentication and user accounts
- Add automated tests
- Add production deployment
- Improve retrieval evaluation and monitoring

## License

This project is intended for educational and portfolio purposes.
