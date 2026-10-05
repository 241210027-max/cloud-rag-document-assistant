from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


INDEX_NAME = "document-assistant"


def get_embeddings(gemini_api_key):
    """Create the Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=768,
        google_api_key=gemini_api_key,
    )


def get_vectorstore(gemini_api_key, pinecone_api_key, namespace):
    """Create a Pinecone vector store for one document namespace."""

    embeddings = get_embeddings(gemini_api_key)

    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key,
        namespace=namespace,
    )


def add_documents(vectorstore, documents):
    """Store document chunks in the vector store."""
    return vectorstore.add_documents(documents)


def get_retriever(vectorstore, k=3):
    """Create a retriever for the current document namespace."""
    return vectorstore.as_retriever(
        search_kwargs={"k": k}
    )