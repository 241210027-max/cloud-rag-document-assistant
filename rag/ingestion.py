import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF and split it into chunks.

    Returns:
        list: Chunked LangChain Document objects with source metadata.
    """
    tmp_file_path = None

    try:
        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Extract pages
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()

        # Add useful source metadata
        filename = uploaded_file.name

        for page in pages:
            page.metadata["source"] = filename

        # Split into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(pages)

        return chunks

    finally:
        # Always clean up the temporary PDF
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)