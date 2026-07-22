import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the secret API keys from our hidden .env file
load_dotenv()

def load_pdf_document(file_path):
    """
    Takes a path to a PDF file, extracts the raw text page by page,
    and returns a list of LangChain Document objects.
    """
    print(f"--- Starting translation of: {file_path} ---")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"--- Successfully loaded {len(pages)} pages ---")
    return pages

def split_text_into_chunks(pages):
    """
    Takes the loaded pages and chunks them into smaller semantic blocks
    so the LLM can digest them easily without losing context.
    """
    print("--- Starting text chunking process ---")
    
    # Configure our intelligent text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Max characters per chunk
        chunk_overlap=200,     # Overlap to preserve context between chunks
        length_function=len
    )
    
    # Split the document pages into smaller chunks
    chunks = text_splitter.split_documents(pages)
    
    print(f"--- Successfully split document into {len(chunks)} chunks ---")
    return chunks

if __name__ == "__main__":
    # Test our loader and splitter with a local sample file
    sample_file = "sample.pdf" 
    
    if os.path.exists(sample_file):
        # Step 2: Load Document
        document_pages = load_pdf_document(sample_file)
        
        # Step 3: Chunk Text
        document_chunks = split_text_into_chunks(document_pages)
        
        # Verify it worked by printing the first chunk
        print("\n--- Snippet of Chunk 1 Content ---")
        print(document_chunks[0].page_content)
    else:
        print(f"Error: Please place a file named '{sample_file}' in this folder to test.")