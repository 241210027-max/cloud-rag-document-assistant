import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load the secret API keys
load_dotenv()

# 2. Safely grab keys
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
    pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"]
except (KeyError, FileNotFoundError):
    gemini_api_key = None
    pinecone_api_key = None
    st.error("Missing API Keys! Please check your Streamlit Secrets.")

if pinecone_api_key:
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="centered")
st.title("📄 Cloud RAG Document Assistant")
st.write("Upload a PDF, and the AI will answer your questions strictly based on its contents.")
st.markdown("---")

@st.cache_resource
def get_existing_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=768,
        google_api_key=gemini_api_key 
    )
    vectorstore = PineconeVectorStore(
        index_name="document-assistant", 
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )
    return vectorstore

vector_db = get_existing_vectorstore()

# --- FILE UPLOAD WIDGET ---
uploaded_file = st.file_uploader("Upload a new PDF document", type="pdf")

# If the user uploads a file, process it
if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Extracting text and uploading to database..."):
            
            # 1. Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # 2. Load the PDF
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load()
            
            # 3. Chunk the text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)
            
            # 4. Upload the new vectors to Pinecone
            vector_db.add_documents(chunks)
            
            # 5. Clean up the temporary file
            os.remove(tmp_file_path)
            
            st.success("Document successfully processed and added to the AI's memory!")

st.markdown("---")

# --- USER INPUT ---
user_query = st.text_input("What would you like to know about the uploaded documents?")

if st.button("Ask AI"):
    if user_query:
        with st.spinner("Searching documents and generating answer..."):
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite", 
                temperature=0,
                max_retries=5,
                google_api_key=gemini_api_key
            )
            
            system_prompt = (
                "You are an intelligent assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer based on the context, say that you don't know. "
                "Keep the answer concise and strictly based on the document.\n\n"
                "Context:\n{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            response = rag_chain.invoke({"input": user_query})
            
            st.success("Done!")
            st.write(response["answer"])
    else:
        st.warning("Please enter a question first.")