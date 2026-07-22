import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load the secret API keys (for local development)
load_dotenv()

# 2. Safely grab keys whether we are running locally (.env) or in the Cloud (st.secrets)
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
    pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"]
except (KeyError, FileNotFoundError):
    gemini_api_key = None
    pinecone_api_key = None
    st.error("Missing API Keys! Please check your Streamlit Secrets.")

# Ensure Pinecone can find its key in the cloud environment
if pinecone_api_key:
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="centered")
st.title("📄 Cloud RAG Document Assistant")
st.write("Ask any question about the uploaded document, and the AI will answer strictly based on the text.")
st.markdown("---")

@st.cache_resource
def get_existing_vectorstore():
    # Explicitly hand-deliver the Google API key to bypass Pydantic validation errors
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

# Connect to the database
vector_db = get_existing_vectorstore()

# --- USER INPUT ---
user_query = st.text_input("What would you like to know about the document?")

if st.button("Ask AI"):
    if user_query:
        with st.spinner("Searching document and generating answer..."):
            
            # Explicitly hand-deliver the API key to the Brain as well
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