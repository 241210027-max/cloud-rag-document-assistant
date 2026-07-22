import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load the secret API keys
load_dotenv()

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="centered")
st.title("📄 Cloud RAG Document Assistant")
st.write("Ask any question about the uploaded document, and the AI will answer strictly based on the text.")
st.markdown("---")

@st.cache_resource
def get_existing_vectorstore():
    """
    Connects to Pinecone. We use @st.cache_resource so Streamlit doesn't 
    re-establish the database connection every time the user clicks a button.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=768
    )
    vectorstore = PineconeVectorStore(
        index_name="document-assistant", 
        embedding=embeddings
    )
    return vectorstore

# Connect to the database
vector_db = get_existing_vectorstore()

# --- USER INPUT ---
user_query = st.text_input("What would you like to know about the document?")

if st.button("Ask AI"):
    if user_query:
        # Show a loading spinner while the cloud does the work
        with st.spinner("Searching document and generating answer..."):
            
            # 1. Initialize the Brain (Using the fast Lite model)
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite", 
                temperature=0,
                max_retries=5
            )
            
            # 2. Create the Prompt
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
            
            # 3. Create the Chains
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            # 4. Get the response
            response = rag_chain.invoke({"input": user_query})
            
            # 5. Display the result on the web page
            st.success("Done!")
            st.write(response["answer"])
    else:
        st.warning("Please enter a question first.")