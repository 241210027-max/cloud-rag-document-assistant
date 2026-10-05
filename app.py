import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from rag.ingestion import process_pdf
from rag.retrieval import (
    get_vectorstore,
    add_documents,
    get_retriever,
)
from rag.generation import (
    get_llm,
    create_rag_chain,
    ask_question,
    get_sources,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="centered",
)
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .document-card {
            padding: 1rem 1.25rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            margin: 1rem 0;
        }

        .source-card {
            padding: 0.75rem 1rem;
            border-left: 4px solid #4f46e5;
            background: rgba(79, 70, 229, 0.05);
            border-radius: 6px;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# API Keys
# --------------------------------------------------

try:
    gemini_api_key = (
        os.getenv("GEMINI_API_KEY")
        or st.secrets["GEMINI_API_KEY"]
    )

    pinecone_api_key = (
        os.getenv("PINECONE_API_KEY")
        or st.secrets["PINECONE_API_KEY"]
    )

except (KeyError, FileNotFoundError):
    gemini_api_key = None
    pinecone_api_key = None


if not gemini_api_key or not pinecone_api_key:
    st.error(
        "Missing API keys. Please configure GEMINI_API_KEY "
        "and PINECONE_API_KEY."
    )
    st.stop()


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "document_namespace" not in st.session_state:
    st.session_state.document_namespace = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📄 Cloud RAG Document Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Upload a PDF and ask questions based strictly on its contents.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("---")


# --------------------------------------------------
# Document Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type="pdf",
)


if uploaded_file is not None:

    st.info(
        f"Selected document: **{uploaded_file.name}**"
    )

    if st.button("Process Document", type="primary"):

        with st.spinner("Processing document..."):

            try:
                # Create a unique namespace for this document
                namespace = f"doc-{uuid.uuid4().hex}"

                # Extract and chunk PDF
                chunks = process_pdf(uploaded_file)

                # Connect to Pinecone namespace
                vectorstore = get_vectorstore(
                    gemini_api_key=gemini_api_key,
                    pinecone_api_key=pinecone_api_key,
                    namespace=namespace,
                )

                # Store chunks
                add_documents(
                    vectorstore=vectorstore,
                    documents=chunks,
                )

                # Save document information
                st.session_state.document_namespace = namespace
                st.session_state.document_name = uploaded_file.name
                st.session_state.document_processed = True

                st.success(
                    f"Successfully processed **{uploaded_file.name}**."
                )

            except Exception as e:

                st.error(
                    "Something went wrong while processing the document."
                )

                st.exception(e)


# --------------------------------------------------
# Document Status
# --------------------------------------------------

if st.session_state.document_processed:

    st.markdown("---")

    st.subheader("📚 Active Document")

    with st.container(border=True):
        st.markdown(
            f"**{st.session_state.document_name}**"
        )

        st.caption(
            "This document is ready for questions."
        )


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

st.markdown("---")

user_query = st.text_input(
    "What would you like to know about the document?"
)


if st.button("Ask AI"):

    if not st.session_state.document_processed:

        st.warning(
            "Please upload and process a PDF before asking a question."
        )

    elif not user_query.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        with st.spinner(
            "Searching the document and generating an answer..."
        ):

            try:

                # Connect to the same document namespace
                vectorstore = get_vectorstore(
                    gemini_api_key=gemini_api_key,
                    pinecone_api_key=pinecone_api_key,
                    namespace=st.session_state.document_namespace,
                )

                # Create retriever
                retriever = get_retriever(
                    vectorstore=vectorstore,
                    k=3,
                )

                # Create Gemini + RAG chain
                llm = get_llm(
                    gemini_api_key
                )

                rag_chain = create_rag_chain(
                    llm=llm,
                    retriever=retriever,
                )

                # Ask question
                response = ask_question(
                    rag_chain=rag_chain,
                    question=user_query,
                )

                # Display answer
                st.success("Done!")

                st.write(
                    response["answer"]
                )

                # Display retrieved sources
                sources = get_sources(response)

                if sources:
                  st.markdown("---")
                  st.subheader("📚 Sources")

                  for source in sources:
                   with st.container(border=True):
                    if source["page"] is not None:
                     st.markdown(
                      f"📄 **{source['source']}**  \n"
                      f"Page **{source['page']}**"
                )
                    else:
                     st.markdown(
                     f"📄 **{source['source']}**"
                )

            except Exception as e:

                st.error(
                    "Something went wrong while generating the answer."
                )

                st.exception(e)