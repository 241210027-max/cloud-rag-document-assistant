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
    page_title="Cloud RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown(
    """
<style>

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 2rem;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        rgba(79, 70, 229, 0.12),
        rgba(59, 130, 246, 0.08)
    );
    border: 1px solid rgba(79, 70, 229, 0.18);
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 750;
    letter-spacing: -0.04em;
    margin-bottom: 0.35rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #6b7280;
    max-width: 760px;
    line-height: 1.6;
}

.section-card {
    padding: 1.4rem;
    border-radius: 16px;
    border: 1px solid rgba(128, 128, 128, 0.20);
    background: rgba(128, 128, 128, 0.025);
    margin-bottom: 1rem;
}

.document-card {
    padding: 1.1rem 1.25rem;
    border-radius: 14px;
    border: 1px solid rgba(79, 70, 229, 0.18);
    background: rgba(79, 70, 229, 0.045);
}

.answer-card {
    padding: 1.4rem 1.5rem;
    border-radius: 16px;
    border: 1px solid rgba(79, 70, 229, 0.20);
    background: rgba(79, 70, 229, 0.035);
    margin-top: 1rem;
    line-height: 1.7;
}

.eyebrow {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.35rem;
}

.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.muted {
    color: #6b7280;
    font-size: 0.9rem;
}

.sidebar-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.sidebar-text {
    color: #6b7280;
    font-size: 0.9rem;
    line-height: 1.5;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 2.7rem;
}

[data-testid="stFileUploader"] {
    border-radius: 14px;
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

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">📚 RAG Assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-text">'
        "Upload a document and ask questions grounded in its contents."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### Current Document")

    if st.session_state.document_processed:

        st.success("Document ready")

        st.caption(
            st.session_state.document_name
        )

        st.metric(
            "Chunks",
            st.session_state.document_chunks,
        )

        if st.button(
            "Clear Document",
            use_container_width=True,
        ):

            st.session_state.document_namespace = None
            st.session_state.document_name = None
            st.session_state.document_processed = False
            st.session_state.document_chunks = 0
            st.session_state.chat_history = []

            st.rerun()

    else:

        st.info("No document loaded")

    st.markdown("---")

    st.markdown("### Pipeline")

    st.caption("📄 PDF ingestion")
    st.caption("🧩 Text chunking")
    st.caption("🧠 Gemini embeddings")
    st.caption("🔎 Pinecone retrieval")
    st.caption("✨ Gemini generation")


# --------------------------------------------------
# Hero Header
# --------------------------------------------------

st.markdown(
    """
<div class="hero">
<div class="eyebrow">AI-Powered Document Search</div>
<div class="hero-title">Cloud RAG Document Assistant</div>
<div class="hero-subtitle">
Upload a PDF, search its contents using semantic retrieval,
and get concise answers grounded in the document.
</div>
</div>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Upload Section
# --------------------------------------------------

st.markdown(
    """
<div class="section-card">
<div class="card-title">📄 Upload your document</div>
<div class="muted">Supported format: PDF</div>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type="pdf",
    label_visibility="collapsed",
)


# --------------------------------------------------
# Process Uploaded Document
# --------------------------------------------------

if uploaded_file is not None:

    st.markdown(
        f"""
<div class="document-card">
<div class="eyebrow">Selected Document</div>
<div class="card-title">📄 {uploaded_file.name}</div>
<div class="muted">Ready to be processed</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    if st.button(
        "⚡ Process Document",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Extracting, embedding, and indexing your document..."
        ):

            try:

                namespace = f"doc-{uuid.uuid4().hex}"

                chunks = process_pdf(uploaded_file)

                vectorstore = get_vectorstore(
                    gemini_api_key=gemini_api_key,
                    pinecone_api_key=pinecone_api_key,
                    namespace=namespace,
                )

                add_documents(
                    vectorstore=vectorstore,
                    documents=chunks,
                )

                st.session_state.document_namespace = namespace
                st.session_state.document_name = uploaded_file.name
                st.session_state.document_processed = True
                st.session_state.document_chunks = len(chunks)
                st.session_state.chat_history = []

                st.success(
                    f"Successfully processed {uploaded_file.name}"
                )

            except Exception as e:

                st.error(
                    "Something went wrong while processing the document."
                )

                st.exception(e)


# --------------------------------------------------
# Active Document
# --------------------------------------------------

if st.session_state.document_processed:

    st.markdown("---")

    st.markdown(
        """
<div class="eyebrow">Active Document</div>
""",
        unsafe_allow_html=True,
    )

    with st.container(border=True):

        col1, col2 = st.columns([4, 1])

        with col1:

            st.markdown(
                f"### 📄 {st.session_state.document_name}"
            )

            st.caption(
                "Document indexed and ready for semantic search."
            )

        with col2:

            st.metric(
                "Chunks",
                st.session_state.document_chunks,
            )


# --------------------------------------------------
# Question Section
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
<div class="eyebrow">Ask Your Document</div>
<div class="card-title">What would you like to know?</div>
""",
    unsafe_allow_html=True,
)

user_query = st.text_input(
    "Question",
    placeholder=(
        "e.g. What are the key topics covered in this document?"
    ),
    label_visibility="collapsed",
)


# --------------------------------------------------
# Example Questions
# --------------------------------------------------

if st.session_state.document_processed:

    st.caption("Try an example:")

    example_columns = st.columns(3)

    examples = [
        "What is this document about?",
        "What are the key topics?",
        "Summarize the important points.",
    ]

    for column, example in zip(
        example_columns,
        examples,
    ):

        with column:

            if st.button(
                example,
                use_container_width=True,
            ):
                user_query = example


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

if st.button(
    "✨ Ask AI",
    type="primary",
    use_container_width=True,
):

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

                vectorstore = get_vectorstore(
                    gemini_api_key=gemini_api_key,
                    pinecone_api_key=pinecone_api_key,
                    namespace=st.session_state.document_namespace,
                )

                retriever = get_retriever(
                    vectorstore=vectorstore,
                    k=3,
                )

                llm = get_llm(
                    gemini_api_key=gemini_api_key,
                )

                rag_chain = create_rag_chain(
                    llm=llm,
                    retriever=retriever,
                )

                response = ask_question(
                    rag_chain=rag_chain,
                    question=user_query,
                )

                answer = response["answer"]

                st.session_state.chat_history.append(
                    {
                        "question": user_query,
                        "answer": answer,
                        "sources": get_sources(response),
                    }
                )

            except Exception as e:

                st.error(
                    "Something went wrong while generating the answer."
                )

                st.exception(e)


# --------------------------------------------------
# Conversation History
# --------------------------------------------------

if st.session_state.chat_history:

    st.markdown("---")

    st.markdown(
        """
<div class="eyebrow">Conversation</div>
<div class="card-title">Previous questions</div>
""",
        unsafe_allow_html=True,
    )

    for index, chat in enumerate(
        reversed(st.session_state.chat_history)
    ):

        with st.container(border=True):

            st.markdown(
                f"**You:** {chat['question']}"
            )

            st.markdown(
                '<div class="answer-card">'
                f"<strong>AI:</strong><br>{chat['answer']}"
                "</div>",
                unsafe_allow_html=True,
            )

            sources = chat["sources"]

            if sources:

                st.markdown("**Sources**")

                for source in sources:

                    if source["page"] is not None:

                        st.caption(
                            f"📄 {source['source']} · "
                            f"Page {source['page']}"
                        )

                    else:

                        st.caption(
                            f"📄 {source['source']}"
                        )

            # ------------------------------------------
            # Download Answer
            # ------------------------------------------

            download_text = (
                f"Question:\n{chat['question']}\n\n"
                f"Answer:\n{chat['answer']}\n\n"
                "Sources:\n"
            )

            for source in sources:

                if source["page"] is not None:

                    download_text += (
                        f"- {source['source']} "
                        f"(Page {source['page']})\n"
                    )

                else:

                    download_text += (
                        f"- {source['source']}\n"
                    )

            st.download_button(
                label="⬇️ Download Answer",
                data=download_text,
                file_name=f"rag_answer_{index + 1}.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"download_answer_{index}",
            )