from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(gemini_api_key):
    """Create the Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        max_retries=5,
        google_api_key=gemini_api_key,
    )


def create_rag_chain(llm, retriever):
    """Create the retrieval + answer-generation chain."""

    system_prompt = (
        "You are an intelligent assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, say that you don't know. "
        "Keep the answer concise and strictly based on the document.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt,
    )

    return create_retrieval_chain(
        retriever,
        question_answer_chain,
    )


def ask_question(rag_chain, question):
    """Run the RAG chain for a user question."""
    return rag_chain.invoke(
        {
            "input": question,
        }
    )


def get_sources(response):
    """Extract unique source references from retrieved documents."""

    sources = []
    seen = set()

    for document in response.get("context", []):
        metadata = document.metadata

        source = metadata.get("source", "Unknown document")
        page = metadata.get("page")
        page_label = metadata.get("page_label")

        if isinstance(page, int):
            page_number = page + 1
        elif page_label is not None:
            page_number = page_label
        else:
            page_number = None

        source_key = (source, page_number)

        if source_key in seen:
            continue

        seen.add(source_key)

        sources.append(
            {
                "source": source,
                "page": page_number,
            }
        )

    return sources