from typing import List

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


# ==================================================
# RAG Answer Generation
# ==================================================


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

        source = metadata.get(
            "source",
            "Unknown document",
        )

        page = metadata.get("page")
        page_label = metadata.get("page_label")

        if isinstance(page, int):

            page_number = page + 1

        elif page_label is not None:

            page_number = page_label

        else:

            page_number = None

        source_key = (
            source,
            page_number,
        )

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


# ==================================================
# Mind Map Data Models
# ==================================================


class MindMapBranch(BaseModel):
    """A major topic in the document mind map."""

    topic: str = Field(
        description="Name of the major topic or theme."
    )

    summary: str = Field(
        description=(
            "A short explanation of what this topic "
            "means in the document."
        )
    )

    points: List[str] = Field(
        description=(
            "Two to four important concepts, facts, "
            "or subtopics belonging to this topic."
        )
    )


class MindMap(BaseModel):
    """Structured representation of a document mind map."""

    title: str = Field(
        description=(
            "A concise title representing the main subject "
            "of the document."
        )
    )

    central_idea: str = Field(
        description=(
            "One short sentence describing the central idea "
            "or purpose of the document."
        )
    )

    branches: List[MindMapBranch] = Field(
        description=(
            "The major topics that branch out from the "
            "central idea. Prefer four to six major topics."
        )
    )


# ==================================================
# Mind Map Generation
# ==================================================


def generate_mind_map(
    document_text,
    gemini_api_key,
):
    """
    Generate structured mind-map data from document text.

    The model returns validated structured data rather than
    free-form text so the UI can later render a reliable
    visual mind map.
    """

    if not document_text.strip():

        raise ValueError(
            "Document text is empty. "
            "Please process a document first."
        )

    # Keep the first version intentionally bounded.
    # This prevents extremely large PDFs from creating
    # an unnecessarily large prompt.
    max_characters = 50000

    document_text = document_text[:max_characters]

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        max_retries=5,
        google_api_key=gemini_api_key,
    )

    structured_llm = llm.with_structured_output(
        MindMap
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert document analyst.

Create a concise one-page mind map from the
provided document.

Your job is to identify the document's main idea
and organize the most important information into
clear hierarchical topics.

Rules:

1. Stay strictly grounded in the provided document.
2. Do not invent facts or topics that are not supported.
3. Identify approximately four to six major branches.
4. Each branch should contain two to four concise points.
5. Avoid long sentences.
6. Prefer important concepts, facts, themes, processes,
   findings, or conclusions.
7. The final structure should be understandable at a glance.
8. Do not include unnecessary introductory text.
""",
            ),
            (
                "human",
                """
Analyze the following document and create its
structured mind map.

DOCUMENT:

{document}
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    result = chain.invoke(
        {
            "document": document_text,
        }
    )

    return result