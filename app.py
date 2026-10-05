import html
import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv

from rag.generation import (
    ask_question,
    create_rag_chain,
    generate_mind_map,
    get_llm,
    get_sources,
)
from rag.ingestion import process_pdf
from rag.retrieval import (
    add_documents,
    get_retriever,
    get_vectorstore,
)


# ==================================================
# Configuration
# ==================================================

load_dotenv()

st.set_page_config(
    page_title="Cloud RAG Document Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# Helpers
# ==================================================

def get_config_value(key):
    """Read from Streamlit secrets, then .env."""

    try:
        value = st.secrets.get(key)

        if value:
            return value

    except Exception:
        pass

    return os.getenv(key)


def render_html(content):
    """Render custom HTML without Markdown code-block leakage."""

    cleaned = content.strip()

    cleaned = re.sub(
        r"\s*\n\s*",
        " ",
        cleaned,
    )

    st.markdown(
        cleaned,
        unsafe_allow_html=True,
    )


# ==================================================
# Session State
# ==================================================

if "document_namespace" not in st.session_state:
    st.session_state.document_namespace = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = 0

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "mind_map" not in st.session_state:
    st.session_state.mind_map = None


# ==================================================
# Global CSS
# ==================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(99, 102, 241, 0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 20%,
            rgba(168, 85, 247, 0.10),
            transparent 28%
        ),
        #080b12;
    color: #f8fafc;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background: #171923;
}

section[data-testid="stSidebar"] > div {
    background: #171923;
}

.hero-wrapper {
    padding: 1rem 0 2.5rem 0;
}

.hero-badge {
    display: inline-block;
    padding: 0.4rem 0.8rem;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(129, 140, 248, 0.30);
    color: #a5b4fc;
    font-size: 0.76rem;
    font-weight: 750;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hero-title {
    font-size: 3.35rem;
    line-height: 1.05;
    font-weight: 850;
    margin: 1rem 0 0.9rem 0;
    letter-spacing: -0.045em;
}

.hero-gradient-text {
    background: linear-gradient(
        90deg,
        #818cf8,
        #c084fc,
        #f0abfc
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.03rem;
    max-width: 780px;
    line-height: 1.7;
}

.hero-status {
    margin-top: 1.1rem;
    color: #94a3b8;
    font-size: 0.88rem;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    margin-right: 7px;
    box-shadow: 0 0 12px rgba(34, 197, 94, 0.55);
}

.sidebar-brand {
    font-size: 1.05rem;
    font-weight: 750;
    color: #f8fafc;
    margin-bottom: 0.7rem;
}

.sidebar-description {
    color: #cbd5e1;
    line-height: 1.6;
    font-size: 0.88rem;
}

.sidebar-section {
    color: #cbd5e1;
    line-height: 1.8;
    font-size: 0.87rem;
    margin: 1.1rem 0;
}

.sidebar-section b {
    color: #f8fafc;
}

.section-eyebrow {
    color: #818cf8;
    font-size: 0.72rem;
    font-weight: 750;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.section-title {
    color: #f8fafc;
    font-size: 1.55rem;
    font-weight: 750;
    margin-bottom: 0.3rem;
}

.section-description {
    color: #94a3b8;
    margin-bottom: 1rem;
    line-height: 1.65;
}

.document-card {
    border: 1px solid rgba(129, 140, 248, 0.20);
    background: rgba(30, 41, 59, 0.45);
    border-radius: 16px;
    padding: 1rem 1.15rem;
    margin: 1rem 0;
}

.document-name {
    font-weight: 700;
    color: #f8fafc;
}

.document-meta {
    color: #94a3b8;
    font-size: 0.82rem;
    margin-top: 0.3rem;
}

.pipeline-card {
    border: 1px solid rgba(148, 163, 184, 0.12);
    background: rgba(15, 23, 42, 0.55);
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
    min-height: 108px;
}

.pipeline-icon {
    font-size: 1.5rem;
    margin-bottom: 0.4rem;
}

.pipeline-name {
    font-size: 0.84rem;
    color: #cbd5e1;
    font-weight: 650;
}

.answer-card {
    border: 1px solid rgba(129, 140, 248, 0.15);
    background: rgba(15, 23, 42, 0.60);
    border-radius: 18px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.answer-label {
    color: #a5b4fc;
    font-size: 0.73rem;
    font-weight: 750;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.source-header {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 650;
    margin-top: 1rem;
    margin-bottom: 0.4rem;
}


/* ==================================================
   Connected Mind Map
   ================================================== */

.mindmap-shell {
    position: relative;
    margin-top: 1.2rem;
    padding: 1.8rem;
    border-radius: 26px;

    background:
        radial-gradient(
            circle at 50% 50%,
            rgba(99, 102, 241, 0.12),
            transparent 42%
        ),
        radial-gradient(
            circle at 15% 20%,
            rgba(168, 85, 247, 0.08),
            transparent 30%
        ),
        rgba(8, 11, 18, 0.85);

    border: 1px solid rgba(129, 140, 248, 0.20);

    overflow: hidden;
}

.mindmap-title {
    text-align: center;
    color: #f8fafc;
    font-size: 1.55rem;
    font-weight: 800;
    margin-bottom: 1.8rem;
}

.mindmap-layout {
    display: grid;

    grid-template-columns:
        minmax(230px, 1fr)
        minmax(300px, 1.15fr)
        minmax(230px, 1fr);

    grid-template-rows:
        minmax(150px, auto)
        minmax(210px, auto)
        minmax(150px, auto);

    column-gap: 2rem;
    row-gap: 1.4rem;

    align-items: center;

    max-width: 1100px;
    margin: 0 auto;
}

.mindmap-center {
    grid-column: 2;
    grid-row: 2;

    position: relative;

    min-height: 210px;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;

    text-align: center;

    padding: 1.5rem;

    border-radius: 28px;

    background:
        linear-gradient(
            135deg,
            rgba(79, 70, 229, 0.48),
            rgba(147, 51, 234, 0.40)
        );

    border: 1px solid rgba(196, 181, 253, 0.45);

    box-shadow:
        0 0 50px rgba(99, 102, 241, 0.18),
        inset 0 0 30px rgba(255, 255, 255, 0.025);

    z-index: 3;
}

.mindmap-center::before {
    content: "";
    position: absolute;

    inset: -10px;

    border-radius: 34px;

    border: 1px solid rgba(129, 140, 248, 0.10);

    pointer-events: none;
}

.mindmap-center-label {
    color: #c4b5fd;
    font-size: 0.70rem;
    font-weight: 800;

    letter-spacing: 0.14em;
    text-transform: uppercase;

    margin-bottom: 0.65rem;
}

.mindmap-center-title {
    color: #ffffff;

    font-size: 1.25rem;
    font-weight: 850;

    line-height: 1.3;

    margin-bottom: 0.65rem;
}

.mindmap-center-description {
    color: #ddd6fe;

    font-size: 0.84rem;

    line-height: 1.5;

    max-width: 340px;
}


/* Branch cards */

.mindmap-branch {
    position: relative;

    min-height: 145px;

    padding: 1rem 1.1rem;

    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(30, 41, 59, 0.82),
            rgba(15, 23, 42, 0.78)
        );

    border: 1px solid rgba(148, 163, 184, 0.16);

    box-shadow:
        0 10px 30px rgba(0, 0, 0, 0.18);

    z-index: 2;
}

.mindmap-branch:hover {
    border-color: rgba(129, 140, 248, 0.40);
}

.mindmap-branch.left {
    margin-right: 0.7rem;
}

.mindmap-branch.right {
    margin-left: 0.7rem;
}


/* Connecting lines */

.mindmap-branch.left::after {
    content: "";

    position: absolute;

    right: -2rem;
    top: 50%;

    width: 2rem;
    height: 2px;

    background:
        linear-gradient(
            90deg,
            rgba(129, 140, 248, 0.25),
            rgba(129, 140, 248, 0.85)
        );
}

.mindmap-branch.right::before {
    content: "";

    position: absolute;

    left: -2rem;
    top: 50%;

    width: 2rem;
    height: 2px;

    background:
        linear-gradient(
            90deg,
            rgba(129, 140, 248, 0.85),
            rgba(129, 140, 248, 0.25)
        );
}


/* Center connector glow */

.mindmap-center::after {
    content: "";

    position: absolute;

    left: -80px;
    right: -80px;

    top: 50%;

    height: 2px;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(129, 140, 248, 0.16),
            transparent
        );

    z-index: -1;
}


.mindmap-branch-title {
    color: #f8fafc;

    font-size: 0.98rem;
    font-weight: 800;

    margin-bottom: 0.35rem;
}

.mindmap-branch-summary {
    color: #94a3b8;

    font-size: 0.78rem;

    line-height: 1.45;

    margin-bottom: 0.55rem;
}

.mindmap-point {
    color: #cbd5e1;

    font-size: 0.76rem;

    line-height: 1.4;

    padding: 0.32rem 0;

    border-top:
        1px solid rgba(148, 163, 184, 0.08);
}

.mindmap-point::before {
    content: "◆";

    color: #818cf8;

    margin-right: 0.4rem;

    font-size: 0.45rem;

    vertical-align: middle;
}


/* Accent variations */

.mindmap-accent-1 {
    border-top: 2px solid #818cf8;
}

.mindmap-accent-2 {
    border-top: 2px solid #c084fc;
}

.mindmap-accent-3 {
    border-top: 2px solid #22d3ee;
}

.mindmap-accent-4 {
    border-top: 2px solid #34d399;
}

.mindmap-accent-5 {
    border-top: 2px solid #fbbf24;
}

.mindmap-accent-6 {
    border-top: 2px solid #fb7185;
}


/* Mobile */

@media (max-width: 850px) {

    .mindmap-layout {
        display: flex;
        flex-direction: column;
    }

    .mindmap-center {
        width: 100%;
        order: 0;
    }

    .mindmap-branch {
        width: 100%;
        margin: 0 !important;
    }

    .mindmap-branch.left::after,
    .mindmap-branch.right::before,
    .mindmap-center::after {
        display: none;
    }

}

@media (max-width: 760px) {

    .hero-title {
        font-size: 2.35rem;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ==================================================
# Sidebar
# ==================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">
            ✦ Cloud RAG Assistant
        </div>
        """
    )

    render_html(
        """
        <div class="sidebar-description">
            Upload a document, ask questions,
            and generate a visual summary.
        </div>
        """
    )

    st.markdown("---")

    render_html(
        """
        <div class="sidebar-section">
            <b>Pipeline</b><br>
            PDF → Chunks → Embeddings → Pinecone → Gemini
        </div>
        """
    )

    render_html(
        """
        <div class="sidebar-section">
            <b>Features</b><br>
            • Semantic document search<br>
            • Grounded answers<br>
            • Source citations<br>
            • Visual mind maps
        </div>
        """
    )


# ==================================================
# Hero
# ==================================================

render_html(
    """
    <div class="hero-wrapper">

        <div class="hero-badge">
            AI-Powered Document Intelligence
        </div>

        <div class="hero-title">
            Turn your documents into
            <span class="hero-gradient-text">
                answers.
            </span>
        </div>

        <div class="hero-subtitle">
            Upload a PDF, retrieve the most relevant information
            with semantic search, and get grounded answers from
            Gemini. You can also generate a visual mind map
            of the document.
        </div>

        <div class="hero-status">
            <span class="status-dot"></span>
            Gemini + Pinecone RAG pipeline ready
        </div>

    </div>
    """
)


# ==================================================
# API Keys
# ==================================================

gemini_api_key = get_config_value(
    "GEMINI_API_KEY"
)

pinecone_api_key = get_config_value(
    "PINECONE_API_KEY"
)


# ==================================================
# Upload
# ==================================================

render_html(
    """
    <div class="section-eyebrow">
        01 — DOCUMENT
    </div>

    <div class="section-title">
        Upload your document
    </div>

    <div class="section-description">
        Start by uploading a PDF. The document will be
        chunked and indexed for semantic retrieval.
    </div>
    """
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    label_visibility="collapsed",
)


if uploaded_file is not None:

    render_html(
        f"""
        <div class="document-card">

            <div class="document-name">
                📄 {html.escape(uploaded_file.name)}
            </div>

            <div class="document-meta">
                Ready to process •
                {uploaded_file.size / 1024:.1f} KB
            </div>

        </div>
        """
    )

    if st.button(
        "Process Document",
        type="primary",
        use_container_width=True,
    ):

        if not gemini_api_key or not pinecone_api_key:

            st.error(
                "API keys are not configured."
            )

        else:

            with st.spinner(
                "Extracting, chunking and indexing your document..."
            ):

                try:

                    chunks = process_pdf(
                        uploaded_file
                    )

                    document_text = "\n\n".join(
                        chunk.page_content
                        for chunk in chunks
                    )

                    namespace = (
                        f"doc-{uuid.uuid4().hex}"
                    )

                    vectorstore = get_vectorstore(
                        gemini_api_key,
                        pinecone_api_key,
                        namespace,
                    )

                    add_documents(
                        vectorstore,
                        chunks,
                    )

                    st.session_state.document_namespace = namespace
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.document_processed = True
                    st.session_state.document_chunks = len(chunks)
                    st.session_state.document_text = document_text
                    st.session_state.chat_history = []
                    st.session_state.mind_map = None

                    st.success(
                        "Document processed successfully."
                    )

                except Exception as exc:

                    st.error(
                        f"Document processing failed: {exc}"
                    )


# ==================================================
# Indexed Document
# ==================================================

if st.session_state.document_processed:

    render_html(
        """
        <div class="section-eyebrow">
            02 — INDEXED DOCUMENT
        </div>
        """
    )

    render_html(
        f"""
        <div class="document-card">

            <div class="document-name">
                ✓ {html.escape(
                    st.session_state.document_name
                )}
            </div>

            <div class="document-meta">
                {st.session_state.document_chunks}
                chunks indexed in Pinecone
            </div>

        </div>
        """
    )


# ==================================================
# Pipeline
# ==================================================

if st.session_state.document_processed:

    render_html(
        """
        <div class="section-eyebrow">
            03 — PIPELINE
        </div>

        <div class="section-title">
            How your document is processed
        </div>

        <div class="section-description">
            Your question flows through the same retrieval
            pipeline before Gemini generates the answer.
        </div>
        """
    )

    cols = st.columns(5)

    pipeline_steps = [
        ("📄", "Upload"),
        ("✂️", "Chunk"),
        ("🧠", "Embed"),
        ("🔎", "Retrieve"),
        ("✨", "Generate"),
    ]

    for column, (icon, name) in zip(
        cols,
        pipeline_steps,
    ):

        with column:

            render_html(
                f"""
                <div class="pipeline-card">

                    <div class="pipeline-icon">
                        {icon}
                    </div>

                    <div class="pipeline-name">
                        {name}
                    </div>

                </div>
                """
            )


# ==================================================
# Mind Map Generation
# ==================================================

if st.session_state.document_processed:

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    render_html(
        """
        <div class="section-eyebrow">
            04 — VISUAL SUMMARY
        </div>

        <div class="section-title">
            Generate a document mind map
        </div>

        <div class="section-description">
            Gemini will identify the central idea and
            the most important topics in your document
            and organize them into a visual summary.
        </div>
        """
    )

    if st.button(
        "🧠 Generate Mind Map",
        type="secondary",
        use_container_width=True,
    ):

        if not gemini_api_key:

            st.error(
                "Gemini API key is not configured."
            )

        elif not st.session_state.document_text.strip():

            st.error(
                "No document text is available."
            )

        else:

            with st.spinner(
                "Analyzing document and creating mind map..."
            ):

                try:

                    mind_map = generate_mind_map(
                        st.session_state.document_text,
                        gemini_api_key,
                    )

                    st.session_state.mind_map = mind_map

                except Exception as exc:

                    st.error(
                        f"Mind map generation failed: {exc}"
                    )


# ==================================================
# Connected Mind Map Rendering + SVG Export
# ==================================================

if st.session_state.mind_map is not None:

    mind_map = st.session_state.mind_map

    if hasattr(mind_map, "model_dump"):
        mind_map_data = mind_map.model_dump()
    else:
        mind_map_data = mind_map

    title = str(
        mind_map_data.get(
            "title",
            "Document Mind Map",
        )
    )

    central_idea = str(
        mind_map_data.get(
            "central_idea",
            "",
        )
    )

    branches = mind_map_data.get(
        "branches",
        [],
    )

    branches = branches[:6]

    # --------------------------------------------------
    # HTML Mind Map
    # --------------------------------------------------

    positions = [
        ("left", "1", "1"),
        ("right", "2", "1"),
        ("left", "3", "2"),
        ("right", "4", "2"),
        ("left", "5", "3"),
        ("right", "6", "3"),
    ]

    branch_html = ""

    for index, branch in enumerate(branches):

        side, accent, row = positions[index]

        topic = html.escape(
            str(
                branch.get(
                    "topic",
                    "Topic",
                )
            )
        )

        summary = html.escape(
            str(
                branch.get(
                    "summary",
                    "",
                )
            )
        )

        points = branch.get(
            "points",
            [],
        )

        points_html = ""

        for point in points:

            points_html += f"""
                <div class="mindmap-point">
                    {html.escape(str(point))}
                </div>
            """

        column = 1 if side == "left" else 3

        branch_html += f"""
            <div
                class="mindmap-branch {side} mindmap-accent-{accent}"
                style="
                    grid-column: {column};
                    grid-row: {row};
                "
            >

                <div class="mindmap-branch-title">
                    {topic}
                </div>

                <div class="mindmap-branch-summary">
                    {summary}
                </div>

                {points_html}

            </div>
        """

    mindmap_html = f"""
        <div class="mindmap-shell">

            <div class="mindmap-title">
                {html.escape(title)}
            </div>

            <div class="mindmap-layout">

                {branch_html}

                <div class="mindmap-center">

                    <div class="mindmap-center-label">
                        Central Idea
                    </div>

                    <div class="mindmap-center-title">
                        {html.escape(title)}
                    </div>

                    <div class="mindmap-center-description">
                        {html.escape(central_idea)}
                    </div>

                </div>

            </div>

        </div>
    """

    render_html(
        mindmap_html
    )


        # --------------------------------------------------
    # Professional SVG Generator
    # --------------------------------------------------

    def build_mindmap_svg(
        title,
        central_idea,
        branches,
    ):
        """
        Build a standalone, properly wrapped SVG mind map.

        The SVG is independent of Streamlit and can be opened
        directly in a browser, Figma, Illustrator, Inkscape,
        or other vector applications.
        """

        def escape_text(value):
            return html.escape(
                str(value),
                quote=True,
            )

        def wrap_text(
            text,
            max_chars,
        ):
            """
            Simple word-based wrapping for SVG text.
            """

            words = str(text).split()

            lines = []
            current = ""

            for word in words:

                candidate = (
                    f"{current} {word}"
                    if current
                    else word
                )

                if len(candidate) <= max_chars:

                    current = candidate

                else:

                    if current:
                        lines.append(current)

                    current = word

            if current:
                lines.append(current)

            return lines

        def svg_text_lines(
            lines,
            x,
            y,
            font_size,
            color,
            weight="400",
            line_height=20,
            anchor="start",
        ):
            """
            Convert wrapped lines into SVG tspans.
            """

            output = f"""
<text
    x="{x}"
    y="{y}"
    text-anchor="{anchor}"
    fill="{color}"
    font-family="Arial, Helvetica, sans-serif"
    font-size="{font_size}px"
    font-weight="{weight}"
>
"""

            for index, line in enumerate(lines):

                line_y = (
                    y + index * line_height
                )

                output += f"""
    <tspan
        x="{x}"
        y="{line_y}"
    >
        {escape_text(line)}
    </tspan>
"""

            output += """
</text>
"""

            return output

        # --------------------------------------------------
        # Canvas
        # --------------------------------------------------

        width = 1600
        height = 1000

        center_x = 800
        center_y = 500

        center_width = 430
        center_height = 210

        branch_width = 390
        branch_height = 190

        left_x = 80
        right_x = 1130

        branch_y_positions = [
            180,
            500,
            820,
        ]

        colors = [
            "#818cf8",
            "#c084fc",
            "#22d3ee",
            "#34d399",
            "#fbbf24",
            "#fb7185",
        ]

        svg = []

        # --------------------------------------------------
        # SVG Header
        # --------------------------------------------------

        svg.append(
            f"""
<svg
    xmlns="http://www.w3.org/2000/svg"
    width="{width}"
    height="{height}"
    viewBox="0 0 {width} {height}"
>
"""
        )

        # --------------------------------------------------
        # Definitions
        # --------------------------------------------------

        svg.append(
            """
<defs>

    <linearGradient
        id="background"
        x1="0%"
        y1="0%"
        x2="100%"
        y2="100%"
    >
        <stop
            offset="0%"
            stop-color="#080b12"
        />

        <stop
            offset="100%"
            stop-color="#11152a"
        />
    </linearGradient>


    <linearGradient
        id="centerGradient"
        x1="0%"
        y1="0%"
        x2="100%"
        y2="100%"
    >
        <stop
            offset="0%"
            stop-color="#4f46e5"
        />

        <stop
            offset="100%"
            stop-color="#9333ea"
        />
    </linearGradient>


    <filter
        id="shadow"
        x="-30%"
        y="-30%"
        width="160%"
        height="160%"
    >

        <feDropShadow
            dx="0"
            dy="12"
            stdDeviation="16"
            flood-color="#000000"
            flood-opacity="0.35"
        />

    </filter>

</defs>
"""
        )

        # --------------------------------------------------
        # Background
        # --------------------------------------------------

        svg.append(
            f"""
<rect
    x="0"
    y="0"
    width="{width}"
    height="{height}"
    rx="32"
    fill="url(#background)"
/>
"""
        )

        # --------------------------------------------------
        # Main Title
        # --------------------------------------------------

        title_lines = wrap_text(
            title,
            52,
        )

        svg.append(
            svg_text_lines(
                title_lines[:2],
                center_x,
                55,
                28,
                "#f8fafc",
                "700",
                34,
                "middle",
            )
        )

        # --------------------------------------------------
        # Branch Data
        # --------------------------------------------------

        branch_positions = []

        for index, branch in enumerate(branches[:6]):

            row = index // 2

            side = (
                "left"
                if index % 2 == 0
                else "right"
            )

            x = (
                left_x
                if side == "left"
                else right_x
            )

            y = branch_y_positions[row]

            branch_positions.append(
                (
                    x,
                    y,
                    side,
                )
            )

        # --------------------------------------------------
        # Connectors
        # --------------------------------------------------

        center_left = (
            center_x
            - center_width / 2
        )

        center_right = (
            center_x
            + center_width / 2
        )

        for index, (
            x,
            y,
            side,
        ) in enumerate(branch_positions):

            color = colors[
                index % len(colors)
            ]

            branch_center_y = (
                y + branch_height / 2
            )

            if side == "left":

                start_x = (
                    x + branch_width
                )

                end_x = center_left

            else:

                start_x = x

                end_x = center_right

            svg.append(
                f"""
<path
    d="
        M {start_x} {branch_center_y}
        C
        {(start_x + end_x) / 2}
        {branch_center_y},
        {(start_x + end_x) / 2}
        {center_y},
        {end_x}
        {center_y}
    "
    fill="none"
    stroke="{color}"
    stroke-width="3"
    stroke-linecap="round"
    opacity="0.55"
/>
"""
            )

        # --------------------------------------------------
        # Center Node
        # --------------------------------------------------

        center_top = (
            center_y
            - center_height / 2
        )

        svg.append(
            f"""
<rect
    x="{center_left}"
    y="{center_top}"
    width="{center_width}"
    height="{center_height}"
    rx="30"
    fill="url(#centerGradient)"
    stroke="#c4b5fd"
    stroke-width="2"
    filter="url(#shadow)"
/>
"""
        )

        svg.append(
            f"""
<text
    x="{center_x}"
    y="{center_top + 45}"
    text-anchor="middle"
    fill="#ddd6fe"
    font-family="Arial, Helvetica, sans-serif"
    font-size="12px"
    font-weight="700"
    letter-spacing="2"
>
    CENTRAL IDEA
</text>
"""
        )

        center_title_lines = wrap_text(
            title,
            31,
        )

        svg.append(
            svg_text_lines(
                center_title_lines[:2],
                center_x,
                center_top + 82,
                19,
                "#ffffff",
                "700",
                25,
                "middle",
            )
        )

        idea_lines = wrap_text(
            central_idea,
            48,
        )

        svg.append(
            svg_text_lines(
                idea_lines[:3],
                center_x,
                center_top + 135,
                12,
                "#ddd6fe",
                "400",
                18,
                "middle",
            )
        )

        # --------------------------------------------------
        # Branch Cards
        # --------------------------------------------------

        for index, branch in enumerate(
            branches[:6]
        ):

            x, y, side = branch_positions[index]

            color = colors[
                index % len(colors)
            ]

            # Card

            svg.append(
                f"""
<rect
    x="{x}"
    y="{y}"
    width="{branch_width}"
    height="{branch_height}"
    rx="22"
    fill="#111827"
    stroke="{color}"
    stroke-width="2"
    filter="url(#shadow)"
/>
"""
            )

            # Accent bar

            svg.append(
                f"""
<rect
    x="{x}"
    y="{y}"
    width="7"
    height="{branch_height}"
    rx="3"
    fill="{color}"
/>
"""
            )

            # Topic

            topic = branch.get(
                "topic",
                "Topic",
            )

            topic_lines = wrap_text(
                topic,
                34,
            )

            svg.append(
                svg_text_lines(
                    topic_lines[:2],
                    x + 28,
                    y + 35,
                    17,
                    "#f8fafc",
                    "700",
                    21,
                )
            )

            # Summary

            summary = branch.get(
                "summary",
                "",
            )

            summary_lines = wrap_text(
                summary,
                54,
            )

            svg.append(
                svg_text_lines(
                    summary_lines[:2],
                    x + 28,
                    y + 75,
                    11,
                    "#94a3b8",
                    "400",
                    16,
                )
            )

            # Points

            points = branch.get(
                "points",
                [],
            )

            point_y = (
                y + 112
            )

            for point in points[:3]:

                point_lines = wrap_text(
                    point,
                    48,
                )

                if not point_lines:
                    continue

                # Bullet

                svg.append(
                    f"""
<circle
    cx="{x + 30}"
    cy="{point_y - 4}"
    r="3"
    fill="{color}"
/>
"""
                )

                svg.append(
                    svg_text_lines(
                        point_lines[:1],
                        x + 42,
                        point_y,
                        10,
                        "#cbd5e1",
                        "400",
                        14,
                    )
                )

                point_y += 22

        # --------------------------------------------------
        # Footer
        # --------------------------------------------------

        svg.append(
            f"""
<text
    x="{center_x}"
    y="{height - 28}"
    text-anchor="middle"
    fill="#64748b"
    font-family="Arial, Helvetica, sans-serif"
    font-size="10px"
>
    Generated from document content • Cloud RAG Document Assistant
</text>
"""
        )

        # --------------------------------------------------
        # Close SVG
        # --------------------------------------------------

        svg.append(
            """
</svg>
"""
        )

        return "".join(svg)


    # --------------------------------------------------
    # Generate SVG
    # --------------------------------------------------

    svg_data = build_mindmap_svg(
        title=title,
        central_idea=central_idea,
        branches=branches,
    )


    # --------------------------------------------------
    # Download Buttons
    # --------------------------------------------------

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    download_cols = st.columns(2)

    with download_cols[0]:

        st.download_button(
            label="⬇️ Download Mind Map SVG",
            data=svg_data,
            file_name="document_mind_map.svg",
            mime="image/svg+xml",
            use_container_width=True,
        )

    with download_cols[1]:

        st.download_button(
            label="⬇️ Download Mind Map Data",
            data=str(mind_map_data),
            file_name="document_mind_map.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ==================================================
# Question Answering
# ==================================================

if st.session_state.document_processed:

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    render_html(
        """
        <div class="section-eyebrow">
            05 — ASK YOUR DOCUMENT
        </div>

        <div class="section-title">
            Ask anything about your document
        </div>

        <div class="section-description">
            Questions are answered using the most relevant
            document chunks retrieved from Pinecone.
        </div>
        """
    )

    question = st.text_input(
        "Question",
        placeholder="Ask a question about your document...",
        label_visibility="collapsed",
    )

    st.markdown(
        "Try an example:"
    )

    example_cols = st.columns(3)

    examples = [
        "What is the main idea?",
        "What are the key findings?",
        "Summarize the important points.",
    ]

    for column, example in zip(
        example_cols,
        examples,
    ):

        with column:

            if st.button(
                example,
                use_container_width=True,
            ):

                question = example

    if st.button(
        "✨ Ask AI",
        type="primary",
        use_container_width=True,
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        elif not gemini_api_key or not pinecone_api_key:

            st.error(
                "API keys are not configured."
            )

        else:

            with st.spinner(
                "Searching your document..."
            ):

                try:

                    vectorstore = get_vectorstore(
                        gemini_api_key,
                        pinecone_api_key,
                        st.session_state.document_namespace,
                    )

                    retriever = get_retriever(
                        vectorstore,
                        k=3,
                    )

                    llm = get_llm(
                        gemini_api_key
                    )

                    rag_chain = create_rag_chain(
                        llm,
                        retriever,
                    )

                    response = ask_question(
                        rag_chain,
                        question,
                    )

                    answer = response.get(
                        "answer",
                        "I couldn't find an answer.",
                    )

                    sources = get_sources(
                        response
                    )

                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "sources": sources,
                        }
                    )

                except Exception as exc:

                    st.error(
                        f"Question answering failed: {exc}"
                    )


# ==================================================
# Conversation History
# ==================================================

if st.session_state.chat_history:

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    render_html(
        """
        <div class="section-eyebrow">
            06 — CONVERSATION
        </div>

        <div class="section-title">
            Previous answers
        </div>
        """
    )

    for index, item in enumerate(
        reversed(
            st.session_state.chat_history
        )
    ):

        render_html(
            f"""
            <div class="answer-card">

                <div class="answer-label">
                    Question
                </div>

                <div>
                    {html.escape(
                        item["question"]
                    )}
                </div>

                <br>

                <div class="answer-label">
                    Answer
                </div>

                <div>
                    {html.escape(
                        item["answer"]
                    )}
                </div>

            </div>
            """
        )

        sources = item.get(
            "sources",
            [],
        )

        if sources:

            render_html(
                """
                <div class="source-header">
                    Sources
                </div>
                """
            )

            for source in sources:

                source_name = html.escape(
                    str(
                        source.get(
                            "source",
                            "Unknown document",
                        )
                    )
                )

                page = source.get(
                    "page"
                )

                if page is not None:

                    st.caption(
                        f"📄 {source_name} — Page {page}"
                    )

                else:

                    st.caption(
                        f"📄 {source_name}"
                    )

        st.download_button(
            label="⬇️ Download Answer",
            data=item["answer"],
            file_name=(
                f"answer_"
                f"{len(st.session_state.chat_history) - index}.txt"
            ),
            mime="text/plain",
            key=f"download_answer_{index}",
        )


# ==================================================
# Clear Document
# ==================================================

if st.session_state.document_processed:

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    if st.button(
        "Clear Document",
        use_container_width=True,
    ):

        st.session_state.document_namespace = None
        st.session_state.document_name = None
        st.session_state.document_processed = False
        st.session_state.document_chunks = 0
        st.session_state.document_text = ""
        st.session_state.chat_history = []
        st.session_state.mind_map = None

        st.rerun()