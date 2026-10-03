import os
import json
import hashlib
from pathlib import Path

import streamlit as st
from rag.ingest import ingest_documents
from rag.retriever import Retriever
from rag.generator import generate_answer
from rag.evaluator import evaluate_queries

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="🤖", layout="wide")

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

st.title("🤖 RAG Knowledge Assistant")
st.caption("Upload documents → extract → clean → chunk → embed → retrieve → generate grounded answers")

st.sidebar.header("🔑 LLM Configuration")
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
if "openai_model" not in st.session_state:
    st.session_state.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.session_state.openai_api_key = st.sidebar.text_input(
    "OpenAI API key",
    type="password",
    value=st.session_state.openai_api_key,
    help="Add your OpenAI API key to enable grounded answers from the LLM.",
)
st.session_state.openai_model = st.sidebar.text_input(
    "OpenAI model",
    value=st.session_state.openai_model,
    help="Example: gpt-4o-mini, gpt-4.1-mini, gpt-5-mini",
)

if st.session_state.openai_api_key:
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
if st.session_state.openai_model:
    os.environ["OPENAI_MODEL"] = st.session_state.openai_model

if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever(INDEX_DIR)

tabs = st.tabs(["📚 Documents", "💬 Ask", "🧪 Evaluation", "📊 System"])

with tabs[0]:
    st.subheader("Document ingestion")
    uploads = st.file_uploader(
        "Upload PDFs, DOCX, TXT, CSV, Excel, or JSON files",
        type=[
            "pdf",
            "docx",
            "txt",
            "csv",
            "tsv",
            "xls",
            "xlsx",
            "json",
            "jsonl",
            "ndjson",
        ],
        accept_multiple_files=True,
    )

    if uploads and st.button("Process documents", type="primary"):
        saved = []
        for file in uploads:
            path = UPLOAD_DIR / file.name
            path.write_bytes(file.getvalue())
            saved.append(path)

        with st.spinner("Extracting, cleaning, chunking and indexing..."):
            stats = ingest_documents(saved, INDEX_DIR)

        st.success(
            f"Processed {stats['documents']} documents, "
            f"{stats['pages']} pages/sections and {stats['chunks']} chunks."
        )
        st.session_state.retriever = Retriever(INDEX_DIR)

    metadata_file = INDEX_DIR / "metadata.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        st.write(f"**Indexed chunks:** {len(metadata):,}")
        st.dataframe(metadata[:100], use_container_width=True)

with tabs[1]:
    st.subheader("Ask your documents")
    question = st.text_area(
        "Question",
        placeholder="What are the company's main revenue drivers?",
        height=100,
    )
    k = st.slider("Retrieved chunks", 1, 8, 5)

    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Searching documents and generating answer..."):
            results = st.session_state.retriever.search(question, k=k)
            answer = generate_answer(
                question,
                results,
                api_key=st.session_state.get("openai_api_key"),
                model=st.session_state.get("openai_model"),
            )

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Retrieved sources")
        if not results:
            st.warning("No relevant document chunks were found.")
        for i, r in enumerate(results, 1):
            with st.expander(
                f"{i}. {r['source']} — page/section {r['page']} — score {r['score']:.3f}"
            ):
                st.write(r["text"])

with tabs[2]:
    st.subheader("RAG evaluation")
    st.write(
        "Create `data/evaluation/questions.json` containing questions, expected "
        "keywords, and optionally expected sources."
    )

    if st.button("Run evaluation"):
        with st.spinner("Running evaluation queries..."):
            report = evaluate_queries(st.session_state.retriever)
        st.json(report)

with tabs[3]:
    st.subheader("System information")
    metadata_file = INDEX_DIR / "metadata.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        docs = sorted(set(x["source"] for x in metadata))
        st.metric("Documents", len(docs))
        st.metric("Indexed chunks", len(metadata))
        st.write("### Documents")
        for d in docs:
            st.write("•", d)
    else:
        st.info("No documents indexed yet.")

st.divider()
st.caption(
    "Prototype architecture: Streamlit + Sentence Transformers + FAISS + optional OpenAI API. "
    "For production, add authentication, persistent storage, background jobs, monitoring and access controls."
)
