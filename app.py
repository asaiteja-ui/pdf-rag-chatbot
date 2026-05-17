"""
PDF RAG Chatbot — Streamlit app
================================
Upload one or more PDFs, ask questions, get answers grounded in the documents.
Each answer includes the source page(s) it was drawn from.

Run:
    streamlit run app.py
"""

import streamlit as st
from rag import build_vectorstore, get_answer
from utils import extract_text_from_pdfs

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF Q&A Chatbot")
st.caption("Upload PDFs → Ask questions → Get answers with source references")

# ── Session state ──────────────────────────────────────────────────────────────
# Streamlit reruns the whole script on every interaction.
# st.session_state persists data across reruns within a session.
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None      # FAISS index, built after upload
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # list of (question, answer, sources)
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []

# ── Sidebar: upload + settings ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-...",
        help="Your key is used only for this session and never stored.",
    )

    st.divider()
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    chunk_size = st.slider(
        "Chunk size (tokens)",
        min_value=200,
        max_value=1500,
        value=500,
        step=100,
        help="Smaller chunks = more precise retrieval. Larger = more context per chunk.",
    )

    chunk_overlap = st.slider(
        "Chunk overlap (tokens)",
        min_value=0,
        max_value=300,
        value=50,
        step=25,
        help="Overlap prevents answers from being cut off at chunk boundaries.",
    )

    top_k = st.slider(
        "Chunks to retrieve (k)",
        min_value=1,
        max_value=8,
        value=3,
        help="How many text chunks are sent to the LLM as context.",
    )

    process_btn = st.button("Process PDFs", type="primary", disabled=not uploaded_files)

    if process_btn:
        if not api_key:
            st.error("Please enter your OpenAI API key first.")
        else:
            with st.spinner("Reading and indexing your PDFs…"):
                try:
                    # Extract text with page metadata from all uploaded PDFs
                    docs = extract_text_from_pdfs(uploaded_files)

                    # Build FAISS vectorstore from the extracted documents
                    st.session_state.vectorstore = build_vectorstore(
                        docs,
                        api_key=api_key,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    st.session_state.pdf_names = [f.name for f in uploaded_files]
                    st.session_state.chat_history = []   # reset chat on new upload
                    st.success(f"Indexed {len(docs)} pages from {len(uploaded_files)} PDF(s).")
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")

    # Show currently loaded documents
    if st.session_state.pdf_names:
        st.divider()
        st.subheader("Loaded documents")
        for name in st.session_state.pdf_names:
            st.markdown(f"- {name}")

    if st.session_state.chat_history:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

# ── Main panel: chat interface ─────────────────────────────────────────────────
if st.session_state.vectorstore is None:
    st.info("Upload PDFs in the sidebar and click **Process PDFs** to get started.")
else:
    # Render existing chat history
    for question, answer, sources in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            if sources:
                with st.expander("Source references"):
                    for src in sources:
                        st.markdown(
                            f"**{src['file']}** — page {src['page']}\n\n"
                            f"> {src['snippet']}"
                        )

    # Chat input
    question = st.chat_input("Ask a question about your documents…")

    if question:
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer, sources = get_answer(
                            question=question,
                            vectorstore=st.session_state.vectorstore,
                            api_key=api_key,
                            top_k=top_k,
                        )
                        st.write(answer)
                        if sources:
                            with st.expander("Source references"):
                                for src in sources:
                                    st.markdown(
                                        f"**{src['file']}** — page {src['page']}\n\n"
                                        f"> {src['snippet']}"
                                    )
                    except Exception as e:
                        answer = f"Error: {e}"
                        sources = []
                        st.error(answer)

            # Save to history
            st.session_state.chat_history.append((question, answer, sources))
