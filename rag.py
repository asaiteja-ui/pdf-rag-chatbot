"""
rag.py — core RAG logic (zero LangChain, pure openai + faiss)
==============================================================
No LangChain at all — this eliminates every possible 'proxies' conflict.

Dependencies used:
  - openai        : embeddings + chat
  - faiss-cpu     : vector similarity search
  - pypdf         : already used by utils.py

Two responsibilities:
  1. build_vectorstore()  — chunk text, embed, store in FAISS
  2. get_answer()         — search FAISS, call GPT, return answer + sources
"""

import re
import faiss
import numpy as np
from openai import OpenAI


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly
based on the provided document context.

Rules:
- Use ONLY the information in the context below to answer.
- If the answer is not in the context, say "I couldn't find that in the uploaded documents."
- Be concise and direct. Cite page numbers when relevant.
- Never make up information.

Context:
{context}

Question: {question}

Answer:"""


# ── Text chunking (no LangChain needed) ───────────────────────────────────────
def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks by character count.
    Tries to break at paragraph or sentence boundaries where possible.
    """
    # Split on paragraph breaks first
    paragraphs = re.split(r"\n\s*\n", text)
    chunks     = []
    current    = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) <= chunk_size:
            current += (" " if current else "") + para
        else:
            if current:
                chunks.append(current.strip())
            # If a single paragraph is longer than chunk_size, split it further
            if len(para) > chunk_size:
                words   = para.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= chunk_size:
                        current += (" " if current else "") + word
                    else:
                        if current:
                            chunks.append(current.strip())
                        # Start next chunk with overlap from previous
                        overlap_text = " ".join(current.split()[-overlap:]) if current else ""
                        current = (overlap_text + " " if overlap_text else "") + word
            else:
                # Start next chunk with overlap from previous chunk
                overlap_words = " ".join(current.split()[-overlap:]) if current else ""
                current = (overlap_words + " " if overlap_words else "") + para

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c.strip()) > 20]  # drop tiny fragments


# ── Embedding via raw OpenAI SDK ───────────────────────────────────────────────
def _embed(texts: list[str], api_key: str) -> np.ndarray:
    """
    Embed a list of texts using OpenAI's API directly.
    Returns a float32 numpy array of shape (len(texts), 1536).
    """
    client   = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


# ── Vectorstore: a simple dataclass holding FAISS index + chunk metadata ───────
class VectorStore:
    """
    Wraps a FAISS flat index with the original text chunks and their metadata.
    No LangChain involved — just numpy arrays and faiss.
    """
    def __init__(self, index: faiss.IndexFlatIP, chunks: list[str], metadata: list[dict]):
        self.index    = index      # FAISS inner-product index (cosine sim on normalised vecs)
        self.chunks   = chunks     # original text of each chunk
        self.metadata = metadata   # {"source": filename, "page": int} per chunk

    def search(self, query_vec: np.ndarray, k: int = 3):
        """Return top-k (chunk_text, metadata) pairs for a query vector."""
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)  # normalise
        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        _, indices = self.index.search(query_vec, k)
        results = []
        for idx in indices[0]:
            if idx >= 0:
                results.append((self.chunks[idx], self.metadata[idx]))
        return results


# ── Build vectorstore ──────────────────────────────────────────────────────────
def build_vectorstore(
    docs,
    api_key: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> VectorStore:
    """
    Chunk all pages, embed them, and build a FAISS index.

    Args:
        docs:          List of objects with .page_content and .metadata
                       (LangChain Document objects from utils.py).
        api_key:       OpenAI API key.
        chunk_size:    Target characters per chunk.
        chunk_overlap: Approximate word overlap between adjacent chunks.

    Returns:
        VectorStore instance ready for similarity search.
    """
    all_chunks   = []
    all_metadata = []

    for doc in docs:
        page_chunks = _chunk_text(doc.page_content, chunk_size, chunk_overlap)
        for chunk in page_chunks:
            all_chunks.append(chunk)
            all_metadata.append(doc.metadata)

    if not all_chunks:
        raise ValueError(
            "No text could be extracted from the uploaded PDFs. "
            "Make sure they are text-based PDFs, not scanned images."
        )

    # Embed in batches of 100 to stay within API limits
    all_vectors = []
    batch_size  = 100
    for i in range(0, len(all_chunks), batch_size):
        batch   = all_chunks[i : i + batch_size]
        vectors = _embed(batch, api_key)
        all_vectors.append(vectors)

    matrix = np.vstack(all_vectors).astype(np.float32)

    # Normalise for cosine similarity via inner product
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix = matrix / norms

    # Build FAISS index — IndexFlatIP = exact inner product (cosine after normalisation)
    dim   = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    return VectorStore(index, all_chunks, all_metadata)


# ── Answer a question ──────────────────────────────────────────────────────────
def get_answer(
    question: str,
    vectorstore: VectorStore,
    api_key: str,
    top_k: int = 3,
):
    """
    Embed the question, retrieve top-k chunks, ask GPT-4o-mini, return answer.

    Args:
        question:     User's question string.
        vectorstore:  VectorStore from build_vectorstore().
        api_key:      OpenAI API key.
        top_k:        Number of chunks to pass as context.

    Returns:
        answer  (str)  — GPT-generated answer.
        sources (list) — [{file, page, snippet}, …]
    """
    # Step 1: Embed the question
    query_vec = _embed([question], api_key)[0]

    # Step 2: Retrieve top-k matching chunks
    results = vectorstore.search(query_vec, k=top_k)

    # Step 3: Format context
    context_parts = []
    for chunk_text, meta in results:
        file_name  = meta.get("source", "unknown")
        page_num   = meta.get("page", 0)
        page_label = (page_num + 1) if isinstance(page_num, int) else page_num
        context_parts.append(f"[{file_name}, page {page_label}]\n{chunk_text}")
    context_text = "\n\n".join(context_parts)

    # Step 4: Build prompt and call GPT-4o-mini via raw SDK
    filled_prompt = SYSTEM_PROMPT.format(
        context=context_text,
        question=question,
    )

    client   = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": filled_prompt}],
    )
    answer = response.choices[0].message.content

    # Step 5: Build source citations
    sources = []
    seen    = set()
    for chunk_text, meta in results:
        file_name = meta.get("source", "Unknown file")
        page_num  = meta.get("page", "?")
        key       = (file_name, page_num)
        if key not in seen:
            seen.add(key)
            snippet = chunk_text[:200].replace("\n", " ").strip()
            sources.append({
                "file":    file_name,
                "page":    (page_num + 1) if isinstance(page_num, int) else page_num,
                "snippet": snippet + "…" if len(chunk_text) > 200 else snippet,
            })

    return answer, sources
