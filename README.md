# PDF Q&A Chatbot — RAG with LangChain + FAISS + Streamlit

Ask questions about any PDF and get accurate, cited answers powered by GPT-4o-mini.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)

---

## What it does

- Upload one or more PDFs via a drag-and-drop interface
- Ask any question in plain English
- Get a concise answer grounded in the document content
- Every answer includes the exact page(s) it was drawn from
- Adjustable chunk size, overlap, and retrieval count

## How it works (RAG pipeline)

```
PDF upload
    ↓
Text extraction (PyPDF, page by page)
    ↓
Chunking (RecursiveCharacterTextSplitter)
    ↓
Embedding (OpenAI text-embedding-3-small)
    ↓
FAISS vector index
    ↓
User question → embed → similarity search → top-k chunks
    ↓
Prompt: system instructions + chunks + question → GPT-4o-mini
    ↓
Answer + source citations
```

## Screenshot

> *(Add a screenshot or Loom video link here after recording your demo)*

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/pdf-rag-chatbot
cd pdf-rag-chatbot
pip install -r requirements.txt
```

### 2. Run

```bash
streamlit run app.py
```

### 3. Use

1. Enter your OpenAI API key in the sidebar
2. Upload one or more PDFs
3. Click **Process PDFs**
4. Start asking questions

## Project structure

```
pdf-rag-chatbot/
├── app.py          # Streamlit UI — chat interface, sidebar controls
├── rag.py          # Core RAG logic — vectorstore building + QA chain
├── utils.py        # PDF text extraction with page metadata
├── requirements.txt
└── README.md
```

## Customisation ideas

| Change | How |
|--------|-----|
| Use a different LLM | Swap `ChatOpenAI` for `ChatAnthropic` or a local Ollama model |
| Persist the index | Call `vectorstore.save_local("index")` and `FAISS.load_local(...)` |
| Support scanned PDFs | Add `pytesseract` OCR before the chunking step |
| Add memory | Use `ConversationalRetrievalChain` instead of `RetrievalQA` |
| Deploy | Push to Streamlit Cloud (free) — add your API key as a secret |

## Tech stack

- [LangChain](https://python.langchain.com/) — RAG orchestration
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [OpenAI](https://platform.openai.com/) — embeddings + LLM
- [Streamlit](https://streamlit.io/) — web UI

## License

MIT
