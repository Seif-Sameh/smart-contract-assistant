# 📜 Smart Contract Assistant

> **RAG-powered contract Q&A with summarization, guardrails & evaluation**  
> Groq `openai/gpt-oss-120b` · FAISS · LangChain · Gradio · FastAPI/LangServe

---

## 🏗️ Architecture

```
smart_contract_assistant/
├── backend/
│   ├── api/
│   │   └── server.py          # FastAPI + LangServe microservice
│   ├── pipelines/
│   │   ├── ingestion.py       # Extract → Chunk → Embed → FAISS
│   │   ├── retrieval.py       # Semantic search + LLM Q&A (ConversationalRAG)
│   │   ├── summarization.py   # Map-reduce document summarization
│   │   ├── guardrails.py      # Input/output safety & factuality checks
│   │   └── evaluation.py      # Faithfulness, relevance & citation metrics
│   └── utils/
│       └── config.py          # Centralized config from .env
├── frontend/
│   └── app.py                 # Gradio UI (Upload · Chat · Summary · Eval)
├── data/
│   └── vectorstore/           # FAISS index (auto-created on upload)
├── tests/
│   └── test_pipelines.py      # Unit tests (guardrails, chunking)
├── SmartContractAssistant_Colab.ipynb  # ← START HERE for Colab
├── requirements.txt
└── .env.example
```

### System Flow

```
User Upload (PDF/DOCX)
        │
        ▼
  Ingestion Pipeline
  Extract → Chunk (1000/150) → Embed (MiniLM-L6-v2) → FAISS
        │
        ▼
  User Question
        │
        ▼
  Input Guardrails ──(blocked)──► Return safety message
        │
        ▼
  FAISS Semantic Retrieval (top-5 chunks)
        │
        ▼
  Groq openai/gpt-oss-120b
  (ConversationalRetrievalChain + citations)
        │
        ▼
  Output Guardrails (legal disclaimer, hallucination flag)
        │
        ▼
  Gradio UI (Answer + Sources)
```

---

## 🚀 Quick Start (Google Colab — Recommended)

1. Upload the project folder to Colab (or mount Google Drive)
2. Open **`SmartContractAssistant_Colab.ipynb`**
3. Add your Groq API key to Colab Secrets as `GROQ_API_KEY`  
   *(Left sidebar → 🔑 icon → Add secret)*
4. Run all cells — a public Gradio link will be generated automatically

---

## 💻 Local Setup

```bash
# 1. Clone & enter project
git clone <repo_url>
cd smart_contract_assistant

# 2. Create virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...

# 5. Start backend
uvicorn backend.api.server:app --host 0.0.0.0 --port 8000 --reload

# 6. Start frontend (new terminal)
python frontend/app.py
```

Then open: `http://localhost:7860`

---

## 🔧 Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *required* | Get at https://console.groq.com |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per query |
| `ENABLE_GUARDRAILS` | `true` | Toggle guardrails |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload PDF/DOCX |
| `POST` | `/ask` | Ask a question (RAG) |
| `POST` | `/summarize` | Summarize loaded document |
| `POST` | `/evaluate` | Run evaluation suite |
| `POST` | `/clear-history` | Reset conversation memory |
| `GET` | `/document-info` | Info about loaded document |
| `GET` | `/chain/playground` | LangServe interactive playground |

**Example: Ask a question**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms?", "enable_guardrails": true}'
```

---

## 🛡️ Guardrails

**Input checks:**
- Prompt injection detection (regex patterns)
- Question length validation (5–2000 chars)
- Off-topic relevance heuristics

**Output checks:**
- Legal disclaimer injection for legal advice questions
- Hallucination flag when no source chunks retrieved
- Prompt leakage detection & redaction

---

## 📊 Evaluation Metrics

| Metric | Method | Range |
|--------|--------|-------|
| **Faithfulness** | LLM judge: is answer grounded in retrieved chunks? | 0–1 |
| **Answer Relevance** | LLM judge: does answer address the question? | 0–1 |
| **Citation Coverage** | Heuristic: fraction of sources cited in answer | 0–1 |
| **Latency** | Wall-clock time per question | seconds |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## ⚠️ Limitations & Disclaimer

- English documents only (initial version)
- Performance depends on document quality (scanned PDFs may extract poorly)
- Not suitable for production without additional security hardening
- **This tool does not provide legal advice.** Always consult a qualified attorney.

---

## 🔮 Future Enhancements

- Multi-document search across a corpus
- Domain-specific fine-tuned embedding models
- Role-based access control
- Cloud deployment (Docker/Kubernetes)
- RAGAS integration for deeper evaluation
- Multi-language support
