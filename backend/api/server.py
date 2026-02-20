"""
backend/api/server.py
FastAPI + LangServe microservice exposing all pipelines as REST endpoints.
"""

import os
import time
import tempfile
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableLambda

try:
    from langserve import add_routes
    LANGSERVE_AVAILABLE = True
except ImportError:
    LANGSERVE_AVAILABLE = False
    print("[Server] LangServe not available, skipping /chain routes")

from backend.utils.config import config
from backend.pipelines.ingestion import ingest_document
from backend.pipelines.retrieval import get_pipeline, reload_pipeline
from backend.pipelines.summarization import summarize_document
from backend.pipelines.guardrails import apply_guardrails
from backend.pipelines.evaluation import run_evaluation_suite

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Smart Contract Assistant API",
    description="RAG-powered contract Q&A with summarization, guardrails & evaluation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for ingested document metadata
_doc_store: dict = {}


# ─────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    enable_guardrails: bool = True


class QuestionResponse(BaseModel):
    answer: str
    sources: List[dict]
    guardrail_warning: str = ""
    latency_seconds: float = 0.0


class EvalRequest(BaseModel):
    questions: List[str]


class ClearRequest(BaseModel):
    confirm: bool = False


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": config.LLM_MODEL}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    reset_store: bool = False,
):
    """Upload and ingest a PDF or DOCX document."""
    allowed_types = ["application/pdf",
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                     "application/msword"]
    if file.content_type not in allowed_types and not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(400, "Only PDF and DOCX files are supported.")

    # Save to temp file
    suffix = ".pdf" if file.filename.endswith(".pdf") else ".docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingest_document(tmp_path, reset_store=reset_store)
        _doc_store["current"] = result
        # Reload retrieval pipeline after new ingestion
        reload_pipeline()
        return JSONResponse(content={
            "status": "success",
            "message": f"Document '{result['file_name']}' ingested successfully.",
            "chunks": result["num_chunks"],
            "char_count": result["char_count"],
        })
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/ask", response_model=QuestionResponse)
def ask_question(req: QuestionRequest):
    """Ask a question about the uploaded document."""
    start = time.time()

    # Input guardrails
    if req.enable_guardrails:
        proceed, warning, _ = apply_guardrails(req.question)
        if not proceed:
            return QuestionResponse(
                answer=warning,
                sources=[],
                guardrail_warning=warning,
            )
    else:
        warning = ""

    pipeline = get_pipeline()
    if not pipeline.chain:
        loaded = pipeline.load()
        if not loaded:
            raise HTTPException(400, "No document has been uploaded yet. Please upload a document first.")

    try:
        result = pipeline.ask(req.question)
    except Exception as e:
        raise HTTPException(500, f"Retrieval failed: {str(e)}")

    # Output guardrails
    if req.enable_guardrails:
        _, _, processed_answer = apply_guardrails(
            req.question,
            answer=result["answer"],
            sources=result["sources"],
        )
        answer = processed_answer or result["answer"]
    else:
        answer = result["answer"]

    return QuestionResponse(
        answer=answer,
        sources=result["sources"],
        guardrail_warning=warning,
        latency_seconds=round(time.time() - start, 2),
    )


@app.post("/summarize")
def summarize():
    """Summarize the currently loaded document."""
    if "current" not in _doc_store:
        raise HTTPException(400, "No document ingested yet.")
    doc = _doc_store["current"]
    try:
        result = summarize_document(
            raw_text=doc["raw_text"],
            file_name=doc["file_name"],
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(500, f"Summarization failed: {str(e)}")


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    """Run evaluation suite on provided questions."""
    pipeline = get_pipeline()
    if not pipeline.chain:
        loaded = pipeline.load()
        if not loaded:
            raise HTTPException(400, "No document loaded for evaluation.")

    test_cases = [{"question": q} for q in req.questions]
    try:
        results = run_evaluation_suite(test_cases, pipeline)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {str(e)}")


@app.post("/clear-history")
def clear_history():
    """Clear conversation memory."""
    pipeline = get_pipeline()
    pipeline.clear_history()
    return {"status": "Conversation history cleared."}


@app.get("/document-info")
def document_info():
    """Get info about the currently loaded document."""
    if "current" not in _doc_store:
        return {"status": "No document loaded."}
    doc = _doc_store["current"]
    return {
        "file_name": doc["file_name"],
        "file_type": doc["file_type"],
        "num_chunks": doc["num_chunks"],
        "char_count": doc["char_count"],
    }


# ─────────────────────────────────────────────
# LangServe Route (optional)
# ─────────────────────────────────────────────
if LANGSERVE_AVAILABLE:
    def _ask_chain(inputs: dict) -> dict:
        pipeline = get_pipeline()
        if not pipeline.chain:
            pipeline.load()
        return pipeline.ask(inputs.get("question", ""))

    add_routes(
        app,
        RunnableLambda(_ask_chain),
        path="/chain",
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    config.validate()
    uvicorn.run(
        "backend.api.server:app",
        host=config.BACKEND_HOST,
        port=config.BACKEND_PORT,
        reload=True,
    )
