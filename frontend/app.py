"""
frontend/app.py
Gradio UI — Upload, Chat, Summary & Evaluation tabs.
Designed to run in Google Colab or locally.
"""

import os
import time
import json
import tempfile
import requests
from pathlib import Path

import gradio as gr

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def api(endpoint: str, method: str = "GET", **kwargs):
    """Helper for backend API calls."""
    url = f"{BACKEND_URL}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=60)
        elif method == "POST_JSON":
            resp = requests.post(url, json=kwargs.get("data"), timeout=120)
        elif method == "POST_FILE":
            resp = requests.post(url, files=kwargs.get("files"),
                                 params=kwargs.get("params", {}), timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "❌ Cannot connect to backend. Make sure the FastAPI server is running."}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
chat_history = []
current_doc_info = {}


# ─────────────────────────────────────────────
# Upload Tab
# ─────────────────────────────────────────────
def upload_document(file, reset_store):
    if file is None:
        return "⚠️ Please select a file to upload.", ""

    with open(file.name, "rb") as f:
        file_content = f.read()
    file_name = Path(file.name).name
    content_type = "application/pdf" if file_name.endswith(".pdf") else \
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    result = api(
        "/upload",
        method="POST_FILE",
        files={"file": (file_name, file_content, content_type)},
        params={"reset_store": str(reset_store).lower()},
    )

    if "error" in result:
        return f"❌ {result['error']}", ""

    doc_detail = (
        f"📄 **{result.get('message', 'Uploaded')}**\n"
        f"- Chunks created: `{result.get('chunks', '?')}`\n"
        f"- Characters extracted: `{result.get('char_count', '?'):,}`"
    )
    status = "✅ Document ready — switch to the **Chat** tab to ask questions!"
    return status, doc_detail


# ─────────────────────────────────────────────
# Chat Tab
# ─────────────────────────────────────────────
def chat(message, history, enable_guardrails, show_sources):
    if not message.strip():
        return history, "", ""

    result = api(
        "/ask",
        method="POST_JSON",
        data={"question": message, "enable_guardrails": enable_guardrails},
    )

    if "error" in result:
        history.append((message, f"❌ {result['error']}"))
        return history, "", ""

    answer = result.get("answer", "No answer returned.")
    sources = result.get("sources", [])
    warning = result.get("guardrail_warning", "")
    latency = result.get("latency_seconds", 0)

    # Prepend warning if any
    if warning:
        answer = f"⚠️ {warning}\n\n{answer}"

    history.append((message, answer))

    # Format sources panel
    sources_text = ""
    if show_sources and sources:
        sources_text = f"**📎 Sources Retrieved ({len(sources)}) — {latency}s**\n\n"
        for i, src in enumerate(sources, 1):
            sources_text += (
                f"**[{i}] {src['file']} — Chunk #{src['chunk_id']}**\n"
                f"> {src['content']}\n\n"
            )
    elif show_sources:
        sources_text = "*No source chunks retrieved for this answer.*"

    return history, "", sources_text


def clear_chat_history():
    api("/clear-history", method="POST_JSON", data={})
    return [], "", ""


# ─────────────────────────────────────────────
# Summary Tab
# ─────────────────────────────────────────────
def generate_summary():
    result = api("/summarize", method="POST_JSON", data={})
    if "error" in result:
        return f"❌ {result['error']}"
    summary = result.get("summary", "Summary not available.")
    n_chunks = result.get("num_chunks_processed", "?")
    return f"*Processed {n_chunks} chunks*\n\n---\n\n{summary}"


# ─────────────────────────────────────────────
# Evaluation Tab
# ─────────────────────────────────────────────
def run_evaluation(questions_text):
    questions = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]
    if not questions:
        return "⚠️ Please enter at least one question (one per line).", ""

    result = api("/evaluate", method="POST_JSON", data={"questions": questions})

    if "error" in result:
        return f"❌ {result['error']}", ""

    # Summary metrics
    summary = (
        f"## 📊 Evaluation Results ({result['num_questions']} questions)\n\n"
        f"| Metric | Score |\n|--------|-------|\n"
        f"| Faithfulness | `{result['avg_faithfulness']:.2%}` |\n"
        f"| Answer Relevance | `{result['avg_answer_relevance']:.2%}` |\n"
        f"| Citation Coverage | `{result['avg_citation_coverage']:.2%}` |\n"
        f"| Avg Latency | `{result['avg_latency_seconds']}s` |\n"
    )

    # Per-question details
    details = "\n\n---\n\n## Per-Question Results\n\n"
    for i, r in enumerate(result.get("results", []), 1):
        details += (
            f"**Q{i}: {r['question']}**\n"
            f"- Faithfulness: `{r['faithfulness_score']:.2%}` | "
            f"Relevance: `{r['answer_relevance_score']:.2%}` | "
            f"Citations: `{r['citation_coverage']:.2%}` | "
            f"Latency: `{r['latency_seconds']}s`\n"
            f"- Answer: {r['answer'][:200]}...\n\n"
        )

    return summary, details


# ─────────────────────────────────────────────
# Build Gradio App
# ─────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="📜 Smart Contract Assistant",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .gradio-container { max-width: 1100px; margin: auto; }
        .sources-panel { font-size: 0.85em; background: #f8f9fa; padding: 12px; border-radius: 8px; }
        footer { display: none !important; }
        """,
    ) as demo:

        # Header
        gr.Markdown(
            """
            # 📜 Smart Contract Assistant
            **Powered by Groq `openai/gpt-oss-120b` · FAISS · LangChain · Sentence Transformers**
            
            Upload a contract or legal document, then ask questions, generate summaries, or run evaluations.
            """
        )

        with gr.Tabs():

            # ── Tab 1: Upload ──────────────────────────────────
            with gr.Tab("📁 Upload Document"):
                gr.Markdown("### Upload a PDF or DOCX contract/document")
                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="Select PDF or DOCX",
                            file_types=[".pdf", ".docx"],
                        )
                        reset_checkbox = gr.Checkbox(
                            label="Reset vector store (clear previous documents)",
                            value=True,
                        )
                        upload_btn = gr.Button("🚀 Upload & Process", variant="primary")
                    with gr.Column(scale=3):
                        upload_status = gr.Markdown("*No document uploaded yet.*")
                        upload_detail = gr.Markdown("")

                upload_btn.click(
                    upload_document,
                    inputs=[file_input, reset_checkbox],
                    outputs=[upload_status, upload_detail],
                )

            # ── Tab 2: Chat ────────────────────────────────────
            with gr.Tab("💬 Chat with Document"):
                gr.Markdown("### Ask questions about your uploaded document")
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=450,
                            bubble_full_width=False,
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask a question about the document...",
                                label="",
                                scale=5,
                                show_label=False,
                            )
                            send_btn = gr.Button("Send ➤", variant="primary", scale=1)
                        with gr.Row():
                            guardrails_toggle = gr.Checkbox(label="Enable Guardrails", value=True)
                            sources_toggle = gr.Checkbox(label="Show Sources", value=True)
                            clear_btn = gr.Button("🗑️ Clear History", scale=1)
                    with gr.Column(scale=2):
                        sources_panel = gr.Markdown(
                            "*Sources will appear here after each response.*",
                            elem_classes=["sources-panel"],
                        )

                send_btn.click(
                    chat,
                    inputs=[msg_input, chatbot, guardrails_toggle, sources_toggle],
                    outputs=[chatbot, msg_input, sources_panel],
                )
                msg_input.submit(
                    chat,
                    inputs=[msg_input, chatbot, guardrails_toggle, sources_toggle],
                    outputs=[chatbot, msg_input, sources_panel],
                )
                clear_btn.click(
                    clear_chat_history,
                    outputs=[chatbot, msg_input, sources_panel],
                )

                gr.Examples(
                    examples=[
                        ["What are the main obligations of each party?"],
                        ["What are the payment terms?"],
                        ["When does this contract expire or renew?"],
                        ["What are the termination conditions?"],
                        ["Are there any penalty clauses?"],
                        ["Summarize the key risks in this contract."],
                    ],
                    inputs=msg_input,
                    label="📌 Example Questions",
                )

            # ── Tab 3: Summary ─────────────────────────────────
            with gr.Tab("📋 Document Summary"):
                gr.Markdown(
                    "### Auto-generate a structured summary of your document\n"
                    "*Uses map-reduce summarization over the full document text.*"
                )
                summarize_btn = gr.Button("📋 Generate Summary", variant="primary")
                summary_output = gr.Markdown(
                    "*Upload a document first, then click to generate a summary.*"
                )
                summarize_btn.click(
                    generate_summary,
                    outputs=[summary_output],
                )

            # ── Tab 4: Evaluation ──────────────────────────────
            with gr.Tab("📊 Evaluation"):
                gr.Markdown(
                    "### RAG Pipeline Evaluation\n"
                    "Enter test questions (one per line) to evaluate faithfulness, "
                    "relevance, and citation coverage."
                )
                eval_input = gr.Textbox(
                    label="Test Questions (one per line)",
                    placeholder="What are the payment terms?\nWho are the parties involved?\nWhat is the contract duration?",
                    lines=6,
                )
                eval_btn = gr.Button("▶ Run Evaluation", variant="primary")
                with gr.Row():
                    eval_summary = gr.Markdown("*Results will appear here.*")
                    eval_details = gr.Markdown("")

                eval_btn.click(
                    run_evaluation,
                    inputs=[eval_input],
                    outputs=[eval_summary, eval_details],
                )

        # Footer
        gr.Markdown(
            """
            ---
            ⚠️ *This tool is for informational purposes only. It does not provide legal advice.*  
            🔒 *All processing is local. No data is sent externally except to the Groq API for LLM inference.*
            """
        )

    return demo


# ─────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,   # Creates public URL — useful for Colab
        show_error=True,
    )
