"""
backend/pipelines/evaluation.py
Evaluation pipeline for RAG quality using custom metrics
(faithfulness, relevance, citation coverage).
Optionally integrates with RAGAS for deeper evaluation.
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from backend.utils.config import config


@dataclass
class EvalResult:
    question: str
    answer: str
    sources_retrieved: int
    faithfulness_score: float       # 0-1: is answer grounded in sources?
    answer_relevance_score: float   # 0-1: does answer address the question?
    citation_coverage: float        # 0-1: fraction of answer claims cited
    latency_seconds: float
    notes: str = ""

    def to_dict(self):
        return asdict(self)


# ─────────────────────────────────────────────
# LLM-based Faithfulness Evaluator
# ─────────────────────────────────────────────
FAITHFULNESS_PROMPT = PromptTemplate(
    input_variables=["answer", "context"],
    template="""You are an evaluation judge. Score how faithfully the answer is grounded in the provided context.

Rules:
- Score 1.0: Every claim in the answer is directly supported by the context.
- Score 0.5: Most claims are supported; minor unsupported details.
- Score 0.0: Answer contains significant claims NOT found in the context.

Context:
{context}

Answer:
{answer}

Respond ONLY with a JSON object like: {{"score": 0.8, "reason": "brief reason"}}
""",
)

RELEVANCE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""Score how well the answer addresses the question asked.

- Score 1.0: Answer directly and completely addresses the question.
- Score 0.5: Answer partially addresses the question.
- Score 0.0: Answer does not address the question.

Question: {question}
Answer: {answer}

Respond ONLY with a JSON object like: {{"score": 0.9, "reason": "brief reason"}}
""",
)


def _llm_score(prompt_template: PromptTemplate, **kwargs) -> tuple[float, str]:
    """Run a scoring prompt and parse the JSON response."""
    llm = ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=config.LLM_MODEL,
        temperature=0.0,
        max_tokens=150,
    )
    prompt = prompt_template.format(**kwargs)
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        # Strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return float(parsed.get("score", 0.5)), parsed.get("reason", "")
    except Exception as e:
        print(f"[Eval] Scoring failed: {e}")
        return 0.5, f"Parse error: {e}"


def _citation_coverage(answer: str, sources: List[Dict]) -> float:
    """Heuristic: check what fraction of source chunks are cited in the answer."""
    if not sources:
        return 0.0
    cited = sum(
        1 for s in sources
        if f"Chunk {s.get('chunk_id', -999)}" in answer
        or s.get("file", "") in answer
    )
    return round(cited / len(sources), 2)


# ─────────────────────────────────────────────
# Evaluate Single QA Pair
# ─────────────────────────────────────────────
def evaluate_qa(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    latency: float = 0.0,
) -> EvalResult:
    """Evaluate a single QA pair."""
    context_text = "\n\n".join(
        [f"[Chunk {s['chunk_id']}]: {s['content']}" for s in sources]
    )

    faithfulness, f_note = _llm_score(
        FAITHFULNESS_PROMPT, answer=answer, context=context_text or "No context."
    )
    relevance, r_note = _llm_score(
        RELEVANCE_PROMPT, question=question, answer=answer
    )
    citation = _citation_coverage(answer, sources)

    return EvalResult(
        question=question,
        answer=answer,
        sources_retrieved=len(sources),
        faithfulness_score=round(faithfulness, 2),
        answer_relevance_score=round(relevance, 2),
        citation_coverage=citation,
        latency_seconds=round(latency, 2),
        notes=f"Faithfulness: {f_note} | Relevance: {r_note}",
    )


# ─────────────────────────────────────────────
# Batch Evaluation
# ─────────────────────────────────────────────
def run_evaluation_suite(
    test_cases: List[Dict[str, str]],
    pipeline,
) -> Dict[str, Any]:
    """
    Run a batch of test cases through the RAG pipeline and score each.

    Args:
        test_cases: List of {"question": str} dicts.
        pipeline: RetrievalPipeline instance.

    Returns:
        dict with per-question results and aggregate metrics.
    """
    results = []
    for tc in test_cases:
        question = tc["question"]
        start = time.time()
        try:
            response = pipeline.ask(question)
            latency = time.time() - start
            result = evaluate_qa(
                question=question,
                answer=response["answer"],
                sources=response["sources"],
                latency=latency,
            )
        except Exception as e:
            latency = time.time() - start
            result = EvalResult(
                question=question,
                answer=f"ERROR: {e}",
                sources_retrieved=0,
                faithfulness_score=0.0,
                answer_relevance_score=0.0,
                citation_coverage=0.0,
                latency_seconds=round(latency, 2),
                notes="Pipeline error",
            )
        results.append(result.to_dict())

    # Aggregate
    n = len(results)
    avg_faithfulness = sum(r["faithfulness_score"] for r in results) / n if n else 0
    avg_relevance = sum(r["answer_relevance_score"] for r in results) / n if n else 0
    avg_citation = sum(r["citation_coverage"] for r in results) / n if n else 0
    avg_latency = sum(r["latency_seconds"] for r in results) / n if n else 0

    return {
        "num_questions": n,
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_answer_relevance": round(avg_relevance, 3),
        "avg_citation_coverage": round(avg_citation, 3),
        "avg_latency_seconds": round(avg_latency, 2),
        "results": results,
    }
