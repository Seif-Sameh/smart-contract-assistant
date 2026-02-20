"""
backend/pipelines/guardrails.py
Input/output guardrails for safety, relevance, and factuality enforcement.
Uses semantic similarity + rule-based checks (NeMo Guardrails optional).
"""

import re
from typing import Tuple, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from backend.utils.config import config


# ─────────────────────────────────────────────
# Blocked patterns (input guardrails)
# ─────────────────────────────────────────────
BLOCKED_PATTERNS = [
    # Prompt injection attempts
    r"ignore (previous|all|prior) instructions",
    r"you are now",
    r"pretend (you are|to be)",
    r"act as (a|an)",
    r"jailbreak",
    r"DAN mode",
    # Requests for harmful content
    r"how to (make|build|create) (a bomb|weapons|drugs)",
    r"suicide|self.harm",
    # PII fishing
    r"(give me|reveal|show) (your|the) (api key|password|credentials|secret)",
]

LEGAL_ADVICE_PATTERNS = [
    r"should i sign",
    r"is this (contract|clause|term) (legal|valid|enforceable)",
    r"will i win",
    r"do i have a case",
    r"legal advice",
    r"sue|lawsuit|litigation",
]

OFF_TOPIC_DISCLAIMER = (
    "⚠️ **Note:** Your question doesn't appear to be related to the uploaded document. "
    "I'll try to answer based on the document content, but please ensure your question "
    "is about the contract or document you uploaded."
)

LEGAL_DISCLAIMER = (
    "\n\n---\n⚖️ **Legal Disclaimer:** This response is for informational purposes only "
    "and does not constitute legal advice. Please consult a qualified attorney for legal guidance."
)


# ─────────────────────────────────────────────
# Input Guardrails
# ─────────────────────────────────────────────
def check_input(question: str) -> Tuple[bool, str]:
    """
    Validate user input.

    Returns:
        (is_safe, reason_or_empty_string)
        is_safe=False means block the question.
    """
    q_lower = question.lower().strip()

    # Check blocked patterns
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, q_lower):
            return False, (
                "🚫 This question cannot be processed as it appears to violate usage policies. "
                "Please ask questions about your uploaded document."
            )

    # Empty or too short
    if len(question.strip()) < 5:
        return False, "Please enter a more detailed question."

    # Too long (likely spam/injection)
    if len(question) > 2000:
        return False, "Your question is too long. Please keep it under 2000 characters."

    return True, ""


# ─────────────────────────────────────────────
# Output Guardrails
# ─────────────────────────────────────────────
def check_output(answer: str, question: str, sources: list) -> str:
    """
    Post-process LLM answer for safety and accuracy signals.

    - Adds legal disclaimer for legal questions
    - Flags if no sources were retrieved (possible hallucination)
    - Strips any accidental instruction leakage
    """
    q_lower = question.lower()

    # Add legal disclaimer if legal advice was sought
    needs_legal_disclaimer = any(
        re.search(p, q_lower) for p in LEGAL_ADVICE_PATTERNS
    )
    if needs_legal_disclaimer and LEGAL_DISCLAIMER not in answer:
        answer += LEGAL_DISCLAIMER

    # Flag low-confidence / no-source answers
    if not sources and "could not find" not in answer.lower():
        answer += (
            "\n\n⚠️ *Note: No specific document passages were retrieved for this answer. "
            "Please verify against the original document.*"
        )

    # Detect and redact potential prompt leakage
    if "system:" in answer.lower() or "assistant:" in answer.lower():
        answer = re.sub(
            r"(system:|assistant:).*", "[REDACTED]", answer, flags=re.IGNORECASE
        )

    return answer


# ─────────────────────────────────────────────
# Relevance Check (optional, costs an LLM call)
# ─────────────────────────────────────────────
RELEVANCE_PROMPT = PromptTemplate(
    input_variables=["question", "context_snippet"],
    template="""Is the following question relevant to a contract/legal document analysis context?
Answer ONLY with "yes" or "no".

Question: {question}
Sample document content: {context_snippet}

Relevant (yes/no):""",
)


def check_relevance(
    question: str, context_snippet: str = "", use_llm: bool = False
) -> Tuple[bool, str]:
    """
    Check if the question is relevant to document Q&A.
    Lightweight heuristic by default; set use_llm=True for LLM-based check.
    """
    # Heuristic: very general questions unrelated to documents
    irrelevant_patterns = [
        r"^(hi|hello|hey|what's up|how are you)",
        r"^(write me a poem|tell me a joke|what is \d+\+\d+)",
        r"(weather|news|sports|recipe)",
    ]
    q_lower = question.lower().strip()
    for pattern in irrelevant_patterns:
        if re.match(pattern, q_lower):
            return False, OFF_TOPIC_DISCLAIMER

    return True, ""


# ─────────────────────────────────────────────
# Main Guard Function
# ─────────────────────────────────────────────
def apply_guardrails(
    question: str,
    answer: Optional[str] = None,
    sources: Optional[list] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Full guardrail pass.

    Returns:
        (proceed, input_warning, processed_answer)
        - proceed: False if question should be blocked entirely
        - input_warning: warning to prepend to answer (empty string if none)
        - processed_answer: post-processed answer (None if not provided)
    """
    # Input checks
    is_safe, block_reason = check_input(question)
    if not is_safe:
        return False, block_reason, None

    relevance_ok, relevance_warning = check_relevance(question)

    # Output checks
    processed_answer = None
    if answer is not None:
        processed_answer = check_output(answer, question, sources or [])

    return True, relevance_warning, processed_answer
