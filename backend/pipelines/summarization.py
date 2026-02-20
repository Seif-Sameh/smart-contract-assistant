"""
backend/pipelines/summarization.py
Contract summarization using LangChain's map-reduce strategy for long documents.
"""

from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from backend.utils.config import config


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────
MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a contract analyst. Summarize the following section of a legal document.
Focus on: parties involved, key obligations, rights, dates, financial terms, and penalties.

Document Section:
{text}

Concise Summary:""",
)

COMBINE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are an expert contract analyst. Based on the following summaries of different sections,
create a comprehensive structured summary of the entire document.

Format your response as:
## Document Overview
[Brief description of document type and parties]

## Key Parties
[List all parties and their roles]

## Core Obligations & Rights
[Key obligations for each party]

## Financial Terms
[Fees, payments, penalties, compensation]

## Important Dates & Deadlines
[Effective dates, deadlines, renewal terms]

## Termination & Exit Clauses
[How the agreement can be ended]

## Risk Factors & Notable Clauses
[Unusual, risky, or important clauses to note]

## Disclaimer
*This summary is for informational purposes only and does not constitute legal advice.*

Section Summaries:
{text}

Comprehensive Summary:""",
)


# ─────────────────────────────────────────────
# Summarization Pipeline
# ─────────────────────────────────────────────
def summarize_document(raw_text: str, file_name: str = "document") -> Dict[str, Any]:
    """
    Summarize a full document using map-reduce strategy.

    Args:
        raw_text: Full extracted text from the document.
        file_name: Name of the document for display.

    Returns:
        dict with summary text and metadata.
    """
    llm = ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=config.LLM_MODEL,
        temperature=0.0,
        max_tokens=config.LLM_MAX_TOKENS,
    )

    # Split into chunks for map step
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.create_documents([raw_text])
    print(f"[Summarization] Summarizing {len(chunks)} chunks from '{file_name}'...")

    # Choose strategy based on document length
    if len(chunks) == 1:
        # Short doc — use stuff chain
        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=COMBINE_PROMPT,
            verbose=False,
        )
    else:
        # Long doc — use map-reduce
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=MAP_PROMPT,
            combine_prompt=COMBINE_PROMPT,
            verbose=False,
            token_max=4000,
        )

    result = chain.invoke({"input_documents": chunks})
    summary = result.get("output_text", "Summary could not be generated.")

    return {
        "status": "success",
        "file_name": file_name,
        "summary": summary,
        "num_chunks_processed": len(chunks),
        "original_length": len(raw_text),
        "summary_length": len(summary),
    }
