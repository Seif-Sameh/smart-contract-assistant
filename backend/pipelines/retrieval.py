"""
backend/pipelines/retrieval.py
RAG pipeline: semantic retrieval → LLM answer generation with source citations.
Supports conversation history.
"""

from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document

from backend.utils.config import config
from backend.pipelines.ingestion import load_vectorstore


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert contract analyst assistant. Your job is to answer questions about uploaded contracts, legal documents, or reports based STRICTLY on the provided context.

Rules:
1. Only answer based on the document context provided. Do NOT hallucinate or use outside knowledge.
2. Always cite the source chunk(s) you used in your answer using [Source: <filename>, Chunk <id>].
3. If the answer is not found in the context, clearly state: "I could not find this information in the uploaded document."
4. Be precise and professional. Use bullet points for lists.
5. Never provide legal advice. Always add a disclaimer for legal matters.

Context from document:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(
        "Chat History:\n{chat_history}\n\nQuestion: {question}"
    ),
])

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question, rephrase the follow-up question 
to be a standalone question that captures all necessary context.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""
)


# ─────────────────────────────────────────────
# Retrieval Pipeline
# ─────────────────────────────────────────────
class RetrievalPipeline:
    def __init__(self):
        self.vectorstore: Optional[FAISS] = None
        self.chain: Optional[ConversationalRetrievalChain] = None
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=6,  # keep last 6 turns
        )

    def load(self) -> bool:
        """Load vectorstore and build retrieval chain."""
        try:
            self.vectorstore = load_vectorstore(config.VECTORSTORE_PATH)
            retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": config.RETRIEVAL_TOP_K,
                    "score_threshold": config.RETRIEVAL_SCORE_THRESHOLD,
                },
            )
            llm = get_llm()
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": QA_PROMPT},
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                condense_question_llm=ChatGroq(
                    api_key=config.GROQ_API_KEY,
                    model=config.LLM_MODEL,
                    temperature=0.0,
                    max_tokens=256,
                ),
                verbose=False,
            )
            return True
        except Exception as e:
            print(f"[Retrieval] Failed to load pipeline: {e}")
            return False

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and return answer with sources.

        Returns:
            {
                "answer": str,
                "sources": [{"file": str, "chunk_id": int, "content": str}],
                "chat_history": list
            }
        """
        if not self.chain:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        result = self.chain.invoke({"question": question})

        # Format source documents
        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            key = (
                doc.metadata.get("source", "unknown"),
                doc.metadata.get("chunk_id", -1),
            )
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": doc.metadata.get("source", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", -1),
                    "content": doc.page_content[:300] + "..."
                    if len(doc.page_content) > 300
                    else doc.page_content,
                })

        return {
            "answer": result["answer"],
            "sources": sources,
        }

    def clear_history(self):
        """Reset conversation memory."""
        self.memory.clear()


# Singleton instance
_pipeline: Optional[RetrievalPipeline] = None


def get_pipeline() -> RetrievalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RetrievalPipeline()
    return _pipeline


def reload_pipeline() -> bool:
    """Force reload pipeline (call after new document ingestion)."""
    global _pipeline
    _pipeline = RetrievalPipeline()
    return _pipeline.load()
