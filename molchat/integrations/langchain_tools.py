"""LangChain integration: MolChat tools + retriever + an LCEL RAG chain.

This wires MolChat's existing pieces into real LangChain primitives:
- the four tools become ``StructuredTool`` objects (usable by any LangChain agent),
- the text knowledge base becomes a LangChain ``BaseRetriever``,
- an LCEL chain (retriever -> prompt -> LLM -> parser) performs RAG.

Requires ``langchain-core`` (optional dependency). The LLM step accepts any
callable ``(system, user) -> str`` so it runs with the local transformers model
or a stub in tests.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import StructuredTool

from ..knowledge import TextKnowledgeBase
from ..tools import Toolbox

_SYSTEM = (
    "You are MolChat, a molecular chemistry assistant. Answer using ONLY the "
    "provided context. Be concise and do not invent numbers."
)


def build_langchain_tools(toolbox: Optional[Toolbox] = None) -> List[StructuredTool]:
    tb = toolbox or Toolbox()

    def rdkit_descriptors(molecule: str) -> str:
        """Exact physicochemical descriptors for a molecule name or SMILES."""
        return str(tb.rdkit_descriptors(molecule))

    def moleco_predict(molecule: str) -> str:
        """Blood-brain-barrier permeability prediction for a molecule."""
        return str(tb.moleco_predict(molecule))

    def rag_search(molecule: str) -> str:
        """Most similar molecules by molecular embedding."""
        return str(tb.rag_search(molecule))

    def knowledge_search(query: str) -> str:
        """Relevant text passages from the chemistry/ML knowledge base."""
        return str(tb.knowledge_search(query))

    return [
        StructuredTool.from_function(rdkit_descriptors),
        StructuredTool.from_function(moleco_predict),
        StructuredTool.from_function(rag_search),
        StructuredTool.from_function(knowledge_search),
    ]


class MolChatRetriever(BaseRetriever):
    """LangChain retriever over the MolChat text knowledge base."""

    kb: TextKnowledgeBase
    k: int = 3
    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return [
            Document(
                page_content=f"{p.title}: {p.text}",
                metadata={"id": p.id, "score": p.score},
            )
            for p in self.kb.search(query, k=self.k)
        ]


def _format_docs(docs: List[Document]) -> str:
    return "\n".join(f"- {d.page_content}" for d in docs)


def build_rag_chain(generator: Optional[Callable] = None, k: int = 3):
    """Return an LCEL RAG chain: retrieve -> prompt -> LLM -> string.

    ``generator`` is any callable ``(system, user) -> str``. If None, the local
    transformers generator is used (downloads a small model on first run).
    """
    if generator is None:
        from ..rag_generate import get_generator

        generator = get_generator()

    retriever = MolChatRetriever(kb=TextKnowledgeBase(), k=k)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", "Context:\n{context}\n\nQuestion: {question}")]
    )

    def _run_llm(prompt_value) -> str:
        return generator(_SYSTEM, prompt_value.to_string())

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | RunnableLambda(_run_llm)
        | StrOutputParser()
    )
