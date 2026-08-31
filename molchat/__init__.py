"""MolChat — a RAG + function-calling agent over molecular representations.

Built on top of the Moleco chemical-language-model project (used with permission)
as a personal engineering extension: the molecular embedding produced by the
representation model is used to populate a vector store, and a tool-calling agent
answers natural-language questions about molecules by invoking RDKit descriptors,
a property predictor, and molecular-similarity retrieval.

The public surface is intentionally small:
    - embeddings.get_embedder()  -> pluggable molecular embedding backend
    - vectorstore.VectorStore    -> FAISS-backed similarity index
    - rag.RagIndex               -> retrieval over the bundled molecule corpus
    - tools                      -> function-calling tool registry + JSON schemas
    - agent.Agent                -> the tool-calling loop
    - llm.get_llm()              -> pluggable LLM backend (rule-based default)
"""

__version__ = "0.1.0"
