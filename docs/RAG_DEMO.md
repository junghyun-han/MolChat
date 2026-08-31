# MolChat — real RAG generation demo

Generator backend: `transformers(Qwen/Qwen2.5-0.5B-Instruct,mps)`

### Q: Is aspirin likely to cross the blood-brain barrier, and why?
- retrieved passages: ['bbb_rules', 'stimulant', 'cns_design'], molecule: aspirin

**A:** Yes, aspirin is likely to cross the blood-brain barrier. It has a low polar surface area and moderate lipophilicity, making it easy for it to pass through the barrier.

### Q: What makes a molecule able to cross the blood-brain barrier?
- retrieved passages: ['bbb_rules', 'stimulant', 'cns_design']

**A:** A molecule can cross the blood-brain barrier if its topological polar surface area is below about 90 square angstroms, its molecular weight is under roughly 450 daltons, and its lipophilicity is moderate.

### Q: In one sentence, what is retrieval-augmented generation?
- retrieved passages: ['rag', 'vector_search', 'contrastive']

**A:** Retrieval-augmented generation uses retrieval-augmented systems to retrieve relevant documents from a knowledge base and supply them as context for a language model-generated answer.

