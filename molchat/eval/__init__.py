"""Evaluation for MolChat (P3): retrieval A/B, tool-grounded faithfulness
judging (rule-based default, LLM-as-judge optional), and agent A/B (RAG on/off).

All default metrics run offline with no API key, so they execute in CI. The
LLM-as-judge backend is optional and used only when a key is present.
"""
