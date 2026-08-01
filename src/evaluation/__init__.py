"""
Evaluation package.

retrieval_metrics is imported eagerly (pure Python, no heavy deps).

LLMJudge / JudgeScore are lazy-loaded via PEP 562 __getattr__ because they
pull in openai — and importing openai before DenseRetriever on Windows
causes a later torch load to hang. Same fix as retrieval/__init__.py.

Consumers can still write:
    from evaluation import LLMJudge   # triggers lazy import at first access
Or import directly from the module:
    from evaluation.judge import LLMJudge
"""

from evaluation.retrieval_metrics import (
    aggregate_metrics,
    evaluate_query,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    page_in_range_rate,
    precision_at_k,
    recall_at_k,
    retrieved_doc_ids,
)

__all__ = [
    "JudgeScore",
    "LLMJudge",
    "aggregate_metrics",
    "evaluate_query",
    "hit_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "page_in_range_rate",
    "precision_at_k",
    "recall_at_k",
    "retrieved_doc_ids",
]


def __getattr__(name: str):
    if name in ("LLMJudge", "JudgeScore"):
        from evaluation.judge import JudgeScore, LLMJudge
        return {"LLMJudge": LLMJudge, "JudgeScore": JudgeScore}[name]
    raise AttributeError(f"module 'evaluation' has no attribute {name!r}")
