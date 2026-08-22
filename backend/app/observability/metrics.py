import numpy as np
import time
from typing import List, Dict, Any

class ObservabilityMetrics:
    """Tracks latency percentiles (P50, P70, P100), success/failure rates, and query stats."""

    def __init__(self):
        self.query_logs: List[Dict[str, Any]] = []

    def record_query(
        self,
        query: str,
        stt_latency_ms: float,
        embedding_latency_ms: float,
        retrieval_latency_ms: float,
        reranking_latency_ms: float,
        llm_latency_ms: float,
        total_latency_ms: float,
        chunks_count: int,
        top_similarity_score: float,
        guardrail_status: str,
        can_answer: bool,
        is_success: bool = True
    ):
        log_entry = {
            "timestamp": time.time(),
            "query": query,
            "stt_latency_ms": round(stt_latency_ms, 2),
            "embedding_latency_ms": round(embedding_latency_ms, 2),
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "reranking_latency_ms": round(reranking_latency_ms, 2),
            "llm_latency_ms": round(llm_latency_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "chunks_count": chunks_count,
            "top_similarity_score": round(top_similarity_score, 4),
            "guardrail_status": guardrail_status,
            "can_answer": can_answer,
            "is_success": is_success
        }
        self.query_logs.append(log_entry)

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        if not self.query_logs:
            return {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "unanswered_queries": 0,
                "guardrail_rejections": 0,
                "latency_p50": 0.0,
                "latency_p70": 0.0,
                "latency_p100": 0.0,
                "latency_breakdown": {
                    "stt_avg": 0.0,
                    "embedding_avg": 0.0,
                    "retrieval_avg": 0.0,
                    "reranking_avg": 0.0,
                    "llm_avg": 0.0
                },
                "recent_queries": []
            }

        total = len(self.query_logs)
        successful = sum(1 for q in self.query_logs if q["is_success"])
        failed = total - successful
        unanswered = sum(1 for q in self.query_logs if not q["can_answer"])
        guardrail_rejections = sum(1 for q in self.query_logs if "REJECTED" in q["guardrail_status"])

        totals_array = np.array([q["total_latency_ms"] for q in self.query_logs])
        
        p50 = float(np.percentile(totals_array, 50))
        p70 = float(np.percentile(totals_array, 70))
        p100 = float(np.max(totals_array))

        stt_avg = float(np.mean([q["stt_latency_ms"] for q in self.query_logs]))
        emb_avg = float(np.mean([q["embedding_latency_ms"] for q in self.query_logs]))
        ret_avg = float(np.mean([q["retrieval_latency_ms"] for q in self.query_logs]))
        rerank_avg = float(np.mean([q["reranking_latency_ms"] for q in self.query_logs]))
        llm_avg = float(np.mean([q["llm_latency_ms"] for q in self.query_logs]))

        return {
            "total_queries": total,
            "successful_queries": successful,
            "failed_queries": failed,
            "unanswered_queries": unanswered,
            "guardrail_rejections": guardrail_rejections,
            "latency_p50": round(p50, 2),
            "latency_p70": round(p70, 2),
            "latency_p100": round(p100, 2),
            "latency_breakdown": {
                "stt_avg": round(stt_avg, 2),
                "embedding_avg": round(emb_avg, 2),
                "retrieval_avg": round(ret_avg, 2),
                "reranking_avg": round(rerank_avg, 2),
                "llm_avg": round(llm_avg, 2)
            },
            "recent_queries": self.query_logs[-15:]
        }

metrics_tracker = ObservabilityMetrics()
