"""Smoke test for relevance fix. Run: python backend/scripts/test_relevance_fix.py"""
import sys, os, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.app.guardrails.verifier import GuardrailVerifier
from backend.app.vector_store.store import VectorStore, MIN_SIMILARITY_THRESHOLD
from backend.app.ingestion.msmarco_ingestor import _make_chunk

print("=" * 60)
print("   RELEVANCE FIX SMOKE TEST")
print("=" * 60)
print(f"Vector store MIN_SIMILARITY_THRESHOLD = {MIN_SIMILARITY_THRESHOLD}")
print(f"Verifier RELEVANCE_THRESHOLD = {GuardrailVerifier.RELEVANCE_THRESHOLD}")
print(f"Verifier SIMILARITY_SCORE_THRESHOLD = {GuardrailVerifier.SIMILARITY_SCORE_THRESHOLD}")
print()

# Build isolated store with MEPAP passage (the bug repro) + ML passage (relevant)
tmp = tempfile.mktemp(suffix='.json')
vs = VectorStore(persistence_path=tmp)

mepap_chunk = _make_chunk(
    text="MEPAP stands for Medicaid Enrolled Provider Assistance Program. "
         "The MEPAP practicum requires 90 hours of supervised fieldwork and certification exam passing.",
    idx=0, query_id=1051211, eng_query="what is mepap certification",
    is_selected=1, passage_index=0, language="eng_Latn", target_lang="hin_Deva", strategy="recursive"
)
ml_chunk = _make_chunk(
    text="Machine learning is a method of data analysis that automates analytical model building "
         "using statistical algorithms trained on data.",
    idx=0, query_id=900001, eng_query="what is machine learning",
    is_selected=1, passage_index=0, language="eng_Latn", target_lang="hin_Deva", strategy="recursive"
)
vs.add_chunks([mepap_chunk, ml_chunk])

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

# ─── TEST 1: IRRELEVANT query — vector store threshold should block MEPAP ────
q1 = "give me a timetable to study of 12 hours"
r1 = vs.search(q1, top_k=4)
top1 = r1[0]["similarity_score"] if r1 else 0.0
ok1 = len(r1) == 0
print(f"[TEST 1 - IRRELEVANT QUERY: vector store gate]")
print(f"  Query     : {q1!r}")
print(f"  Results   : {len(r1)} chunks returned (top_score={top1:.4f})")
print(f"  Expected  : 0 chunks (score below {MIN_SIMILARITY_THRESHOLD})")
print(f"  Result    : {PASS if ok1 else FAIL}")
print()

# ─── TEST 2: RELEVANT query — ML chunk should come through ───────────────────
q2 = "What is machine learning?"
r2 = vs.search(q2, top_k=4)
top2 = r2[0]["similarity_score"] if r2 else 0.0
ok2 = len(r2) > 0 and r2[0]["query_id"] == 900001
print(f"[TEST 2 - RELEVANT QUERY: vector store allows through]")
print(f"  Query     : {q2!r}")
print(f"  Results   : {len(r2)} chunks (top_score={top2:.4f}, qid={r2[0]['query_id'] if r2 else 'N/A'})")
print(f"  Expected  : >= 1 chunk with qid=900001")
print(f"  Result    : {PASS if ok2 else FAIL}")
print()

# ─── TEST 3: Guardrail Pass 0 — off-topic even if sim score forced high ───────
q3 = "give me a timetable to study of 12 hours"
forced_mepap = dict(mepap_chunk)
forced_mepap["similarity_score"] = 0.45   # force past sim threshold to isolate Pass 0
forced_mepap["rerank_score"] = 0.50
v3 = GuardrailVerifier.verify_groundedness(
    q3, [forced_mepap],
    {"answer": "The MEPAP practicum requires 90 hours.", "confidence": 0.78, "can_answer": True, "sources": []}
)
ok3 = not v3["can_answer"]
print(f"[TEST 3 - GUARDRAIL PASS 0: off-topic rejected even with forced sim=0.45]")
print(f"  Query     : {q3!r}")
print(f"  Status    : {v3['guardrail_status']}")
print(f"  can_answer: {v3['can_answer']}")
print(f"  relevance : {v3.get('query_relevance_score', 'N/A')}")
print(f"  reason    : {v3.get('guardrail_reason', '')[:100]}")
print(f"  Expected  : can_answer=False, status=REJECTED_OFF_TOPIC")
print(f"  Result    : {PASS if ok3 else FAIL}")
print()

# ─── TEST 4: Guardrail PASSES for genuinely relevant query ───────────────────
q4 = "What is machine learning?"
ml_with_scores = dict(ml_chunk)
ml_with_scores["similarity_score"] = 0.62
ml_with_scores["rerank_score"] = 0.70
v4 = GuardrailVerifier.verify_groundedness(
    q4, [ml_with_scores],
    {
        "answer": "Machine learning is a method of data analysis that automates model building using statistical algorithms.",
        "confidence": 0.88, "can_answer": True, "sources": ["MSMARCO-XI (query_id=900001)"]
    }
)
ok4 = v4["can_answer"]
print(f"[TEST 4 - GUARDRAIL PASSES for relevant query]")
print(f"  Query     : {q4!r}")
print(f"  Status    : {v4['guardrail_status']}")
print(f"  can_answer: {v4['can_answer']}")
print(f"  relevance : {v4.get('query_relevance_score', 'N/A')}")
print(f"  reason    : {v4.get('guardrail_reason', '')[:100]}")
print(f"  Expected  : can_answer=True, status=PASSED")
print(f"  Result    : {PASS if ok4 else FAIL}")
print()

# ─── TEST 5: BORDERLINE — "impact of Manhattan Project" (off-corpus) ─────────
q5 = "What was the immediate impact of the success of the Manhattan Project?"
r5 = vs.search(q5, top_k=4)
top5 = r5[0]["similarity_score"] if r5 else 0.0
print(f"[TEST 5 - BORDERLINE: Manhattan Project (not in our small test store)]")
print(f"  Query     : {q5!r}")
print(f"  Results   : {len(r5)} chunks (top_score={top5:.4f})")
print(f"  Expected  : 0 chunks (no relevant passage in store)")
print(f"  Result    : {PASS if len(r5) == 0 else FAIL + ' (scored too high: ' + str(top5) + ')'}")
print()

total = sum([ok1, ok2, ok3, ok4])
print("=" * 60)
print(f"  SUMMARY: {total}/4 core tests passed")
print("=" * 60)
