export interface LatencyBreakdown {
  stt_ms: number;
  embedding_ms: number;
  retrieval_ms: number;
  reranking_ms: number;
  llm_ms: number;
  total_ms: number;
}

export interface RetrievedChunk {
  chunk_id: string;
  text: string;
  document_name: string;
  page: number;
  section: string;
  source_location: string;
  similarity_score: number;
  rerank_score?: number;
  chunking_strategy: string;
  // MSMARCO-XI metadata
  dataset_source?: string;
  dataset_name?: string;
  query_id?: number;
  eng_query?: string;
  passage_index?: number;
  is_selected?: number;
  language_code?: string;
  target_lang?: string;
}

export interface MSMARCOSource {
  query_id: number;
  eng_query: string;
  passage_index: number;
  is_selected: number;
  language_code: string;
  target_lang: string;
  source_location: string;
  similarity_score: number;
  rerank_score: number;
  chunking_strategy: string;
}

export interface MSMARCOStats {
  indexed: boolean;
  total_chunks_in_store: number;
  msmarco_chunks: number;
  selected_passage_chunks: number;
  strategy_breakdown: Record<string, number>;
  marker: {
    status?: string;
    dataset?: string;
    strategy?: string;
    records_processed?: number;
    passages_processed?: number;
    chunks_added?: number;
    elapsed_sec?: number;
    indexed_at?: string;
  };
}

export interface QueryResponse {
  answer: string;
  confidence: number;
  sources: string[];
  can_answer: boolean;
  msmarco_sources?: MSMARCOSource[];
  knowledge_source?: string;
  latency: LatencyBreakdown;
  debug: {
    sanitized_query: string;
    retrieved_chunks: RetrievedChunk[];
    reranked_chunks: RetrievedChunk[];
    llm_draft: any;
    guardrail_status: string;
    guardrail_reason: string;
  };
  tts?: {
    text: string;
    latency_ms: number;
    status: string;
  };
}

export interface DocumentInfo {
  filename: string;
  pages: number;
  chunks_count: number;
  strategy: string;
  file_size: number;
}

export interface ObservabilityMetrics {
  total_queries: number;
  successful_queries: number;
  failed_queries: number;
  unanswered_queries: number;
  guardrail_rejections: number;
  latency_p50: number;
  latency_p70: number;
  latency_p100: number;
  latency_breakdown: {
    stt_avg: number;
    embedding_avg: number;
    retrieval_avg: number;
    reranking_avg: number;
    llm_avg: number;
  };
  recent_queries: any[];
}

const API_BASE = '/api';

async function fetchWithRetry(url: string, options: RequestInit, retries = 2, delay = 500): Promise<Response> {
  try {
    const res = await fetch(url, options);
    if (!res.ok && retries > 0) {
      await new Promise(resolve => setTimeout(resolve, delay));
      return fetchWithRetry(url, options, retries - 1, delay * 2);
    }
    return res;
  } catch (err) {
    if (retries > 0) {
      await new Promise(resolve => setTimeout(resolve, delay));
      return fetchWithRetry(url, options, retries - 1, delay * 2);
    }
    throw err;
  }
}

export async function sendRagQuery(
  query: string,
  chunkingStrategy: string = 'recursive',
  topK: number = 4,
  sttLatencyMs: number = 0
): Promise<QueryResponse> {
  const res = await fetchWithRetry(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      chunking_strategy: chunkingStrategy,
      top_k: topK,
      stt_latency_ms: sttLatencyMs,
    }),
  });

  if (!res.ok) {
    throw new Error(`Query failed: ${res.statusText}`);
  }

  return res.json();
}

export async function uploadDocument(
  file: File,
  chunkingStrategy: string = 'recursive',
  chunkSize: number = 500,
  chunkOverlap: number = 50
) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('chunking_strategy', chunkingStrategy);
  formData.append('chunk_size', chunkSize.toString());
  formData.append('chunk_overlap', chunkOverlap.toString());

  const res = await fetchWithRetry(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Upload failed: ${res.statusText}`);
  }

  return res.json();
}

export async function fetchDocuments(): Promise<{ total_documents: number; total_chunks: number; documents: DocumentInfo[] }> {
  const res = await fetchWithRetry(`${API_BASE}/documents`, { method: 'GET' });
  if (!res.ok) throw new Error('Failed to fetch documents');
  return res.json();
}

export async function deleteDocument(docName: string) {
  const res = await fetchWithRetry(`${API_BASE}/documents/${encodeURIComponent(docName)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to delete document');
  return res.json();
}

export async function reindexDocuments(strategy: string, chunkSize: number = 500) {
  const formData = new FormData();
  formData.append('strategy', strategy);
  formData.append('chunk_size', chunkSize.toString());

  const res = await fetchWithRetry(`${API_BASE}/documents/reindex`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Failed to reindex documents');
  return res.json();
}

export async function fetchObservabilityMetrics(): Promise<ObservabilityMetrics> {
  const res = await fetchWithRetry(`${API_BASE}/observability/metrics`, { method: 'GET' });
  if (!res.ok) throw new Error('Failed to fetch metrics');
  return res.json();
}

// ── MSMARCO-XI API ────────────────────────────────────────────────────────────

export async function fetchMSMARCOStats(): Promise<MSMARCOStats> {
  const res = await fetchWithRetry(`${API_BASE}/msmarco/stats`, { method: 'GET' });
  if (!res.ok) throw new Error('Failed to fetch MSMARCO-XI stats');
  return res.json();
}

export async function ingestMSMARCO(
  maxRecords: number = 200,
  strategy: string = 'recursive',
  force: boolean = false
) {
  const res = await fetchWithRetry(`${API_BASE}/msmarco/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ max_records: maxRecords, strategy, force }),
  });
  if (!res.ok) throw new Error(`MSMARCO-XI ingestion failed: ${res.statusText}`);
  return res.json();
}

export async function clearMSMARCOIndex() {
  const res = await fetchWithRetry(`${API_BASE}/msmarco/clear`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to clear MSMARCO-XI index');
  return res.json();
}

// Alias used by KnowledgeBase.tsx
export const triggerMSMARCOIngest = ingestMSMARCO;

