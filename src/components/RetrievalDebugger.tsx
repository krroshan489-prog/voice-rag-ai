import React from 'react';
import { Terminal, Database, ArrowRight, ShieldAlert, Cpu, Sparkles, CheckCircle, Globe } from 'lucide-react';
import { QueryResponse } from '../services/api';

interface DebuggerProps {
  lastResponse: QueryResponse | null;
}

export const RetrievalDebugger: React.FC<DebuggerProps> = ({ lastResponse }) => {
  if (!lastResponse || !lastResponse.debug) {
    return (
      <div className="max-w-5xl mx-auto glass-panel p-8 rounded-2xl text-center space-y-3">
        <Terminal className="w-10 h-10 text-slate-600 mx-auto" />
        <h3 className="text-slate-300 font-semibold text-base">Retrieval Debugger Standby</h3>
        <p className="text-slate-500 text-xs max-w-md mx-auto">
          Execute a voice or text query to inspect step-by-step MSMARCO-XI vector search, reranking, and guardrail outputs.
        </p>
        <div className="flex justify-center pt-2">
          <span className="flex items-center space-x-1.5 px-3 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-[11px] font-mono">
            <Database className="w-3 h-3" />
            <span>Knowledge Source: ai4bharat/MSMARCO-XI</span>
          </span>
        </div>
      </div>
    );
  }

  const { debug } = lastResponse;
  const groundednessPass = debug.groundedness_pass ?? 1;

  return (
    <div className="max-w-5xl mx-auto space-y-6 px-4 py-2">

      {/* ── Header ────────────────────────────────────────────────────────────── */}
      <div className="glass-panel p-5 rounded-2xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex items-center space-x-3">
          <Terminal className="w-6 h-6 text-cyan-400" />
          <div>
            <h2 className="text-lg font-bold text-slate-100">Step-by-Step RAG Retrieval Debugger</h2>
            <p className="text-xs text-slate-400">
              Transparent pipeline: MSMARCO-XI vector search → reranking → guardrail verification
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {/* Knowledge Source badge */}
          <span className="flex items-center space-x-1 px-2.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-[11px] font-mono">
            <Database className="w-3 h-3" />
            <span>MSMARCO-XI</span>
          </span>
          {/* Guardrail status badge */}
          <span className={`px-3 py-1 rounded-full text-xs font-mono border ${
            debug.guardrail_status === 'PASSED' || debug.guardrail_status === 'PASSED_STRICT_REGEN'
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
              : 'bg-amber-500/10 border-amber-500/30 text-amber-400'
          }`}>
            {debug.guardrail_status}
          </span>
          {/* Groundedness pass indicator */}
          {groundednessPass === 2 && (
            <span className="px-2.5 py-1 rounded-full text-[10px] font-mono border bg-purple-500/10 border-purple-500/30 text-purple-400">
              2nd-Pass Regen
            </span>
          )}
        </div>
      </div>

      {/* ── Visual Pipeline Steps ─────────────────────────────────────────────── */}
      <div className="space-y-4">

        {/* STEP 1: Query Preprocessing */}
        <div className="glass-panel p-5 rounded-xl space-y-2 border-l-4 border-l-cyan-400">
          <div className="flex items-center space-x-2 text-xs font-mono text-cyan-400">
            <span className="bg-cyan-500/20 px-2 py-0.5 rounded">STEP 1</span>
            <span>Query Normalizer & Sanitizer</span>
          </div>
          <p className="text-slate-200 font-mono text-sm">"{debug.sanitized_query}"</p>
        </div>

        {/* STEP 2: Vector Search */}
        <div className="glass-panel p-5 rounded-xl space-y-3 border-l-4 border-l-blue-500">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded flex items-center space-x-1">
              <Database className="w-3.5 h-3.5" />
              <span>STEP 2: MSMARCO-XI Dense Vector Search ({debug.retrieved_chunks.length} candidate chunks)</span>
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {debug.retrieved_chunks.map((c, i) => (
              <div key={i} className="bg-slate-950/80 p-3 rounded-lg border border-slate-800 text-xs space-y-1.5">
                <div className="flex items-center justify-between font-mono text-[11px] text-slate-400">
                  <span className="truncate max-w-[200px] text-cyan-300">{c.source_location}</span>
                  <span className="text-cyan-400 font-bold ml-2 flex-shrink-0">Sim: {c.similarity_score}</span>
                </div>
                {/* MSMARCO-XI metadata row */}
                {c.dataset_source === 'MSMARCO-XI' && (
                  <div className="flex items-center space-x-2 text-[10px] font-mono text-slate-500">
                    <span className="text-cyan-500">QID: {c.query_id}</span>
                    {c.is_selected === 1 && (
                      <span className="text-emerald-400 border border-emerald-500/30 px-1 rounded">Selected ✓</span>
                    )}
                    {c.target_lang && (
                      <span className="flex items-center space-x-0.5">
                        <Globe className="w-2.5 h-2.5" />
                        <span>{c.target_lang}</span>
                      </span>
                    )}
                  </div>
                )}
                <p className="text-slate-300 line-clamp-3 text-[11px] leading-relaxed">{c.text}</p>
              </div>
            ))}
          </div>
        </div>

        {/* STEP 3: Two-Stage Reranking */}
        <div className="glass-panel p-5 rounded-xl space-y-3 border-l-4 border-l-purple-500">
          <div className="flex items-center text-xs font-mono">
            <span className="bg-purple-500/20 text-purple-400 px-2 py-0.5 rounded flex items-center space-x-1">
              <Cpu className="w-3.5 h-3.5" />
              <span>STEP 3: Two-Stage Cross-Encoder Reranking</span>
            </span>
          </div>
          <div className="space-y-2">
            {debug.reranked_chunks.map((c, i) => (
              <div key={i} className="bg-slate-950/90 p-3 rounded-lg border border-purple-500/20 text-xs flex items-start justify-between gap-4">
                <div className="space-y-1 flex-1 min-w-0">
                  <div className="flex items-center space-x-2 flex-wrap">
                    <span className="font-mono text-cyan-300 text-[11px]">{c.source_location}</span>
                    {c.query_id !== undefined && (
                      <span className="font-mono text-[10px] text-slate-500">QID:{c.query_id}</span>
                    )}
                    {c.is_selected === 1 && (
                      <span className="text-[10px] text-emerald-400 border border-emerald-500/20 px-1 rounded">
                        Selected
                      </span>
                    )}
                  </div>
                  <p className="text-slate-200 text-[11px] leading-relaxed line-clamp-2">{c.text}</p>
                </div>
                <div className="font-mono text-right flex-shrink-0">
                  <span className="text-purple-400 font-bold text-sm block">Score: {c.rerank_score}</span>
                  <span className="text-[10px] text-slate-500">Rank #{i + 1}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* STEP 4 & 5: LLM Draft & Guardrail */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="glass-panel p-5 rounded-xl space-y-2 border-l-4 border-l-emerald-500">
            <div className="flex items-center space-x-2 text-xs font-mono text-emerald-400">
              <Sparkles className="w-3.5 h-3.5" />
              <span>
                STEP 4: Grounded LLM Draft
                {groundednessPass === 2 && (
                  <span className="ml-1 text-purple-400">(+ Strict Regen)</span>
                )}
              </span>
            </div>
            <pre className="bg-slate-950 p-3 rounded-lg text-emerald-300 text-[11px] font-mono whitespace-pre-wrap overflow-x-auto max-h-64">
              {JSON.stringify(debug.llm_draft, null, 2)}
            </pre>
          </div>

          <div className="glass-panel p-5 rounded-xl space-y-2 border-l-4 border-l-amber-500">
            <div className="flex items-center space-x-2 text-xs font-mono text-amber-400">
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>STEP 5: 3-Pass Guardrail Decision</span>
            </div>
            <div className="bg-slate-950 p-3 rounded-lg text-xs space-y-2 font-mono">
              <div className="flex justify-between">
                <span className="text-slate-400">Status:</span>
                <span className={`font-bold ${
                  debug.guardrail_status.startsWith('PASSED') ? 'text-emerald-400' : 'text-amber-300'
                }`}>{debug.guardrail_status}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Pass reached:</span>
                <span className="text-slate-300">{groundednessPass}</span>
              </div>
              {/* New: Pass 0 relevance & similarity scores */}
              {(debug as any).query_relevance_score !== undefined && (
                <div className="flex justify-between">
                  <span className="text-slate-400">Query relevance (Pass 0):</span>
                  <span className={`font-bold ${
                    (debug as any).query_relevance_score >= 0.12 ? 'text-emerald-400' : 'text-rose-400'
                  }`}>{((debug as any).query_relevance_score * 100).toFixed(0)}%</span>
                </div>
              )}
              {(debug as any).top_similarity_score !== undefined && (
                <div className="flex justify-between">
                  <span className="text-slate-400">Top similarity score:</span>
                  <span className={`font-bold ${
                    (debug as any).top_similarity_score >= 0.30 ? 'text-emerald-400' : 'text-rose-400'
                  }`}>{((debug as any).top_similarity_score).toFixed(3)}</span>
                </div>
              )}
              <div className="border-t border-slate-800 pt-1.5">
                <span className="text-slate-400 block mb-1">Verification rationale:</span>
                <p className="text-slate-300 text-[11px] leading-relaxed bg-slate-900/80 p-2 rounded border border-slate-800">
                  {debug.guardrail_reason}
                </p>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};
