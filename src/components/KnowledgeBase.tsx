import React, { useState, useEffect } from 'react';
import {
  Upload, FileText, Trash2, Layers, RefreshCw, CheckCircle2,
  FileUp, AlertCircle, Database, Activity, Globe, Loader2
} from 'lucide-react';
import {
  fetchDocuments, uploadDocument, deleteDocument, reindexDocuments,
  fetchMSMARCOStats, triggerMSMARCOIngest, DocumentInfo, MSMARCOStats
} from '../services/api';

interface KnowledgeBaseProps {
  currentStrategy: string;
  onStrategyChange: (strategy: string) => void;
}

export const KnowledgeBase: React.FC<KnowledgeBaseProps> = ({ currentStrategy, onStrategyChange }) => {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [isReindexing, setIsReindexing] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [msmarcoStats, setMsmarcoStats] = useState<MSMARCOStats | null>(null);

  useEffect(() => {
    loadDocuments();
    loadMSMARCOStats();
  }, []);

  const loadDocuments = async () => {
    try {
      const data = await fetchDocuments();
      setDocuments(data.documents);
      setTotalChunks(data.total_chunks);
    } catch (err) {
      console.error(err);
    }
  };

  const loadMSMARCOStats = async () => {
    try {
      const stats = await fetchMSMARCOStats();
      setMsmarcoStats(stats);
    } catch (err) {
      console.error('Failed to load MSMARCO stats', err);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setIsUploading(true);
    setUploadStatus(null);
    try {
      for (let i = 0; i < files.length; i++) {
        await uploadDocument(files[i], currentStrategy, 500, 50);
      }
      setUploadStatus('Document uploaded and indexed successfully!');
      await loadDocuments();
    } catch (err: any) {
      setUploadStatus(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (docName: string) => {
    try {
      await deleteDocument(docName);
      await loadDocuments();
    } catch (err) {
      console.error(err);
    }
  };

  const handleReindex = async () => {
    setIsReindexing(true);
    try {
      await reindexDocuments(currentStrategy);
      await loadDocuments();
    } catch (err) {
      console.error(err);
    } finally {
      setIsReindexing(false);
    }
  };

  const handleIngestMSMARCO = async (force: boolean = false) => {
    setIsIngesting(true);
    try {
      await triggerMSMARCOIngest(5000, currentStrategy, force);
      // Poll for completion
      setTimeout(async () => {
        await loadMSMARCOStats();
        await loadDocuments();
        setIsIngesting(false);
      }, 3000);
    } catch (err) {
      console.error(err);
      setIsIngesting(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6 px-4 py-2">

      {/* ── MSMARCO-XI Corpus Status Card ────────────────────────────────────── */}
      <div className="glass-panel p-6 rounded-2xl space-y-4 border border-cyan-500/20">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-cyan-500/10 rounded-lg border border-cyan-500/30">
              <Database className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100">
                Knowledge Source: MSMARCO-XI
              </h2>
              <p className="text-[11px] text-slate-400 font-mono">
                ai4bharat/MSMARCO-XI · Multilingual MS MARCO · 97,941 validation records
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {msmarcoStats?.indexed ? (
              <span className="flex items-center space-x-1 text-xs px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-mono">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Indexed</span>
              </span>
            ) : (
              <span className="flex items-center space-x-1 text-xs px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-400 font-mono">
                <AlertCircle className="w-3.5 h-3.5" />
                <span>Not Indexed</span>
              </span>
            )}
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            {
              label: 'Total Chunks',
              value: msmarcoStats?.total_chunks_in_store?.toLocaleString() ?? '—',
              icon: <Layers className="w-4 h-4 text-cyan-400" />,
            },
            {
              label: 'Unique Records',
              value: msmarcoStats?.marker?.records_processed?.toLocaleString() ?? '—',
              icon: <FileText className="w-4 h-4 text-purple-400" />,
            },
            {
              label: 'Languages',
              value: '14 Indic',
              icon: <Globe className="w-4 h-4 text-emerald-400" />,
            },
            {
              label: 'Index Strategy',
              value: msmarcoStats?.index_metadata?.strategy ?? currentStrategy,
              icon: <Activity className="w-4 h-4 text-amber-400" />,
            },
          ].map((stat) => (
            <div
              key={stat.label}
              className="bg-slate-900/60 rounded-xl p-3 border border-slate-800 text-center space-y-1"
            >
              <div className="flex justify-center">{stat.icon}</div>
              <div className="text-lg font-bold text-slate-100 font-mono">{stat.value}</div>
              <div className="text-[10px] text-slate-400 uppercase tracking-wide">{stat.label}</div>
            </div>
          ))}
        </div>

        {/* Index Metadata */}
        {msmarcoStats?.index_metadata?.indexed_at && (
          <div className="text-[10px] text-slate-500 font-mono flex flex-wrap gap-4">
            <span>Records Processed: {msmarcoStats.index_metadata.records_processed?.toLocaleString()}</span>
            <span>Passages: {msmarcoStats.index_metadata.passages_processed?.toLocaleString()}</span>
            <span>Elapsed: {msmarcoStats.index_metadata.elapsed_sec}s</span>
            <span>Indexed At: {new Date(msmarcoStats.index_metadata.indexed_at).toLocaleString()}</span>
          </div>
        )}

        {/* Re-Ingest Actions */}
        <div className="flex items-center space-x-3 pt-1">
          <button
            onClick={() => handleIngestMSMARCO(false)}
            disabled={isIngesting || !!msmarcoStats?.is_indexed}
            className="flex items-center space-x-2 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 text-xs px-4 py-2 rounded-xl border border-cyan-500/30 disabled:opacity-40 transition-all"
          >
            {isIngesting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Database className="w-3.5 h-3.5" />}
            <span>{isIngesting ? 'Ingesting MSMARCO-XI...' : 'Ingest MSMARCO-XI'}</span>
          </button>
          <button
            onClick={() => handleIngestMSMARCO(true)}
            disabled={isIngesting}
            className="flex items-center space-x-2 bg-slate-800 hover:bg-slate-700 text-slate-400 text-xs px-4 py-2 rounded-xl border border-slate-700 disabled:opacity-40 transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isIngesting ? 'animate-spin' : ''}`} />
            <span>Force Re-Ingest</span>
          </button>
        </div>
      </div>

      {/* ── Chunking Strategy Configurator ───────────────────────────────────── */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
          <div>
            <h2 className="text-xl font-bold text-slate-100 flex items-center space-x-2">
              <FileText className="w-5 h-5 text-cyan-400" />
              <span>RAG Knowledge Base & Ingestion Pipeline</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Manage uploaded documents and configure multi-strategy vector chunking algorithms.
            </p>
          </div>
          <div className="flex items-center space-x-3 text-xs font-mono">
            <span className="bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300">
              Total Docs: <strong className="text-cyan-400">{documents.length}</strong>
            </span>
            <span className="bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300">
              Total Chunks: <strong className="text-cyan-400">{totalChunks}</strong>
            </span>
          </div>
        </div>

        {/* Chunking Strategy Selector */}
        <div className="space-y-2">
          <label className="text-xs text-slate-400 font-mono flex items-center space-x-1">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span>Select Chunking Strategy:</span>
          </label>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {[
              { id: 'fixed', label: 'A. Fixed-Size Chunking', desc: 'Fixed character windowing with token overlap' },
              { id: 'recursive', label: 'B. Recursive / Semantic', desc: 'Splits on headers, paragraphs, and sentences' },
              { id: 'structure_aware', label: 'C. Structure-Aware', desc: 'Preserves headers, lists, sections, and tables' },
            ].map((strat) => (
              <button
                key={strat.id}
                onClick={() => onStrategyChange(strat.id)}
                className={`p-3 rounded-xl text-left border transition-all ${
                  currentStrategy === strat.id
                    ? 'bg-cyan-500/10 border-cyan-400 text-cyan-300 cyan-border-glow'
                    : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
                }`}
              >
                <div className="font-semibold text-sm">{strat.label}</div>
                <div className="text-xs opacity-75 mt-0.5">{strat.desc}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center justify-between pt-2">
          <span className="text-xs text-slate-400 italic">
            Re-indexing rebuilds vector embeddings using the active strategy (user docs only; MSMARCO-XI preserved).
          </span>
          <button
            onClick={handleReindex}
            disabled={isReindexing || documents.length === 0}
            className="flex items-center space-x-2 bg-slate-800 hover:bg-slate-700 text-cyan-400 text-xs px-4 py-2 rounded-xl border border-slate-700 disabled:opacity-50 transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isReindexing ? 'animate-spin' : ''}`} />
            <span>{isReindexing ? 'Re-Indexing...' : 'Re-Index Documents'}</span>
          </button>
        </div>
      </div>

      {/* ── Upload Zone ──────────────────────────────────────────────────────── */}
      <div className="glass-panel p-6 rounded-2xl text-center border-dashed border-2 border-slate-700 hover:border-cyan-500/50 transition-all">
        <label className="cursor-pointer flex flex-col items-center space-y-3">
          <div className="p-4 bg-slate-900 rounded-full border border-slate-800 text-cyan-400">
            <FileUp className="w-8 h-8" />
          </div>
          <div>
            <span className="text-slate-200 font-medium text-sm">
              Click or drag documents to upload (PDF, TXT, MD, DOCX)
            </span>
            <p className="text-xs text-slate-500 mt-1">
              Supports automated text cleaning, metadata extraction, and vector index generation.
            </p>
          </div>
          <input
            type="file"
            multiple
            accept=".pdf,.txt,.md,.markdown,.docx"
            onChange={handleFileUpload}
            className="hidden"
          />
        </label>
        {isUploading && (
          <div className="mt-3 text-xs text-cyan-400 font-mono animate-pulse">
            Processing & Indexing Document Chunks...
          </div>
        )}
        {uploadStatus && (
          <div className="mt-3 text-xs text-emerald-400 flex items-center justify-center space-x-1">
            <CheckCircle2 className="w-4 h-4" />
            <span>{uploadStatus}</span>
          </div>
        )}
      </div>

      {/* ── Documents Registry Table ─────────────────────────────────────────── */}
      <div className="glass-panel p-6 rounded-2xl space-y-3">
        <h3 className="text-sm font-semibold text-slate-300 font-mono uppercase tracking-wider">
          Ingested Documents Registry
        </h3>
        {documents.length === 0 ? (
          <div className="text-center py-8 text-slate-500 text-sm">
            No user-uploaded documents. MSMARCO-XI is the primary knowledge source.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-900/80 text-slate-400 font-mono uppercase text-[11px] border-b border-slate-800">
                <tr>
                  <th className="px-4 py-3">Document Name</th>
                  <th className="px-4 py-3">Pages</th>
                  <th className="px-4 py-3">Chunks</th>
                  <th className="px-4 py-3">Strategy</th>
                  <th className="px-4 py-3">Size</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {documents.map((doc, i) => (
                  <tr key={i} className="hover:bg-slate-900/40 transition-colors">
                    <td className="px-4 py-3 font-medium text-slate-200 flex items-center space-x-2">
                      <FileText className="w-4 h-4 text-cyan-400" />
                      <span>{doc.filename}</span>
                    </td>
                    <td className="px-4 py-3">{doc.pages}</td>
                    <td className="px-4 py-3 font-mono font-bold text-cyan-400">{doc.chunks_count}</td>
                    <td className="px-4 py-3">
                      <span className="bg-slate-900 border border-slate-700 px-2 py-0.5 rounded text-[10px] font-mono text-slate-300 uppercase">
                        {doc.strategy}
                      </span>
                    </td>
                    <td className="px-4 py-3">{Math.round(doc.file_size / 1024)} KB</td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => handleDelete(doc.filename)}
                        className="text-rose-400 hover:text-rose-300 p-1 rounded transition-colors"
                        title="Delete Document"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
