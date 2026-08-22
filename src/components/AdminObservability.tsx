import React, { useState, useEffect } from 'react';
import { BarChart3, Clock, CheckCircle, AlertTriangle, Zap, Activity, RefreshCw } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { fetchObservabilityMetrics, ObservabilityMetrics } from '../services/api';

export const AdminObservability: React.FC = () => {
  const [metrics, setMetrics] = useState<ObservabilityMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    setIsLoading(true);
    try {
      const data = await fetchObservabilityMetrics();
      setMetrics(data);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  if (!metrics) return null;

  const chartData = [
    { name: 'STT Avg', ms: metrics.latency_breakdown.stt_avg },
    { name: 'Embedding Avg', ms: metrics.latency_breakdown.embedding_avg },
    { name: 'Retrieval Avg', ms: metrics.latency_breakdown.retrieval_avg },
    { name: 'Reranking Avg', ms: metrics.latency_breakdown.reranking_avg },
    { name: 'LLM Gen Avg', ms: metrics.latency_breakdown.llm_avg },
  ];

  return (
    <div className="max-w-5xl mx-auto space-y-6 px-4 py-2">
      {/* Header */}
      <div className="glass-panel p-6 rounded-2xl flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Activity className="w-6 h-6 text-cyan-400" />
          <div>
            <h2 className="text-lg font-bold text-slate-100">Developer & Admin Observability Dashboard</h2>
            <p className="text-xs text-slate-400">
              Empirical latency percentiles, reliability metrics, and pipeline bottleneck telemetry.
            </p>
          </div>
        </div>

        <button
          onClick={loadMetrics}
          disabled={isLoading}
          className="bg-slate-900 hover:bg-slate-800 text-cyan-400 p-2 rounded-xl border border-slate-700 transition-colors"
          title="Refresh metrics"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Primary Telemetry Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass-panel p-4 rounded-xl space-y-1">
          <span className="text-xs text-slate-400 font-mono">Total System Queries</span>
          <div className="text-2xl font-extrabold text-slate-100 font-mono">{metrics.total_queries}</div>
          <span className="text-[11px] text-emerald-400">{metrics.successful_queries} Successful</span>
        </div>

        <div className="glass-panel p-4 rounded-xl space-y-1">
          <span className="text-xs text-slate-400 font-mono">P50 Median Latency</span>
          <div className="text-2xl font-extrabold text-cyan-400 font-mono">{metrics.latency_p50} ms</div>
          <span className="text-[11px] text-slate-400">50th Percentile Response</span>
        </div>

        <div className="glass-panel p-4 rounded-xl space-y-1">
          <span className="text-xs text-slate-400 font-mono">P70 Target Latency</span>
          <div className="text-2xl font-extrabold text-purple-400 font-mono">{metrics.latency_p70} ms</div>
          <span className="text-[11px] text-slate-400">70th Percentile Target</span>
        </div>

        <div className="glass-panel p-4 rounded-xl space-y-1">
          <span className="text-xs text-slate-400 font-mono">P100 Max Latency</span>
          <div className="text-2xl font-extrabold text-amber-400 font-mono">{metrics.latency_p100} ms</div>
          <span className="text-[11px] text-amber-400">{metrics.guardrail_rejections} Guardrail Rejections</span>
        </div>
      </div>

      {/* Pipeline Stage Latency Breakdown Chart */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <h3 className="text-sm font-semibold text-slate-200 font-mono uppercase tracking-wider flex items-center space-x-2">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          <span>Stage-by-Stage Average Latency Breakdown (ms)</span>
        </h3>

        <div className="h-64 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff' }}
              />
              <Bar dataKey="ms" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Telemetry Execution Logs Table */}
      <div className="glass-panel p-6 rounded-2xl space-y-3">
        <h3 className="text-sm font-semibold text-slate-200 font-mono uppercase tracking-wider">
          Recent Query Telemetry Trace Logs
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-900/80 text-slate-400 font-mono uppercase text-[11px] border-b border-slate-800">
              <tr>
                <th className="px-4 py-2.5">Query Snippet</th>
                <th className="px-4 py-2.5">Total (ms)</th>
                <th className="px-4 py-2.5">Vector Score</th>
                <th className="px-4 py-2.5">Chunks</th>
                <th className="px-4 py-2.5">Guardrail</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {metrics.recent_queries.map((q, i) => (
                <tr key={i} className="hover:bg-slate-900/40">
                  <td className="px-4 py-2.5 text-slate-200 truncate max-w-[240px]">"{q.query}"</td>
                  <td className="px-4 py-2.5 font-bold text-cyan-400">{q.total_latency_ms} ms</td>
                  <td className="px-4 py-2.5">{q.top_similarity_score}</td>
                  <td className="px-4 py-2.5">{q.chunks_count}</td>
                  <td className="px-4 py-2.5">
                    <span className={`px-2 py-0.5 rounded text-[10px] ${
                      q.guardrail_status === 'PASSED'
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                        : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                    }`}>
                      {q.guardrail_status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
