import React, { useState } from 'react';
import { Play, CheckCircle2, ArrowRight, ShieldCheck, Zap, HelpCircle, Layers } from 'lucide-react';
import { sendRagQuery, QueryResponse } from '../services/api';

interface DemoModeProps {
  onExecuteDemoQuery: (query: string) => void;
  onNavigateTab: (tab: 'voice' | 'knowledge' | 'debugger' | 'observability') => void;
}

export const DemoMode: React.FC<DemoModeProps> = ({ onExecuteDemoQuery, onNavigateTab }) => {
  const [activeStep, setActiveStep] = useState(1);
  const [demoResponse, setDemoResponse] = useState<QueryResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const demoSteps = [
    {
      step: 1,
      title: "1. Upload Knowledge Documents",
      desc: "Demonstrate document parsing (PDF, TXT, MD, DOCX) and multi-strategy vector index creation.",
      actionLabel: "Open Knowledge Base UI",
      action: () => onNavigateTab('knowledge')
    },
    {
      step: 2,
      title: "2 & 3. Voice Query & STT Transcription",
      desc: "Test live voice input via Web Speech API and view real-time STT transcription.",
      actionLabel: "Go to Voice Assistant Screen",
      action: () => onNavigateTab('voice')
    },
    {
      step: 4,
      title: "4. Vector Search & Reranking",
      desc: "Execute query: 'What chunking strategies are supported by the system?'",
      actionLabel: "Run Grounded Query",
      action: async () => {
        setIsRunning(true);
        const res = await sendRagQuery("What chunking strategies are supported by the system?");
        setDemoResponse(res);
        setIsRunning(false);
      }
    },
    {
      step: 5,
      title: "5, 6 & 7. Grounded Answer, Citations & Latency",
      desc: "Verify zero-hallucination grounded generation, page/section citations, and timing breakdown.",
      actionLabel: "Inspect Retrieval Debugger",
      action: () => onNavigateTab('debugger')
    },
    {
      step: 8,
      title: "8 & 9. Unanswerable Question & Hallucination Guardrail",
      desc: "Test unanswerable query: 'What is the capital of Mars?' and verify guardrail rejection.",
      actionLabel: "Test Mars Capital Guardrail",
      action: async () => {
        setIsRunning(true);
        const res = await sendRagQuery("What is the capital of Mars?");
        setDemoResponse(res);
        setIsRunning(false);
      }
    },
    {
      step: 10,
      title: "10. Admin Telemetry & P50/P70/P100 Benchmark",
      desc: "Review real-time empirical latency metrics and stage breakdown charts.",
      actionLabel: "Open Observability Dashboard",
      action: () => onNavigateTab('observability')
    }
  ];

  return (
    <div className="max-w-5xl mx-auto space-y-6 px-4 py-2">
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <div className="flex items-center space-x-3 border-b border-slate-800 pb-4">
          <Play className="w-6 h-6 text-cyan-400" />
          <div>
            <h2 className="text-xl font-bold text-slate-100">Hacker House Goa 2026 Presentation Demo Mode</h2>
            <p className="text-xs text-slate-400">
              Guided 10-step walkthrough designed to present all core Voice-RAG requirements to judges.
            </p>
          </div>
        </div>

        {/* Step Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {demoSteps.map((item) => (
            <div
              key={item.step}
              className={`p-4 rounded-xl border transition-all ${
                activeStep === item.step
                  ? 'bg-cyan-500/10 border-cyan-400 text-slate-100 cyan-border-glow'
                  : 'bg-slate-950/80 border-slate-800 text-slate-400'
              }`}
            >
              <div className="flex items-center justify-between">
                <h3 className="font-bold text-sm text-cyan-300">{item.title}</h3>
                {activeStep > item.step && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
              </div>
              <p className="text-xs text-slate-300 mt-1 mb-3">{item.desc}</p>

              <button
                onClick={() => {
                  setActiveStep(item.step + 1);
                  item.action();
                }}
                disabled={isRunning}
                className="w-full flex items-center justify-center space-x-2 bg-slate-900 hover:bg-slate-800 text-cyan-400 text-xs py-2 rounded-lg border border-slate-700 transition-colors"
              >
                <span>{item.actionLabel}</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>

        {/* Demo Execution Output Panel */}
        {demoResponse && (
          <div className="glass-panel-glow p-5 rounded-xl space-y-3 mt-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-cyan-400">Demo Execution Result</span>
              <span className={`text-xs px-2 py-0.5 rounded font-mono border ${
                demoResponse.can_answer ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-amber-500/10 border-amber-500/30 text-amber-400'
              }`}>
                Guardrail: {demoResponse.debug.guardrail_status}
              </span>
            </div>

            <p className="text-sm text-slate-100 leading-relaxed font-medium">
              "{demoResponse.answer}"
            </p>

            <div className="flex items-center justify-between text-xs font-mono text-slate-400 pt-2 border-t border-slate-800">
              <span>Confidence: {Math.round(demoResponse.confidence * 100)}%</span>
              <span className="text-cyan-300 font-bold">Total Latency: {demoResponse.latency.total_ms} ms</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
