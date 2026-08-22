import React, { useState } from 'react';
import { Mic, FileText, Terminal, Activity, Play, Sparkles, Cpu, Layers } from 'lucide-react';
import { VoiceAssistant } from './components/VoiceAssistant';
import { KnowledgeBase } from './components/KnowledgeBase';
import { RetrievalDebugger } from './components/RetrievalDebugger';
import { AdminObservability } from './components/AdminObservability';
import { DemoMode } from './components/DemoMode';
import { QueryResponse } from './services/api';

// Lightweight CSS-only animated particle background — no canvas, no library
const PARTICLES = Array.from({ length: 25 }, (_, i) => ({
  id: i,
  size:  2 + (i % 4),                          // 2–5px
  left:  (i * 37 + 11) % 100,                  // spread 0–99%
  top:   (i * 53 + 7)  % 100,
  delay: (i * 0.4) % 8,                        // stagger 0–8s
  dur:   12 + (i % 8),                         // 12–19s cycle
  opacity: 0.08 + (i % 5) * 0.06,             // 0.08–0.32
  color: i % 3 === 0 ? '#22d3ee' : i % 3 === 1 ? '#818cf8' : '#34d399',
}));

function AnimatedBackground() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 overflow-hidden"
      style={{ zIndex: 0 }}
    >
      {/* Gradient mesh base */}
      <div className="absolute inset-0"
        style={{
          background: 'radial-gradient(ellipse 80% 60% at 50% 0%, rgba(6,182,212,0.07) 0%, transparent 70%), radial-gradient(ellipse 60% 50% at 80% 100%, rgba(99,102,241,0.06) 0%, transparent 60%)',
        }}
      />
      {/* Floating particles */}
      {PARTICLES.map(p => (
        <div
          key={p.id}
          style={{
            position: 'absolute',
            left: `${p.left}%`,
            top: `${p.top}%`,
            width: p.size,
            height: p.size,
            borderRadius: '50%',
            background: p.color,
            opacity: p.opacity,
            animation: `float-particle ${p.dur}s ${p.delay}s ease-in-out infinite alternate`,
            boxShadow: `0 0 ${p.size * 3}px ${p.color}`,
          }}
        />
      ))}
      {/* Slow-moving grid lines */}
      <div className="absolute inset-0" style={{
        backgroundImage: 'linear-gradient(rgba(34,211,238,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.03) 1px, transparent 1px)',
        backgroundSize: '60px 60px',
        animation: 'grid-drift 30s linear infinite',
      }} />
    </div>
  );
}

export function App() {
  const [activeTab, setActiveTab] = useState<'voice' | 'knowledge' | 'debugger' | 'observability' | 'demo'>('voice');
  const [chunkingStrategy, setChunkingStrategy] = useState<string>('recursive');
  const [lastQueryResponse, setLastQueryResponse] = useState<QueryResponse | null>(null);

  return (
    <div className="min-h-screen flex flex-col bg-[#060a12] text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200" style={{ position: 'relative' }}>
      <AnimatedBackground />
      <div style={{ position: 'relative', zIndex: 10, display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* Top Futuristic Navigation Header */}
      <header className="sticky top-0 z-50 glass-panel border-b border-cyan-500/20 px-4 py-3">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          {/* Logo & Tagline */}
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 text-black shadow-lg cyan-border-glow">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h1 className="font-extrabold text-base tracking-wide text-slate-100 flex items-center space-x-2">
                <span>VOICE RAG AI</span>
                <span className="text-xs bg-cyan-500/10 text-cyan-400 border border-cyan-400/30 px-2 py-0.5 rounded-full font-mono">
                  Goa 2026 #Task2
                </span>
              </h1>
              <p className="text-[11px] text-slate-400 font-mono">End-to-End Grounded Voice Knowledge System</p>
            </div>
          </div>

          {/* Nav Tabs */}
          <nav className="flex items-center space-x-1 bg-slate-950/80 p-1 rounded-xl border border-slate-800 text-xs font-medium">
            <button
              onClick={() => setActiveTab('voice')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition-all ${
                activeTab === 'voice'
                  ? 'bg-cyan-500 text-black font-semibold shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Mic className="w-3.5 h-3.5" />
              <span>Voice Assistant</span>
            </button>

            <button
              onClick={() => setActiveTab('knowledge')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition-all ${
                activeTab === 'knowledge'
                  ? 'bg-cyan-500 text-black font-semibold shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Knowledge Base</span>
            </button>

            <button
              onClick={() => setActiveTab('debugger')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition-all ${
                activeTab === 'debugger'
                  ? 'bg-cyan-500 text-black font-semibold shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Terminal className="w-3.5 h-3.5" />
              <span>Debugger</span>
            </button>

            <button
              onClick={() => setActiveTab('observability')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition-all ${
                activeTab === 'observability'
                  ? 'bg-cyan-500 text-black font-semibold shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>Observability</span>
            </button>

            <button
              onClick={() => setActiveTab('demo')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition-all ${
                activeTab === 'demo'
                  ? 'bg-gradient-to-r from-purple-500 to-cyan-500 text-white font-semibold shadow-md'
                  : 'text-purple-400 hover:text-purple-300'
              }`}
            >
              <Play className="w-3.5 h-3.5" />
              <span>Demo Mode</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Main View Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6">
        {activeTab === 'voice' && (
          <VoiceAssistant
            chunkingStrategy={chunkingStrategy}
            onQueryComplete={(res) => setLastQueryResponse(res)}
          />
        )}

        {activeTab === 'knowledge' && (
          <KnowledgeBase
            currentStrategy={chunkingStrategy}
            onStrategyChange={(s) => setChunkingStrategy(s)}
          />
        )}

        {activeTab === 'debugger' && (
          <RetrievalDebugger lastResponse={lastQueryResponse} />
        )}

        {activeTab === 'observability' && (
          <AdminObservability />
        )}

        {activeTab === 'demo' && (
          <DemoMode
            onExecuteDemoQuery={(q) => {
              setActiveTab('voice');
            }}
            onNavigateTab={(tab) => setActiveTab(tab)}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-4 px-4 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Hacker House Goa 2026 • Task #2 Voice-Enabled RAG System</span>
          <div className="flex items-center space-x-2 text-cyan-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
            <span>Grounding Guardrails Active</span>
          </div>
        </div>
      </footer>
      </div>{/* end z-10 wrapper */}
    </div>
  );
}
export default App;
