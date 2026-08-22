import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Volume2, Send, Zap, ShieldCheck, FileText, AlertTriangle, Database, Globe, CheckSquare } from 'lucide-react';
import { Orb3D } from './Orb3D';
import { sendRagQuery, QueryResponse, MSMARCOSource } from '../services/api';

interface VoiceAssistantProps {
  chunkingStrategy: string;
  onQueryComplete?: (response: QueryResponse) => void;
}

export const VoiceAssistant: React.FC<VoiceAssistantProps> = ({ chunkingStrategy, onQueryComplete }) => {
  const [pipelineState, setPipelineState] = useState<'IDLE' | 'LISTENING' | 'PROCESSING' | 'ANSWERING' | 'ERROR'>('IDLE');
  const [transcript, setTranscript] = useState('');
  const [textInput, setTextInput] = useState('');
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [micPermissionError, setMicPermissionError] = useState<string | null>(null);
  const [audioAmplitude, setAudioAmplitude] = useState(0);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const recognitionRef = useRef<any>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const sttStartTimeRef = useRef<number>(0);
  // Ref mirror of transcript — avoids stale closure in recognition.onend
  const transcriptRef = useRef<string>('');

  // Speech Recognition Initialization
  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        sttStartTimeRef.current = performance.now();
        setPipelineState('LISTENING');
        setMicPermissionError(null);
        startAudioAnalysis();
      };

      recognition.onresult = (event: any) => {
        let currentTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript;
        }
        // Normalise: trim + collapse internal whitespace (same as typed input)
        const normalised = currentTranscript.trim().replace(/\s+/g, ' ');
        transcriptRef.current = normalised;   // always up-to-date, no closure issue
        setTranscript(normalised);
      };

      recognition.onerror = (event: any) => {
        stopAudioAnalysis();
        if (event.error === 'not-allowed') {
          setMicPermissionError('Microphone permission was denied. Please grant microphone access in browser settings.');
        } else {
          setMicPermissionError(`Speech recognition error: ${event.error}`);
        }
        setPipelineState('ERROR');
      };

      recognition.onend = () => {
        stopAudioAnalysis();
        const sttLatency = performance.now() - sttStartTimeRef.current;
        // Use ref — NOT transcript state — to avoid stale closure reading old value
        const finalText = transcriptRef.current.trim();
        if (finalText) {
          handleExecuteQuery(finalText, sttLatency);
        } else {
          setPipelineState('IDLE');
        }
      };

      recognitionRef.current = recognition;
    } else {
      setMicPermissionError('Web Speech API is not natively supported in this browser. Please use text input or Chrome/Edge.');
    }

    return () => {
      stopAudioAnalysis();
    };
  }, []);  // run once — onend uses ref, not state closure

  // Audio Amplitude Analyzer
  const startAudioAnalysis = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStreamRef.current = stream;
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioCtxRef.current = audioCtx;
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const updateAmplitude = () => {
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          sum += dataArray[i];
        }
        const avg = sum / bufferLength;
        setAudioAmplitude(avg / 255);
        animFrameRef.current = requestAnimationFrame(updateAmplitude);
      };

      updateAmplitude();
    } catch (err) {
      console.warn("Audio analysis stream access issue", err);
    }
  };

  const stopAudioAnalysis = () => {
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(track => track.stop());
      micStreamRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    setAudioAmplitude(0);
  };

  const toggleListening = () => {
    if (pipelineState === 'LISTENING') {
      recognitionRef.current?.stop();
    } else {
      setTranscript('');
      setResponse(null);
      setPipelineState('LISTENING');
      try {
        recognitionRef.current?.start();
      } catch (err) {
        startAudioAnalysis();
      }
    }
  };

  const handleExecuteQuery = async (queryText: string, sttMs: number = 0) => {
    if (!queryText.trim()) return;
    setResponse(null);           // clear stale answer immediately
    setPipelineState('PROCESSING');

    try {
      const res = await sendRagQuery(queryText, chunkingStrategy, 4, sttMs);
      setResponse(res);
      setPipelineState('ANSWERING');

      if (onQueryComplete) {
        onQueryComplete(res);
      }

      // Auto Text-To-Speech Playback
      speakText(res.answer);
    } catch (err) {
      setPipelineState('ERROR');
    }
  };

  const speakText = (text: string) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);
      window.speechSynthesis.speak(utterance);
    }
  };

  const isMSMARCOSource = response?.knowledge_source === 'MSMARCO-XI' || (response?.msmarco_sources && response.msmarco_sources.length > 0);

  return (
    <div className="flex flex-col items-center space-y-6 max-w-4xl mx-auto px-4 py-2">
      {/* Knowledge Source Badge */}
      <div className="flex items-center space-x-2 w-full justify-center">
        <div className={`flex items-center space-x-2 px-4 py-1.5 rounded-full border text-xs font-mono font-semibold transition-all ${
          isMSMARCOSource
            ? 'bg-emerald-500/10 border-emerald-400/40 text-emerald-300'
            : 'bg-slate-800/60 border-slate-700 text-slate-400'
        }`}>
          <Database className="w-3.5 h-3.5" />
          <span>Knowledge Source: {isMSMARCOSource ? 'MSMARCO-XI' : 'Document Corpus'}</span>
          {isMSMARCOSource && (
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping ml-1"></span>
          )}
        </div>
      </div>

      {/* 3D Knowledge Orb Centerpiece */}
      <div className="relative">
        <Orb3D state={pipelineState} audioAmplitude={audioAmplitude} />
        {pipelineState === 'LISTENING' && (
          <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 text-xs font-mono animate-pulse">
            LISTENING...
          </div>
        )}
      </div>

      {/* Futuristic Audio Visualizer Waveform Bar */}
      {pipelineState === 'LISTENING' && (
        <div className="flex items-center space-x-1 h-6">
          {[...Array(12)].map((_, i) => (
            <div
              key={i}
              className="w-1 bg-cyan-400 rounded-full transition-all duration-75"
              style={{
                height: `${Math.max(4, Math.sin(i + audioAmplitude * 10) * audioAmplitude * 24)}px`,
              }}
            />
          ))}
        </div>
      )}

      {/* Main Microphone Action Trigger */}
      <div className="flex flex-col items-center space-y-2">
        <button
          onClick={toggleListening}
          className={`relative p-6 rounded-full transition-all duration-300 transform hover:scale-105 active:scale-95 shadow-2xl ${
            pipelineState === 'LISTENING'
              ? 'bg-cyan-500 text-black'
              : 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:from-cyan-400 hover:to-blue-500'
          }`}
          title="Click to speak"
          style={pipelineState !== 'LISTENING' ? {
            boxShadow: '0 0 0 0 rgba(34,211,238,0.4)',
            animation: 'mic-idle-glow 2.5s ease-in-out infinite',
          } : {}}
        >
          {/* Listening ripple rings */}
          {pipelineState === 'LISTENING' && (
            <>
              <span className="absolute inset-0 rounded-full bg-cyan-400/30 animate-ping" style={{ animationDuration: '0.9s' }} />
              <span className="absolute -inset-2 rounded-full bg-cyan-400/15 animate-ping" style={{ animationDuration: '1.3s', animationDelay: '0.2s' }} />
              <span className="absolute -inset-4 rounded-full bg-cyan-400/08 animate-ping" style={{ animationDuration: '1.8s', animationDelay: '0.4s' }} />
            </>
          )}
          {pipelineState === 'LISTENING' ? <MicOff className="w-8 h-8 relative z-10" /> : <Mic className="w-8 h-8 relative z-10" />}
        </button>
        <span className="text-xs text-slate-400 font-mono tracking-wide">
          {pipelineState === 'LISTENING' ? 'Tap to Stop' : 'Hold / Click Mic to Speak'}
        </span>
      </div>

      {/* Permission Error Banner */}
      {micPermissionError && (
        <div className="w-full glass-panel border-amber-500/40 text-amber-300 p-3 rounded-lg flex items-center space-x-3 text-sm">
          <AlertTriangle className="w-5 h-5 flex-shrink-0 text-amber-400" />
          <span>{micPermissionError}</span>
        </div>
      )}

      {/* Live STT Transcription Display */}
      {transcript && (
        <div className="w-full rounded-xl p-4 text-center space-y-1 border"
          style={{
            background: 'rgba(14,22,40,0.75)',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            borderColor: 'rgba(34,211,238,0.25)',
            boxShadow: '0 0 18px -4px rgba(34,211,238,0.15), inset 0 1px 0 rgba(34,211,238,0.08)',
          }}
        >
          <span className="text-[10px] text-cyan-400 font-mono uppercase tracking-[0.15em] font-semibold">You Said</span>
          <p className="text-slate-100 text-lg font-medium italic leading-snug">"{transcript}"</p>
        </div>
      )}

      {/* Fallback Text Input (For Testing/Debugging) */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (textInput) {
            handleExecuteQuery(textInput);
            setTranscript(textInput);
            setTextInput('');
          }
        }}
        className="w-full max-w-xl flex items-center space-x-2"
      >
        <input
          type="text"
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          placeholder="Or type a test query here..."
          className="flex-1 bg-slate-900/80 border border-slate-700/60 rounded-xl px-4 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-all"
        />
        <button
          type="submit"
          disabled={!textInput.trim() || pipelineState === 'PROCESSING'}
          className="bg-slate-800 hover:bg-slate-700 text-cyan-400 p-2.5 rounded-xl border border-slate-700 disabled:opacity-50 transition-all"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>

      {/* Processing skeleton — shown while awaiting backend response */}
      {pipelineState === 'PROCESSING' && !response && (
        <div
          className="w-full rounded-2xl p-6 space-y-4"
          style={{
            background: 'rgba(10,18,36,0.82)',
            backdropFilter: 'blur(24px)',
            border: '1px solid rgba(34,211,238,0.12)',
            animation: 'response-enter 0.2s ease both',
          }}
        >
          <div className="flex items-center space-x-3">
            {/* Spinning indicator */}
            <svg className="animate-spin w-4 h-4 text-cyan-400 flex-shrink-0" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
            </svg>
            <span className="text-xs font-mono text-cyan-400 tracking-widest uppercase">Processing your query…</span>
          </div>
          {/* Shimmer bars */}
          {[70, 90, 55].map((w, i) => (
            <div key={i} className="h-3 rounded-full bg-slate-800/80 overflow-hidden" style={{ width: `${w}%` }}>
              <div className="h-full rounded-full" style={{
                background: 'linear-gradient(90deg, transparent 0%, rgba(34,211,238,0.15) 50%, transparent 100%)',
                animation: `shimmer 1.4s ${i * 0.18}s ease-in-out infinite`,
                backgroundSize: '200% 100%',
              }} />
            </div>
          ))}
        </div>
      )}

      {/* Response Panel */}
      {response && (
        <div
          className="w-full rounded-2xl p-6 space-y-5"
          style={{
            background: 'rgba(10,18,36,0.82)',
            backdropFilter: 'blur(24px)',
            WebkitBackdropFilter: 'blur(24px)',
            border: '1px solid rgba(34,211,238,0.18)',
            boxShadow: response.can_answer
              ? '0 0 32px -8px rgba(16,185,129,0.18), 0 8px 40px rgba(0,0,0,0.5)'
              : '0 0 32px -8px rgba(245,158,11,0.12), 0 8px 40px rgba(0,0,0,0.5)',
            animation: 'response-enter 0.35s cubic-bezier(0.16,1,0.3,1) both',
          }}
        >
          {/* Header row */}
          <div className="flex items-center justify-between pb-3"
            style={{ borderBottom: '1px solid rgba(51,65,85,0.6)' }}
          >
            <div className="flex items-center space-x-2.5">
              {/* Left accent bar */}
              <div className={`w-1 h-5 rounded-full ${
                response.can_answer ? 'bg-emerald-400' : 'bg-amber-400'
              }`}
                style={{ boxShadow: response.can_answer ? '0 0 8px rgba(52,211,153,0.7)' : '0 0 8px rgba(251,191,36,0.6)' }}
              />
              <ShieldCheck className={`w-4.5 h-4.5 ${
                response.can_answer ? 'text-emerald-400' : 'text-amber-400'
              }`} />
              <span className="font-bold text-slate-100 text-sm tracking-wide uppercase"
                style={{ letterSpacing: '0.06em' }}
              >Grounded AI Answer</span>
              {/* Guardrail status badge with glow */}
              {response.debug?.guardrail_status && (
                <span
                  className={`text-[10px] font-mono px-2.5 py-0.5 rounded-full border ${
                    response.debug.guardrail_status === 'PASSED' || response.debug.guardrail_status === 'PASSED_STRICT_REGEN'
                      ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300'
                      : response.debug.guardrail_status === 'REJECTED_UNSAFE_CONTENT'
                      ? 'bg-red-500/10 border-red-500/40 text-red-300'
                      : 'bg-amber-500/10 border-amber-500/40 text-amber-300'
                  }`}
                  style={{
                    boxShadow: response.debug.guardrail_status === 'PASSED' || response.debug.guardrail_status === 'PASSED_STRICT_REGEN'
                      ? '0 0 10px rgba(52,211,153,0.25)'
                      : response.debug.guardrail_status === 'REJECTED_UNSAFE_CONTENT'
                      ? '0 0 10px rgba(239,68,68,0.25)'
                      : '0 0 10px rgba(251,191,36,0.2)',
                  }}
                >
                  {response.debug.guardrail_status}
                </span>
              )}
            </div>

            <div className="flex items-center space-x-3 text-xs font-mono">
              {/* Confidence badge */}
              <span
                className={`px-2.5 py-1 rounded-full border font-semibold ${
                  response.can_answer ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300' : 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                }`}
                style={{
                  boxShadow: response.can_answer
                    ? '0 0 12px rgba(52,211,153,0.2)'
                    : '0 0 12px rgba(251,191,36,0.15)',
                }}
              >
                {Math.round(response.confidence * 100)}% confidence
              </span>
              <button
                onClick={() => speakText(response.answer)}
                className="flex items-center space-x-1 text-cyan-400 hover:text-cyan-300 transition-colors"
              >
                <Volume2 className={`w-4 h-4 ${isSpeaking ? 'animate-pulse text-cyan-300' : ''}`} />
                <span>TTS</span>
              </button>
            </div>
          </div>

          {/* Answer Text */}
          <p className="text-slate-100 text-[15px] leading-[1.75] font-normal tracking-[0.01em]">
            {response.answer}
          </p>

          {/* Gradient separator */}
          {response.can_answer && response.msmarco_sources && response.msmarco_sources.length > 0 && (
            <div style={{ height: 1, background: 'linear-gradient(90deg, transparent, rgba(34,211,238,0.2) 30%, rgba(99,102,241,0.2) 70%, transparent)' }} />
          )}

          {/* MSMARCO-XI Source Citations — only shown when answer is grounded */}
          {response.can_answer && response.msmarco_sources && response.msmarco_sources.length > 0 && (
            <div className="space-y-3">
              <span className="text-[10px] text-emerald-400 font-mono flex items-center space-x-1.5 font-bold uppercase tracking-[0.12em]">
                <Database className="w-3.5 h-3.5" />
                <span>MSMARCO-XI Source Records</span>
              </span>
              <div className="space-y-2">
                {response.msmarco_sources.map((src: MSMARCOSource, i: number) => (
                  <div
                    key={i}
                    className="rounded-xl p-3 text-xs font-mono space-y-1.5 border transition-all duration-200"
                    style={{
                      background: 'rgba(5,14,28,0.7)',
                      borderColor: 'rgba(52,211,153,0.18)',
                      backdropFilter: 'blur(8px)',
                      boxShadow: '0 2px 12px rgba(0,0,0,0.3)',
                    }}
                    onMouseEnter={e => (e.currentTarget.style.boxShadow = '0 0 18px -4px rgba(52,211,153,0.25), 0 2px 12px rgba(0,0,0,0.3)')}
                    onMouseLeave={e => (e.currentTarget.style.boxShadow = '0 2px 12px rgba(0,0,0,0.3)')}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-emerald-300 font-bold">Record ID: {src.query_id}</span>
                      <div className="flex items-center space-x-2">
                        {src.is_selected === 1 && (
                          <span className="flex items-center space-x-1 text-cyan-400 bg-cyan-500/10 border border-cyan-500/30 px-2 py-0.5 rounded-full">
                            <CheckSquare className="w-3 h-3" />
                            <span>Selected</span>
                          </span>
                        )}
                        <span className="flex items-center space-x-1 text-slate-400">
                          <Globe className="w-3 h-3" />
                          <span>{src.language_code}</span>
                        </span>
                      </div>
                    </div>
                    {src.eng_query && (
                      <div className="text-slate-400">
                        <span className="text-slate-500">Original Query: </span>
                        <span className="text-slate-300 italic">"{src.eng_query.substring(0, 80)}{src.eng_query.length > 80 ? '...' : ''}"</span>
                      </div>
                    )}
                    <div className="flex items-center space-x-3 text-[11px] text-slate-500">
                      <span>Passage #{src.passage_index}</span>
                      <span>•</span>
                      <span>Sim: {src.similarity_score.toFixed(3)}</span>
                      <span>•</span>
                      <span>Rerank: {src.rerank_score.toFixed(3)}</span>
                      <span>•</span>
                      <span className="uppercase">{src.chunking_strategy}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Standard Source Citations */}
          {response.sources.length > 0 && (
            <div className="space-y-2 pt-2 border-t border-slate-800/80">
              <span className="text-xs text-slate-400 font-mono flex items-center space-x-1">
                <FileText className="w-3.5 h-3.5 text-cyan-400" />
                <span>Grounding Evidence Sources:</span>
              </span>
              <div className="flex flex-wrap gap-2">
                {response.sources.map((src, i) => (
                  <span
                    key={i}
                    className="bg-slate-900/90 text-cyan-300 border border-cyan-500/20 text-xs px-2.5 py-1 rounded-md font-mono"
                  >
                    {src}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Latency Timing Breakdown */}
          <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800/60 flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
            <div className="flex items-center space-x-1 text-cyan-400">
              <Zap className="w-4 h-4" />
              <span className="font-semibold text-slate-300">Total Latency:</span>
              <span className="text-cyan-300 font-bold">{response.latency.total_ms} ms</span>
            </div>

            <div className="flex items-center space-x-3 text-slate-400 text-[11px]">
              <span>STT: {response.latency.stt_ms}ms</span>
              <span>•</span>
              <span>Emb: {response.latency.embedding_ms}ms</span>
              <span>•</span>
              <span>Ret: {response.latency.retrieval_ms}ms</span>
              <span>•</span>
              <span>Rerank: {response.latency.reranking_ms}ms</span>
              <span>•</span>
              <span>LLM: {response.latency.llm_ms}ms</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
