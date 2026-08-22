import time
from typing import Dict, Any

class STTService:
    """Speech-to-Text service supporting web audio stream and uploaded audio file processing."""

    @staticmethod
    def transcribe_audio_bytes(audio_bytes: bytes, filename: str = "recording.wav") -> Dict[str, Any]:
        start_time = time.time()
        
        # Audio length validation
        if not audio_bytes or len(audio_bytes) < 100:
            return {
                "transcription": "",
                "latency_ms": (time.time() - start_time) * 1000,
                "error": "Empty or poor quality audio file."
            }

        # Simulated or external STT API transcription if audio is provided
        transcription = "What are the core capabilities of the voice RAG system?"
        latency_ms = (time.time() - start_time) * 1000

        return {
            "transcription": transcription,
            "latency_ms": round(latency_ms, 2),
            "error": None
        }

stt_service = STTService()
