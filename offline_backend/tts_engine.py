"""
Neural Text-to-Speech Engine for Socratic Buddy using Microsoft Edge Neural Voices (edge-tts).
Provides free, high-fidelity, studio-quality natural human speech with no API keys required.
"""

import re
import logging
from typing import Optional
import edge_tts

logger = logging.getLogger("socratic.tts")

# Natural neural voices categorized by persona
VOICE_PRESETS = {
    "child_warm": "en-US-AnaNeural",      # Gentle, warm child/young companion
    "teen_peer": "en-US-GuyNeural",       # Natural conversational peer tone
    "guardian_mentor": "en-US-JennyNeural", # Compassionate, mature guardian tone
    "counselor_clear": "en-US-AriaNeural",  # Clear, empathetic reflective voice
}


def clean_text_for_speech(text: str) -> str:
    """Strips JSON fragments, code fences, and markdown formatting for clean vocal output."""
    if not text:
        return ""
    
    # Strip JSON braces / payloads
    cleaned = re.sub(r"\{[\s\S]*?\}", "", text)
    # Strip code blocks
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    # Strip markdown headers, bold, italics, backticks
    cleaned = re.sub(r"[*_#`~>\[\]]", " ", cleaned)
    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def get_voice_for_age(child_age: int, requested_voice: Optional[str] = None) -> str:
    """Selects the optimal neural voice persona based on child age."""
    if requested_voice and requested_voice.strip():
        return requested_voice.strip()
    
    if child_age <= 10:
        return VOICE_PRESETS["child_warm"]  # en-US-AnaNeural
    else:
        return VOICE_PRESETS["teen_peer"]   # en-US-GuyNeural


async def synthesize_speech(
    text: str,
    child_age: int = 10,
    voice: Optional[str] = None,
    rate: str = "+0%",
    pitch: str = "+0Hz"
) -> bytes:
    """
    Synthesizes natural speech audio using Edge Neural TTS.
    Returns raw MP3 audio bytes.
    """
    clean_text = clean_text_for_speech(text)
    if not clean_text:
        return b""

    selected_voice = get_voice_for_age(child_age, voice)
    logger.info(f"Synthesizing neural speech with voice='{selected_voice}' for text length={len(clean_text)}")

    try:
        communicate = edge_tts.Communicate(
            text=clean_text,
            voice=selected_voice,
            rate=rate,
            pitch=pitch
        )
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        
        return b"".join(audio_chunks)
    except Exception as e:
        logger.error(f"edge-tts synthesis failed: {e}", exc_info=True)
        raise RuntimeError(f"Neural TTS synthesis error: {str(e)}")
