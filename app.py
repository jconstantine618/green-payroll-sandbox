# app.py  ──────────────────────────────────────────────────────────────────
"""
Green-Payroll Sales Trainer — Streamlit demo with:
•   text chat  or
•   voice-in  (WebRTC + Whisper-1)      and optional
•   voice-out (ElevenLabs or gTTS)

Works on Streamlit Community Cloud or any Python ≥3.9 runtime.
"""
from __future__ import annotations

import os, io, pathlib, tempfile, wave
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ─────────────────────────────────────────  CONFIG
load_dotenv()                         # enables .env local dev
st.set_page_config(page_title="Sales Trainer", page_icon="🎤", layout="wide")

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY   = os.getenv("ELEVEN_API_KEY")          # optional
WHISPER_MODEL    = "whisper-1"
GPT_MODEL        = "gpt-4o-mini"                        # or "gpt-3.5-turbo"
ASSISTANT_VOICE  = "elevenlabs/Antoni"                  # or any Eleven voice
TMP_DIR          = pathlib.Path(tempfile.gettempdir())

client = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────  DATA
SCENARIOS = {
    "Sunshine Daycare Centers (Child-care)": dict(
        persona   = "Karen Lopez (Owner / Director)",
        background= "Handles HR and scheduling. Wants mobile access.",
        company   = "Sunshine Daycare Centers",
        difficulty= "Easy",
        time      = "10 minutes",
        prompt    = (
            "You are a payroll SaaS sales rep calling Karen Lopez … "
            "Ask discovery questions with empathy and keep responses short."
        ),
    ),
    # Add more scenarios here …
}

# ─────────────────────────────────────────  UTILITIES
def text_to_speech(text: str) -> io.BytesIO | None:
    """Return audio (mp3) bytes for *text* using ElevenLabs or gTTS fallback."""
    if ELEVEN_API_KEY:                                 # ― ElevenLabs
        try:
            from elevenlabs import generate, VoiceSettings, set_api_key
            set_api_key(ELEVEN_API_KEY)
            audio = generate(
                text=text,
                voice=ASSISTANT_VOICE.split("/")[-1],
                model="eleven_multilingual_v2",
                voice_settings=VoiceSettings(stability=0.35, similarity_boost=0.7),
            )
            return io.BytesIO(audio)
        except Exception as e:
            st.warning(f"ElevenLabs TTS failed ({e}). Falling back to gTTS…")

    # ― gTTS fallback
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        st.warning(f"gTTS failed ({e}). No audio will be played.")
        return None


def record_and_transcribe() -> str | None:
    """
    Capture microphone input via WebRTC, save a clean mono/16-bit WAV,
    send it to Whisper-1 and return the transcript (None if not ready).
    """
    ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": False, "audio": True},
        sendback_audio=False,
    )

    if not ctx.audio_receiver:                         # user hasn’t clicked Start
        return None

    frames = ctx.audio_receiver.get_frames(timeout=2)
    if len(frames) < 20:                               # <≈ ½ s of audio → ignore
        return None

    samples = np.concatenate([f.to_ndarray().flatten() for f in frames])
    sr       = frames[0].sample_rate
    wav_path = TMP_DIR / "speech.wav"

    # write mono/16-bit WAV (Whisper’s favourite format)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())

    try:
        rsp = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=open(wav_path, "rb"),
            response_format="text",
        )
        return rsp.text.strip()
    except BadRequestError:
        st.warning("⚠️ Whisper couldn’t understand. Try speaking a bit longer.")
        return None


def chat_completion(user_text: str, history: list[dict]) -> str:
    """Call the LLM with the running message history."""
    msgs = history + [{"role": "user", "content": user_text}]
    rsp  = client.chat.completions.create(model=GPT_MODEL, messages=msgs)
    return rsp.choices[0].message.content.strip()


# ─────────────────────────────────────────  MAIN UI
if "history" not in st.session_state:
    st.session_state.history = []
if "conversation_running" not in st.session_state:
    st.session_state.conversation_running = False

st.sidebar.title("⚙️ Controls")
scenario_key = st.sidebar.selectbox("Choose a scenario", list(SCENARIOS))
speak_mode   = st.sidebar.checkbox("🎙️ Speak instead of type", value=False)
voice_reply  = st.sidebar.checkbox("🔊 Read assistant replies aloud")

scenario = SCENARIOS[scenario_key]
st.title("💬 Chatbot")
st.markdown(
    f"""
**Persona**: {scenario['persona']}  
**Background**: {scenario['background']}  
**Company**: {scenario['company']}  
**Difficulty**: {scenario['difficulty']}  
**Time Available**: {scenario['time']}  
"""
)

# prompt the assistant at first run
if not st.session_state.history:
    st.session_state.history.append(
        {"role": "system", "content": scenario["prompt"]}
    )

# ───────────────  conversation loop
def handle_user_input():
    if speak_mode:
        user_text = record_and_transcribe()
    else:
        user_text = st.chat_input("Your message to the prospect")

    if not user_text:
        return

    with st.chat_message("user"):
        st.write(user_text)

    assistant_text = chat_completion(user_text, st.session_state.history)
    st.session_state.history.extend(
        [{"role": "user", "content": user_text},
         {"role": "assistant", "content": assistant_text}]
    )

    with st.chat_message("assistant"):
        st.write(assistant_text)
        if voice_reply:
            audio_bytes = text_to_speech(assistant_text)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

# main polling loop ― run each page refresh
handle_user_input()

st.divider()
with st.expander("💾 Conversation (debug)"):
    st.json(st.session_state.history)
