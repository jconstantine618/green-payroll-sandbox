# app.py  ──────────────────────────────────────────────────────────────
"""
Green‑Payroll Sales Trainer   (Streamlit + OpenAI + voice I/O)

•  Tick “Speak instead of type” → press Start, talk, Stop, then Send.
•  Assistant can reply with TTS if “Read assistant replies aloud” checked.

Python ≥3.9   •   see requirements.txt
"""
from __future__ import annotations
import os, io, wave, tempfile, pathlib
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ── CONFIG ───────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")          # optional
TMP_DIR        = pathlib.Path(tempfile.gettempdir())
WHISPER_MODEL  = "whisper-1"
GPT_MODEL      = "gpt-4o-mini"
ASSISTANT_VOICE= "elevenlabs/Antoni"

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config("Sales‑Trainer", "🎤", layout="wide")
st.sidebar.title("⚙️ Controls")

# ── SCENARIOS ─────────────────────────────────────────────────────────
SCENARIOS = {
    "Sunshine Daycare Centers (Child‑care)": dict(
        persona    = "Karen Lopez (Owner / Director)",
        background = "Handles HR and scheduling. Wants mobile access.",
        company    = "Sunshine Daycare Centers",
        difficulty = "Easy",
        time       = "10 min",
        prompt     = (
            "You are a payroll SaaS sales rep calling Karen Lopez at "
            "Sunshine Daycare Centers. Ask discovery questions with empathy "
            "and keep responses short."
        ),
    ),
    # …add more scenarios here…
}

# ── TTS UTIL ──────────────────────────────────────────────────────────
def tts_bytes(text: str) -> io.BytesIO | None:
    """Return MP3 bytes via ElevenLabs (preferred) or gTTS fallback."""
    if ELEVEN_API_KEY:
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
            st.warning(f"ElevenLabs TTS failed ({e}); falling back to gTTS…")

    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en", slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        st.warning(f"gTTS failed ({e}). No audio will be played.")
        return None

# ── VOICE IN (WebRTC → Whisper) ───────────────────────────────────────
def record_and_transcribe() -> str | None:
    ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": False, "audio": True},
        sendback_audio=False,
        translations={"select_device": ""},  # hide dropdown label
    )

    if "audio_ready" not in st.session_state:
        st.session_state.audio_ready = False

    def _send_btn_disabled() -> bool:
        return not (ctx and not ctx.state.playing and st.session_state.audio_ready)

    st.button("▶️ Send recording", key="send_audio_btn",
              disabled=_send_btn_disabled())

    # 1️⃣ Recording
    if ctx.state.playing:
        st.session_state.audio_ready = True
        st.info("Recording… press **Stop** when finished, then click **Send**.")
        st.stop()

    # 2️⃣ Waiting for “Send recording”
    if st.session_state.get("send_audio_btn") is False:
        st.stop()

    # 3️⃣ Process
    if ctx.audio_receiver and st.session_state.audio_ready:
        frames = ctx.audio_receiver.get_frames(timeout=1)
        if not frames:
            st.warning("No speech detected. Try again.")
            st.session_state.audio_ready = False
            return None

        samples = np.concatenate([f.to_ndarray().flatten() for f in frames])
        sr = frames[0].sample_rate
        wav_path = TMP_DIR / "speech.wav"
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
            st.session_state.audio_ready = False
            return rsp.text.strip()
        except BadRequestError:
            st.warning("Whisper could not transcribe audio. Try again.")
            st.session_state.audio_ready = False
            return None

# ── GPT CHAT ──────────────────────────────────────────────────────────
def chat(user_msg: str) -> str:
    history = st.session_state.history
    msgs = history + [{"role": "user", "content": user_msg}]
    rsp = client.chat.completions.create(model=GPT_MODEL, messages=msgs)
    assistant_msg = rsp.choices[0].message.content.strip()
    history.extend([{"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}])
    return assistant_msg

# ── STATE INIT ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── SIDEBAR ───────────────────────────────────────────────────────────
scenario_name = st.sidebar.selectbox("Choose a scenario", SCENARIOS.keys())
speak_mode    = st.sidebar.checkbox("🎤 Speak instead of type", value=False)
voice_reply   = st.sidebar.checkbox("🔊 Read assistant replies aloud")
show_debug    = st.sidebar.checkbox("👀 Show debug JSON", value=False)

# ── HEADER ────────────────────────────────────────────────────────────
sc = SCENARIOS[scenario_name]
st.title("💬 Chatbot")
st.markdown(
    f"""
**Persona**: {sc['persona']}  
**Background**: {sc['background']}  
**Company**: {sc['company']}  
**Difficulty**: {sc['difficulty']}  
**Time Available**: {sc['time']}  
"""
)

if not st.session_state.history:
    st.session_state.history.append({"role": "system", "content": sc["prompt"]})

# ── CONVERSATION FLOW ────────────────────────────────────────────────
def handle_turn():
    # ① Get user input
    user_text: str | None
    if speak_mode:
        user_text = record_and_transcribe()
    else:
        user_text = st.chat_input("Your message to the prospect")

    if not user_text:
        return

    # ② Show user bubble
    with st.chat_message("user"):
        st.write(user_text)

    # ③ Assistant response
    assistant_text = chat(user_text)
    with st.chat_message("assistant"):
        st.write(assistant_text)
        if voice_reply:
            audio = tts_bytes(assistant_text)
            if audio:
                st.audio(audio, format="audio/mp3")

handle_turn()

# ── DEBUG (show only if asked) ────────────────────────────────────────
if show_debug:
    with st.expander("📜 Conversation (debug)", expanded=False):
        st.json(st.session_state.history)

# ── CSS HACKS ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Hide the select‑device label reliably */
    div[data-testid="stSelectbox"] label {display:none !important;}

    /* Optional: tighten chat input space on tall monitors */
    .block-container {padding-top: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
