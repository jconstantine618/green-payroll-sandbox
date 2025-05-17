# app.py  ──────────────────────────────────────────────────────────────
"""
Green‑Payroll Sales Trainer   (Streamlit + OpenAI + voice I/O)
"""
from __future__ import annotations
import os, io, wave, tempfile, pathlib, base64, uuid
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ── CONFIG ───────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
ELEVEN_API_KEY  = os.getenv("ELEVEN_API_KEY")          # optional
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # Antoni
TMP_DIR         = pathlib.Path(tempfile.gettempdir())
WHISPER_MODEL   = "whisper-1"
GPT_MODEL       = "gpt-4o-mini"
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
        # 👉 Assistant should BE Karen, not the rep:
        prompt     = (
            "You are **Karen Lopez**, owner/director of Sunshine Daycare Centers. "
            "A payroll SaaS sales rep is calling you.  Answer in first‑person, stay in "
            "character, and keep replies concise and conversational."
        ),
    ),
    # …add more scenarios here…
}

# ── TTS UTILITIES ────────────────────────────────────────────────────
def tts_bytes(text: str) -> io.BytesIO | None:
    """Return an MP3 buffer. Prefers ElevenLabs SDK v1; falls back to gTTS."""
    if ELEVEN_API_KEY:
        try:
            from elevenlabs import VoiceSettings
            from elevenlabs.client import ElevenLabs
            el_client = ElevenLabs(api_key=ELEVEN_API_KEY)
            response  = el_client.text_to_speech.convert(
                text=text,
                voice_id=ELEVEN_VOICE_ID,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
                voice_settings=VoiceSettings(stability=0.35, similarity_boost=0.7),
            )
            audio_bytes = b"".join(response)  # 👈 join the chunk iterator :contentReference[oaicite:0]{index=0}
            return io.BytesIO(audio_bytes)
        except Exception as e:
            st.warning(f"ElevenLabs TTS failed ({e}); falling back to gTTS…")

    # gTTS fallback
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en", slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"gTTS failed ({e}). No audio will be played.")
        return None


def auto_play(audio_buf: io.BytesIO):
    """Autoplay audio via a small <audio> tag."""
    b64 = base64.b64encode(audio_buf.getvalue()).decode()
    uid = uuid.uuid4().hex
    st.markdown(
        f"""
        <audio id="{uid}" autoplay>
          <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,
    )

# ── VOICE‑IN  (WebRTC → Whisper) ─────────────────────────────────────
def record_and_transcribe() -> str | None:
    ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": False, "audio": True},
        sendback_audio=False,
        translations={"select_device": ""},
    )

    if "audio_ready" not in st.session_state:
        st.session_state.audio_ready = False

    def _disabled() -> bool:
        return not (ctx and not ctx.state.playing and st.session_state.audio_ready)

    st.button("▶️ Send recording", key="send_audio_btn", disabled=_disabled())

    if ctx.state.playing:
        st.session_state.audio_ready = True
        st.info("Recording… press **Stop** then **Send**.")
        st.stop()

    if st.session_state.get("send_audio_btn") is False:
        st.stop()

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
    user_text: str | None
    if speak_mode:
        user_text = record_and_transcribe()
    else:
        user_text = st.chat_input("Your message to the prospect")

    if not user_text:
        return

    with st.chat_message("user"):
        st.write(user_text)

    assistant_text = chat(user_text)
    with st.chat_message("assistant"):
        st.write(assistant_text)
        if voice_reply:
            audio = tts_bytes(assistant_text)
            if audio:
                auto_play(audio)

handle_turn()

# ── DEBUG (optional) ─────────────────────────────────────────────────
if show_debug:
    with st.expander("📜 Conversation (debug)", expanded=False):
        st.json(st.session_state.history)

# ── CSS TWEAKS ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      div[data-testid="stSelectbox"] label {display:none !important;}
      .block-container {padding-top: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
