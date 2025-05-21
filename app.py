import streamlit as st
import openai
import os
import json
import pathlib
import time
import sqlite3
import datetime
import base64
from gtts import gTTS

# ── DATABASE SETUP ─────────────────────────────────────
DB_PATH = pathlib.Path(__file__).parent / "leaderboard.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS leaderboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    score INTEGER,
    timestamp TEXT
)
""")
conn.commit()

# ── SALES PILLARS & SCORING ─────────────────────────────
PILLARS = {
    "rapport": ["i understand", "great question", "thank you for sharing", "appreciate"],
    "pain":    ["challenge", "issue", "pain point", "concern", "problem"],
    "needs":   ["what system", "how much time", "are you confident", "success look", "goals"],
    "teach":   ["did you know", "we've seen", "benchmark", "tailor", "in our experience"],
    "close":   ["demo", "free trial", "does this sound", "next step", "move forward"]
}

FEEDBACK_HINTS = {
    "rapport": "Work on building rapport: show empathy, thank them, mirror language.",
    "pain":    "Ask more about challenges or frustrations to uncover deeper pain points.",
    "needs":   "Missed discovery—ask what success looks like and usage details.",
    "teach":   "Provide a quick insight or customer story that reframes their thinking.",
    "close":   "Include a clear next step like a demo, trial, or proposal to close."
}

COMPLIMENTS = {
    "rapport": "Great rapport building—your tone is warm and engaging.",
    "pain":    "Excellent uncovering of core challenges.",
    "needs":   "Strong discovery questions—nicely done.",
    "teach":   "You provided valuable insights that reframed the problem.",
    "close":   "Strong close! You confidently moved to next steps."
}

DEAL_OBJECTIONS = ["budget", "timing", "vendor", "implementation", "support", "approval"]

def calc_score(msgs):
    counts = {p: 0 for p in PILLARS}
    conversation = []
    for m in msgs:
        if m["role"] != "user": continue
        txt = m["content"].lower()
        conversation.append(txt)
        for p, kws in PILLARS.items():
            if any(k in txt for k in kws):
                counts[p] += 1

    # cap each at 3 occurrences → 20 points each
    sub_scores = {p: min(v,3)*(20/3) for p,v in counts.items()}
    total = int(sum(sub_scores.values()))

    # brief feedback lines
    brief_fb = [
        f"{'✅' if sub_scores[p]>=10 else '⚠️'} {p.title()} {int(sub_scores[p])}/20"
        for p in PILLARS
    ]

    # detailed feedback
    details = []
    for p in PILLARS:
        pts = sub_scores[p]
        if pts >= 15:
            details.append(f"**{p.title()}**: {COMPLIMENTS[p]}")
        else:
            details.append(f"**{p.title()}**: {FEEDBACK_HINTS[p]}")

    # objections coverage
    convo = " ".join(conversation)
    found = [o for o in DEAL_OBJECTIONS if o in convo]
    missed = [o for o in DEAL_OBJECTIONS if o not in convo]
    obj_summary = (
        f"**Objections uncovered:** {', '.join(found) if found else 'None'}\n"
        f"**Objections missed:** {', '.join(missed) if missed else 'None'}"
    )
    feedback_detail = "\n\n".join(details + [obj_summary])

    return total, "\n".join(brief_fb), sub_scores, feedback_detail

def generate_follow_up_narrative(sub_scores, scenario, persona):
    name = persona["persona_name"]
    company = scenario["prospect"]
    total = int(sum(sub_scores.values()))
    close_pts = sub_scores.get("close",0)
    rapport_pts = sub_scores.get("rapport",0)
    pain_pts = sub_scores.get("pain",0)

    if total >= 75 and close_pts >= 10:
        return (
            f"You and {name} agreed to move forward and review a proposal. "
            f"{name} was enthusiastic and remembered your key insights. "
            f"The solution was accepted, and {company} is now a successful client."
        )
    elif total >= 50 and close_pts >= 5:
        return (
            f"You left a strong impression. {name} requested a detailed pricing breakdown "
            f"and scheduled a follow-up meeting next week to finalize details."
        )
    elif total >= 35 and rapport_pts >= 10 and pain_pts >= 5:
        return (
            f"You followed up via email and received a brief reply. "
            f"{name} said they’re reviewing internally and may reconnect later this month. "
            f"The opportunity remains open but requires persistence."
        )
    else:
        return (
            f"You left a voicemail and sent a follow-up email, but didn’t hear back. "
            f"After two weeks of silence, it’s safe to assume {name} has moved on. "
            "This opportunity is marked as lost."
        )

# ── TIMER HELPERS ─────────────────────────────────────
def init_timer():
    if "start" not in st.session_state:
        st.session_state.start = time.time()
        st.session_state.cut = False

def show_timer(window):
    elapsed = (time.time() - st.session_state.start) / 60
    remaining = max(0, window - elapsed)
    st.sidebar.markdown("### ⏱️ Time Remaining")
    if remaining <= 1:
        st.sidebar.warning("⚠️ Less than 1 minute remaining!")
    elif remaining <= 3:
        st.sidebar.info(f"⏳ {int(remaining)} minutes left")
    else:
        st.sidebar.write(f"{int(remaining)} minutes remaining")
    return elapsed >= window

# ── OPENAI CLIENT ─────────────────────────────────────
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY missing")
    st.stop()
client = openai.OpenAI(api_key=api_key)

# ── LOAD ARCpoint SCENARIOS ───────────────────────────
DATA = pathlib.Path(__file__).parent / "data" / "arcpoint_scenarios.json"
SCENARIOS = json.loads(DATA.read_text())

# ── PAGE SETUP ────────────────────────────────────────
st.set_page_config(page_title="ARCpoint Sales Trainer", page_icon="💬")
st.title("💬 ARCpoint Sales Training Chatbot")

# ── DOWNLOAD PLAYBOOK BUTTON ──────────────────────────
pdf = pathlib.Path(__file__).parent / "TPA Solutions Play Book.pdf"
if pdf.exists():
    b64 = base64.b64encode(pdf.read_bytes()).decode()
    href = f"data:application/pdf;base64,{b64}"
    st.sidebar.markdown(
        f'<a href="{href}" download="ARCpoint_Sales_Playbook.pdf" style="text-decoration:none">'
        f'<div style="background:#d32f2f;padding:8px;border-radius:4px;text-align:center;color:white">'
        f'Download Sales Playbook</div></a>',
        unsafe_allow_html=True
    )

# ── SCENARIO & PERSONA SELECTION ─────────────────────
names = [f"{s['id']}. {s['prospect']} ({s['category']})" for s in SCENARIOS]
pick = st.sidebar.selectbox("Choose a scenario", names)
scenario = SCENARIOS[names.index(pick)]

# reset on scenario change
if st.session_state.get("scenario") != pick:
    st.session_state.clear()
    st.session_state.scenario = pick

# persona chooser
plist = scenario["decision_makers"]
pidx = st.sidebar.selectbox(
    "Which decision-maker?",
    options=list(range(len(plist))),
    format_func=lambda i: f"{plist[i]['persona_name']} ({plist[i]['persona_role']})",
    index=st.session_state.get("persona_idx", 0)
)
st.session_state.persona_idx = pidx
persona = plist[pidx]

# build system prompt
def build_prompt(scenario, persona):
    time_limit = {"Easy":10,"Medium":15,"Hard":20}[scenario["difficulty"]["level"]]
    others = [p["persona_name"] for p in plist if p!=persona]
    note = f"You know {', '.join(others)} is another stakeholder." if others else ""
    pains = ", ".join(persona["pain_points"])
    return f"""
You are **{persona['persona_name']}**, the **{persona['persona_role']}** at **{scenario['prospect']}**.
• Background: {persona['persona_background']}; Pain points: {pains}.
• Difficulty: {scenario['difficulty']['level']} → {time_limit} minutes.
• {note}

**IMPORTANT:**  
• You are NOT the product expert—do NOT explain drug testing details.  
• Defer specifics back to the rep:  
  “I’m not sure—could you clarify that?”  
  “Can you tell me more about how that works?”

• Speak only as this persona: voice objections, clarifying questions, time pressures, need to check others.  
Stay in character.
""".strip()

prompt = build_prompt(scenario, persona)

# initialize chat & timer
init_timer()
if "msgs" not in st.session_state:
    st.session_state.msgs = [{"role":"system","content":prompt}]

# show persona info
time_limit = {"Easy":10,"Medium":15,"Hard":20}[scenario["difficulty"]["level"]]
st.markdown(f"""
**Persona:** {persona['persona_name']} ({persona['persona_role']})  
**Background:** {persona['persona_background']}  
**Company:** {scenario['prospect']}  
**Time Available:** {time_limit} min  
""")

# ── CHAT LOOP ───────────────────────────────────────────
user_input = st.chat_input("Your message to the prospect")
if user_input and not st.session_state.get("closed", False):
    st.session_state.msgs.append({"role":"user","content":user_input})

    # check time cap & show timer
    end_now = show_timer(time_limit)
    if end_now:
        st.session_state.msgs.append({
            "role":"assistant",
            "content": f"**{persona['persona_name']}**: Sorry, I need another meeting now. Let's continue later."
        })
        st.session_state.closed = True
    else:
        # persona switch on mention
        lower = user_input.lower()
        for idx,p in enumerate(plist):
            if idx!=pidx and p["persona_name"].lower() in lower:
                st.session_state.persona_idx = idx
                persona = plist[idx]
                prompt = build_prompt(scenario, persona)
                st.session_state.msgs[0] = {"role":"system","content":prompt}
                st.session_state.msgs.append({
                    "role":"assistant",
                    "content": f"**{persona['persona_name']} ({persona['persona_role']}) has joined the meeting.**"
                })
                break
        else:
            # call OpenAI
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.msgs
            )
            text = resp.choices[0].message.content.strip()
            st.session_state.msgs.append({"role":"assistant","content":text})

# render chat
for m in st.session_state.msgs[1:]:
    st.chat_message("user" if m["role"]=="user" else "assistant").write(m["content"])
    if m["role"]=="assistant" and st.session_state.get("voice", False):
        gTTS(m["content"]).save("tmp.mp3")
        st.audio(open("tmp.mp3","rb").read(), format="audio/mp3")

# ── SIDEBAR CONTROLS ──────────────────────────────────
voice = st.sidebar.checkbox("🎙️ Voice Mode", key="voice")
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.clear()
    st.rerun()

if st.sidebar.button("🔚 End & Score"):
    if not st.session_state.get("closed", False):
        total, brief_fb, sub_scores, detail_fb = calc_score(st.session_state.msgs)
        st.session_state.closed = True
        st.session_state.score = total
        st.sidebar.success("Scored!")
        st.session_state.brief_fb = brief_fb
        st.session_state.detail_fb = detail_fb
        st.session_state.sub_scores = sub_scores

# after scoring, show breakdown & leaderboard
if st.session_state.get("closed", False):
    # What Happened Next
    narrative = generate_follow_up_narrative(st.session_state.sub_scores, scenario, persona)
    st.sidebar.markdown("### 📘 What Happened Next")
    st.sidebar.write(narrative)

    st.sidebar.markdown("### 🏆 Your Score")
    st.sidebar.write(f"**{st.session_state.score}/100**")
    st.sidebar.markdown("### 📊 Breakdown")
    for p, pts in st.session_state.sub_scores.items():
        st.sidebar.write(f"{p.title()}: {int(pts)}/20")
    st.sidebar.markdown("### 📣 Recommendations")
    st.sidebar.write(st.session_state.detail_fb)

    name = st.sidebar.text_input("Your name:", key="name_input")
    if st.sidebar.button("🏅 Save to Leaderboard") and name:
        cur.execute(
            "INSERT INTO leaderboard(name,score,timestamp) VALUES(?,?,?)",
            (name, st.session_state.score, datetime.datetime.now().isoformat())
        )
        conn.commit()
    st.sidebar.markdown("### 🥇 Top 10 Leaderboard")
    rows = cur.execute(
        "SELECT name,score FROM leaderboard ORDER BY score DESC, timestamp ASC LIMIT 10"
    ).fetchall()
    for i,(n,s) in enumerate(rows, start=1):
        st.sidebar.write(f"{i}. {n} — {s}")
