"""
capstone_streamlit.py — Physics Study Buddy | Streamlit UI
Launch: streamlit run capstone_streamlit.py
"""
import uuid
import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PhysicsBot ⚛️",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg-deep:      #04060f;
    --bg-card:      #080d1a;
    --bg-panel:     #0d1424;
    --accent:       #00d4ff;
    --accent2:      #7c3aed;
    --accent3:      #f59e0b;
    --glow:         rgba(0, 212, 255, 0.15);
    --glow-strong:  rgba(0, 212, 255, 0.35);
    --text-primary: #e8f4fd;
    --text-muted:   #6b8299;
    --border:       rgba(0, 212, 255, 0.12);
    --border-hover: rgba(0, 212, 255, 0.4);
    --font-display: 'Syne', sans-serif;
    --font-mono:    'Space Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: var(--bg-deep) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,212,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(124,58,237,0.06) 0%, transparent 50%) !important;
    font-family: var(--font-mono) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }

.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 900px !important;
}

.pb-header {
    text-align: center;
    padding: 2.5rem 0 2rem;
    position: relative;
}
.pb-header::after {
    content: '';
    display: block;
    width: 200px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    margin: 1.5rem auto 0;
}
.pb-logo {
    font-family: var(--font-display);
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #00d4ff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    line-height: 1;
    margin-bottom: 0.4rem;
}
@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.pb-tagline {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.25em;
    text-transform: uppercase;
}

.welcome-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(124,58,237,0.05));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.welcome-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
}
.welcome-card h4 {
    font-family: var(--font-display);
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0 0 0.4rem 0;
    letter-spacing: 0.05em;
}
.welcome-card p {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.6;
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 1.2rem !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageContent {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(0,212,255,0.04)) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 1rem 1.2rem !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    line-height: 1.7 !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stChatMessageContent {
    background: linear-gradient(135deg, rgba(13,20,36,0.95), rgba(8,13,26,0.95)) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px 16px 16px 16px !important;
    padding: 1.2rem 1.4rem !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    line-height: 1.8 !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 0 0 1px rgba(0,212,255,0.04) !important;
}

[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, var(--accent), #0099bb) !important;
    color: #000 !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, var(--accent2), #5b21b6) !important;
    color: #fff !important;
}

.meta-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid rgba(0,212,255,0.07);
}
.meta-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.05em;
    font-weight: 700;
}
.meta-route   { background: rgba(0,212,255,0.08);  border: 1px solid rgba(0,212,255,0.25); color: var(--accent); }
.meta-faith-pass { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25); color: #10b981; }
.meta-faith-fail { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25); color: var(--accent3); }
.meta-source  { background: rgba(124,58,237,0.08); border: 1px solid rgba(124,58,237,0.25); color: #a78bfa; }

[data-testid="stChatInput"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--border-hover) !important;
    box-shadow: 0 0 0 3px var(--glow) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 12px var(--glow) !important;
    background: rgba(0,212,255,0.04) !important;
}

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1rem 0 !important; }

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    line-height: 1.8 !important;
    color: var(--text-primary) !important;
}
[data-testid="stChatMessage"] strong { color: var(--accent) !important; font-weight: 700 !important; }
[data-testid="stChatMessage"] code {
    background: rgba(0,212,255,0.06) !important;
    border: 1px solid rgba(0,212,255,0.15) !important;
    border-radius: 4px !important;
    padding: 0.1em 0.4em !important;
    color: var(--accent) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.4); }

.session-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.2rem 0.7rem;
    background: rgba(0,212,255,0.04);
    border: 1px solid var(--border);
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--text-muted);
    letter-spacing: 0.06em;
    margin-bottom: 1rem;
}
.session-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #10b981;
    animation: pulse 2s ease infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}

.sidebar-logo     { font-family: 'Syne',sans-serif; font-size: 1.3rem; font-weight: 800; color: var(--accent); letter-spacing: -0.02em; margin-bottom: 0.2rem; }
.sidebar-subtitle { font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-muted); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 1.5rem; }
.sb-label         { font-family: var(--font-mono); font-size: 0.6rem; color: var(--text-muted); letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.6rem; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); }
.topic-item       { display: flex; align-items: center; gap: 0.5rem; padding: 0.32rem 0.5rem; border-radius: 6px; margin-bottom: 0.12rem; font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted); }
.topic-dot        { width: 4px; height: 4px; border-radius: 50%; background: var(--accent); opacity: 0.4; flex-shrink: 0; }
.stat-card        { background: rgba(0,212,255,0.04); border: 1px solid var(--border); border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.4rem; display: flex; justify-content: space-between; align-items: center; }
.stat-label       { font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; }
.stat-value       { font-family: 'Syne',sans-serif; font-size: 1.1rem; font-weight: 700; color: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ── Cache expensive init ───────────────────────────────────────────────────────
@st.cache_resource
def load_app():
    from agent import _get_app
    return _get_app()

app = load_app()

# ── Session state ─────────────────────────────────────────────────────────────
if "thread_id"    not in st.session_state: st.session_state.thread_id    = str(uuid.uuid4())
if "chat"         not in st.session_state: st.session_state.chat         = []
if "messages"     not in st.session_state: st.session_state.messages     = []
if "student_name" not in st.session_state: st.session_state.student_name = None
if "q_count"      not in st.session_state: st.session_state.q_count      = 0
if "faith_scores" not in st.session_state: st.session_state.faith_scores = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚛ PhysicsBot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">B.Tech Study Buddy · 2026</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-label">📚 Knowledge Base</div>', unsafe_allow_html=True)
    for icon, topic in [
        ("⚙️","Newton's Laws"), ("⚡","Work, Energy & Power"), ("🌡️","Thermodynamics"),
        ("🔋","Electrostatics"), ("🔄","Simple Harmonic Motion"), ("💡","Wave Optics"),
        ("🧲","Electromagnetism"), ("⚛️","Quantum Physics"), ("🌀","Rotational Motion"),
        ("🌍","Gravitation"), ("☢️","Nuclear Physics"), ("💧","Fluid Mechanics"),
    ]:
        st.markdown(
            f'<div class="topic-item"><span class="topic-dot"></span>{icon} {topic}</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sb-label">📊 Session Stats</div>', unsafe_allow_html=True)
    avg_faith = (sum(st.session_state.faith_scores) / len(st.session_state.faith_scores)
                 if st.session_state.faith_scores else 0.0)
    faith_color = "#10b981" if avg_faith >= 0.7 else "#f59e0b"
    st.markdown(f"""
        <div class="stat-card">
            <span class="stat-label">Questions</span>
            <span class="stat-value">{st.session_state.q_count}</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Avg Faithfulness</span>
            <span class="stat-value" style="color:{faith_color}">{avg_faith:.2f}</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺  New Conversation"):
        st.session_state.thread_id    = str(uuid.uuid4())
        st.session_state.chat         = []
        st.session_state.messages     = []
        st.session_state.student_name = None
        st.session_state.q_count      = 0
        st.session_state.faith_scores = []
        st.rerun()

    st.markdown(
        '<br><div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;'
        'color:#1e3a5f;text-align:center;">✉ support@physicstudy.edu</div>',
        unsafe_allow_html=True
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pb-header">
    <div class="pb-logo">PhysicsBot</div>
    <div class="pb-tagline">Agentic AI · LangGraph · ChromaDB · Groq</div>
</div>
""", unsafe_allow_html=True)

name_part = f" · {st.session_state.student_name}" if st.session_state.student_name else ""
st.markdown(
    f'<div class="session-pill"><span class="session-dot"></span>'
    f'SESSION {st.session_state.thread_id[:8].upper()}{name_part}</div>',
    unsafe_allow_html=True
)

if not st.session_state.chat:
    st.markdown("""
    <div class="welcome-card">
        <h4>Welcome to PhysicsBot</h4>
        <p>
            Your 24/7 B.Tech Physics Study Buddy — grounded in syllabus knowledge, never hallucinating.<br>
            Ask about any concept from Newton's Laws to Nuclear Physics.<br><br>
            <strong>Tips:</strong>&nbsp; Say <code>My name is ...</code> to personalise &nbsp;·&nbsp;
            Ask <code>What time is it?</code> to test the tool node &nbsp;·&nbsp;
            Try <code>Calculate 2 * 3.14 * 0.5</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Chat history
for item in st.session_state.chat:
    with st.chat_message(item["role"], avatar="🎓" if item["role"] == "user" else "⚛️"):
        st.markdown(item["content"])
        if item["role"] == "assistant" and "meta" in item:
            m = item["meta"]
            route = m.get("route", "")
            faith = m.get("faithfulness", 1.0)
            sources = m.get("sources", [])
            ri = {"retrieve":"📚 RETRIEVE","tool":"🔧 TOOL","memory_only":"💭 MEMORY"}.get(route, f"❓ {route.upper()}")
            fc = "meta-faith-pass" if faith >= 0.7 else "meta-faith-fail"
            fi = "✓" if faith >= 0.7 else "⚠"
            tags = f'<span class="meta-tag meta-route">{ri}</span>'
            tags += f'<span class="meta-tag {fc}">{fi} FAITH {faith:.2f}</span>'
            for s in sources:
                tags += f'<span class="meta-tag meta-source">◈ {s[:22]}</span>'
            st.markdown(f'<div class="meta-row">{tags}</div>', unsafe_allow_html=True)

# Input
name_hint = st.session_state.student_name or "Student"
if question := st.chat_input(f"Ask a physics question, {name_hint}…"):
    with st.chat_message("user", avatar="🎓"):
        st.markdown(question)
    st.session_state.chat.append({"role": "user", "content": question})

    with st.chat_message("assistant", avatar="⚛️"):
        with st.spinner(""):
            from agent import ask
            result = ask(
                question,
                thread_id    = st.session_state.thread_id,
                messages     = st.session_state.messages,
                student_name = st.session_state.student_name,
            )
            answer       = result.get("answer", "Something went wrong. Please try again.")
            route        = result.get("route", "")
            faithfulness = result.get("faithfulness", 1.0)
            sources      = result.get("sources", [])

            st.session_state.messages = result.get("messages", [])
            if result.get("student_name"):
                st.session_state.student_name = result["student_name"]
            st.session_state.q_count += 1
            st.session_state.faith_scores.append(faithfulness)

        st.markdown(answer)

        ri = {"retrieve":"📚 RETRIEVE","tool":"🔧 TOOL","memory_only":"💭 MEMORY"}.get(route, f"❓ {route.upper()}")
        fc = "meta-faith-pass" if faithfulness >= 0.7 else "meta-faith-fail"
        fi = "✓" if faithfulness >= 0.7 else "⚠"
        tags = f'<span class="meta-tag meta-route">{ri}</span>'
        tags += f'<span class="meta-tag {fc}">{fi} FAITH {faithfulness:.2f}</span>'
        for s in sources:
            tags += f'<span class="meta-tag meta-source">◈ {s[:22]}</span>'
        st.markdown(f'<div class="meta-row">{tags}</div>', unsafe_allow_html=True)

    st.session_state.chat.append({
        "role": "assistant", "content": answer,
        "meta": {"route": route, "faithfulness": faithfulness, "sources": sources}
    })
    st.rerun()