"""
UI Components for AI Pipeline Studio
"""
import streamlit as st
from pathlib import Path

def render_header():
    """Render the main header component"""
    st.markdown("""
    <div class="modern-header">
        <h1>🚀 Automatic Lab Generation Using Multi-Agent RL</h1>
        <p>Transform your ideas into interactive virtual labs with AI</p>
    </div>
    """, unsafe_allow_html=True)

def render_theme_toggle():
    """Render theme toggle button (Python only, no JS)"""
    mode = "🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Light Mode"
    if st.button(f"Toggle Theme ({mode})", key="theme_toggle_btn"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

def render_progress_tracker(progress):
    """Render horizontal progress bar only (no steps)"""
    st.markdown(
        f"""
        <div class="progress-container" style="margin-top:1rem;">
            <div class="progress-bar" style="height:18px;">
                <div class="progress-fill" style="width: {progress}%; height:100%;"></div>
            </div>
            <div style="text-align: center; font-weight: 600; color: var(--text-secondary); margin-top:0.5rem;">
                {progress:.1f}% Complete
            </div>
        </div>
        """, unsafe_allow_html=True
    )

def render_step_tracker(current_step, completed_steps):
    """Render horizontal step tracker as boxes below the title"""
    steps = [
        ("requirements", "📝 Requirements"),
        ("review", "👁️ Review"),
        ("implementation", "🔧 Implementation"),
        ("code", "💻 Code"),
        ("documentation", "📚 Documentation"),
        ("website", "🌐 Website"),
    ]
    box_html = '<div style="display:flex;gap:1rem;justify-content:center;margin:1.5rem 0 1.5rem 0;">'
    for step_key, step_label in steps:
        if completed_steps.get(step_key, False):
            color = "background:linear-gradient(135deg,#38a169,#68d391);color:white;border:2px solid #38a169;"
        elif current_step == step_key:
            color = "background:linear-gradient(135deg,#667eea,#764ba2);color:white;border:2px solid #667eea;"
        else:
            color = "background:#e2e8f0;color:#6c757d;border:2px solid #cbd5e1;"
        box_html += f'<div style="padding:1rem 1.5rem;border-radius:12px;font-weight:700;font-size:1.1rem;{color}min-width:120px;text-align:center;transition:all 0.2s;">{step_label}</div>'
    box_html += '</div>'
    st.markdown(box_html, unsafe_allow_html=True)

def render_model_selection():
    """Render AI model selection with temperature and max tokens, and a button to show model limits"""
    model_options = [
        ("gemini-2.5-flash", "Gemini 2.5 Flash", 10, 250_000, 250),
        ("gemini-2.5-flash-lite-preview-06-17", "Gemini 2.5 Flash-Lite Preview 06-17", 15, 250_000, 1000),
        ("gemini-2.5-flash-preview-tts", "Gemini 2.5 Flash Preview TTS", 3, 10_000, 15),
        ("gemini-2.0-flash", "Gemini 2.0 Flash", 15, 1_000_000, 200),
        ("gemini-2.0-flash-preview-image", "Gemini 2.0 Flash Preview Image Generation", 10, 200_000, 100),
        ("gemini-2.0-flash-lite", "Gemini 2.0 Flash-Lite", 30, 1_000_000, 200),
        ("gemini-1.5-flash", "Gemini 1.5 Flash (Deprecated)", 15, 250_000, 50),
        ("gemini-1.5-flash-8b", "Gemini 1.5 Flash-8B (Deprecated)", 15, 250_000, 50),
        ("gemma-3", "Gemma 3 & 3n", 30, 15_000, 14_400),
        ("gemini-embedding-experimental-03-07", "Gemini Embedding Experimental 03-07", 5, None, 100),
    ]

    st.markdown("#### 🤖 AI Configuration")

    model_labels = [label for _, label, *_ in model_options]
    model_keys = [k for k, *_ in model_options]

    default_idx = model_keys.index(
        st.session_state.get("selected_model", model_keys[0])
    ) if st.session_state.get("selected_model", model_keys[0]) in model_keys else 0

    selected_idx = st.selectbox(
        "Choose AI Model:",
        options=list(range(len(model_keys))),
        format_func=lambda i: model_labels[i],
        index=default_idx,
        key="model_selector"
    )

    selected_model = model_keys[selected_idx]

    temp = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("temperature", 0.1),
        step=0.01,
        key="temperature_slider"
    )

    max_tokens = st.number_input(
        "Max Tokens",
        min_value=1,
        max_value=1000000,
        value=st.session_state.get("max_tokens", 100000),
        step=1,
        key="max_tokens_input"
    )

    # Show Model Limits button (no rerun, just set flag)
    if st.button("📊 Show Model Limits", key="show_model_limits_btn"):
        st.session_state.show_model_limits = True
        # Do NOT call st.rerun() here

    if st.session_state.get("show_model_limits", False):
        # Always expand sidebar when showing model limits
        st.markdown("""
            <script>
            window.parent.document.querySelector('section[data-testid="stSidebar"]').style.transform = "translateX(0%)";
            window.parent.document.querySelector('section[data-testid="stSidebar"]').style.width = "400px";
            </script>
        """, unsafe_allow_html=True)
        with st.sidebar:
            st.markdown("### Gemini Model Limits")
            st.markdown("""
| Model | RPM | TPM | RPD |
|-------|-----|------|------|
| Gemini 2.5 Pro | -- | -- | -- |
| Gemini 2.5 Flash | 10 | 250,000 | 250 |
| Gemini 2.5 Flash-Lite Preview 06-17 | 15 | 250,000 | 1,000 |
| Gemini 2.5 Flash Preview TTS | 3 | 10,000 | 15 |
| Gemini 2.5 Pro Preview TTS | -- | -- | -- |
| Gemini 2.0 Flash | 15 | 1,000,000 | 200 |
| Gemini 2.0 Flash Preview Image Generation | 10 | 200,000 | 100 |
| Gemini 2.0 Flash-Lite | 30 | 1,000,000 | 200 |
| Imagen 3 | -- | -- | -- |
| Veo 2 | -- | -- | -- |
| Gemini 1.5 Flash (Deprecated) | 15 | 250,000 | 50 |
| Gemini 1.5 Flash-8B (Deprecated) | 15 | 250,000 | 50 |
| Gemini 1.5 Pro (Deprecated) | -- | -- | -- |
| Gemma 3 & 3n | 30 | 15,000 | 14,400 |
| Gemini Embedding Experimental 03-07 | 5 | -- | 100 |
""")
            st.info("RPM = Requests Per Minute, TPM = Tokens Per Minute, RPD = Requests Per Day")
            if st.button("❌ Close Model Limits", key="close_model_limits_btn"):
                st.session_state.show_model_limits = False
                st.rerun()

    return selected_model, temp, max_tokens


def render_file_upload_for_requirements():
    """Render file upload for requirements step only"""
    uploaded_file = st.file_uploader(
        "Upload Requirements PDF", 
        type=['pdf'],
        help="Upload a PDF containing your project requirements",
        key="requirements_file_uploader"
    )
    if uploaded_file is not None:
        temp_path = Path("temp_requirements.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.session_state.uploaded_file = temp_path
        st.markdown(render_status_badge("success", "✅ PDF uploaded successfully!"), unsafe_allow_html=True)
    return uploaded_file

def render_chat_interface():
    """Render chat interface component (no card)"""
    chat_messages = ""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            chat_messages += f'<div class="chat-message chat-user"><strong>You:</strong> {message["content"]}</div>'
        elif message["role"] == "system":
            chat_messages += f'<div class="chat-message chat-system">{message["content"]}</div>'
        else:
            chat_messages += f'<div class="chat-message chat-ai"><strong>AI:</strong> {message["content"]}</div>'
    st.markdown("#### 🤖 Support Chatbot")
    st.markdown(f"""
    <div class="chat-container">
        {chat_messages}
    </div>
    """, unsafe_allow_html=True)
    chat_input = st.text_input(
        "💬 Ask me anything:",
        placeholder="How can I help you today?",
        key="chat_input"
    )
    send_clicked = st.button("Send 📤", use_container_width=True, key="chat_send_btn")
    clear_clicked = st.button("Clear 🗑️", use_container_width=True, key="chat_clear_btn")
    return chat_input, send_clicked, clear_clicked

def render_footer():
    """Render footer component"""
    st.markdown("---")
    html = f"""
    <div style='text-align: center; color: var(--text-muted); padding: 2rem 0;'>
        <p>🚀 <strong>AI Pipeline Studio</strong> | Powered by LangChain & Google Gemini</p>
        <p>Theme: {'🌙 Dark Mode' if st.session_state.dark_mode else '☀️ Light Mode'}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_status_badge(status_type, message):
    """Render a status badge"""
    return f'<div class="status-badge status-{status_type}">{message}</div>'
