"""
app.py

VBCUA — Voice-Based Concept Understanding Analyser
Interactive Streamlit Dashboard.
Connects to the FastAPI server (running on port 8000) or falls back to direct module execution.
"""

import json
import logging
import os
from pathlib import Path
import requests

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Configure page (must be first)
st.set_page_config(
    page_title="VBCUA — Voice-Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Load modules directly for local execution fallback if FastAPI is offline
# --------------------------------------------------------------------------
try:
    from modules import (
        audio_utils,
        db_manager,
        email_utils,
        gemini_feedback,
        nlp_utils,
        report_generator,
        scoring_engine,
        semantic_eval,
        speech_to_text,
    )
    LOCAL_FALLBACK_AVAILABLE = True
except ImportError:
    LOCAL_FALLBACK_AVAILABLE = False

# Logger
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = "http://localhost:8000"

TIER_COLORS = {
    "Strong Understanding": "#2ecc71",
    "Moderate Understanding": "#f39c12",
    "Poor Understanding": "#e74c3c",
}

REFERENCE_CONCEPTS = {
    "Machine Learning": (
        "Machine learning is a subfield of artificial intelligence in which systems "
        "learn patterns directly from data rather than following explicitly programmed "
        "rules. A model is trained on historical examples, adjusts its internal "
        "parameters to minimize prediction error, and then generalizes that learned "
        "pattern to make predictions on new, unseen data. Common categories include "
        "supervised learning, where labeled data teaches the model to map inputs to "
        "outputs; unsupervised learning, which finds structure in unlabeled data; and "
        "reinforcement learning, where an agent learns by trial and error through "
        "rewards and penalties."
    ),
    "Cloud Computing": (
        "Cloud computing is the on-demand delivery of computing resources — including "
        "servers, storage, databases, networking, software, analytics, and intelligence "
        "— over the internet to offer faster innovation, flexible resources, and "
        "economies of scale. Cloud services are broadly categorised as Infrastructure "
        "as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service "
        "(SaaS). Major deployment models include public cloud, private cloud, and "
        "hybrid cloud, each balancing control, security, and cost differently."
    ),
    "Artificial Intelligence": (
        "Artificial intelligence refers to the simulation of human intelligence "
        "processes by computer systems. These processes include learning from "
        "experience, reasoning to reach conclusions, and self-correction. AI "
        "applications span natural language processing, computer vision, robotics, "
        "expert systems, and autonomous vehicles. Modern AI relies on large datasets, "
        "powerful GPUs, and advanced algorithms such as deep neural networks to solve "
        "complex real-world problems that were previously thought to require human-level "
        "understanding and perception."
    ),
    "Deep Learning": (
        "Deep learning is a subset of machine learning that uses artificial neural "
        "networks with multiple layers — called deep neural networks — to model and "
        "understand complex patterns in data. It excels at tasks like image recognition, "
        "speech synthesis, natural language understanding, and game playing. Deep "
        "learning relies on large amounts of training data and considerable computational "
        "power, typically using GPUs, to automatically learn hierarchical feature "
        "representations directly from raw inputs without manual feature engineering."
    ),
    "Custom (Enter below)": "",
}

# --------------------------------------------------------------------------
# Premium Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0a0f1e 100%);
    }

    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a2744 0%, #1f2937 60%, #1a2438 100%);
        border: 1px solid rgba(46,204,113,0.25);
        border-radius: 16px;
        padding: 30px 40px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #8b949e;
        margin: 0;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(46,204,113,0.15);
        border: 1px solid rgba(46,204,113,0.35);
        color: #2ecc71;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 14px;
        letter-spacing: 0.5px;
    }

    /* Section Labels */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #2ecc71;
        margin-bottom: 10px;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 18px;
    }

    /* Score Display */
    .score-circle {
        text-align: center;
        padding: 20px 0;
    }
    .score-number {
        font-size: 3.8rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
        letter-spacing: -2px;
    }
    .score-denom {
        font-size: 1.4rem;
        color: #8b949e;
        font-weight: 400;
    }
    .score-label {
        font-size: 0.82rem;
        color: #8b949e;
        margin-top: 4px;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Tier Badge */
    .tier-badge {
        display: inline-block;
        padding: 8px 22px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.3px;
        margin-top: 10px;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(13, 17, 23, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.9);
        border-radius: 10px;
        padding: 16px 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e6edf3;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #8b949e;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Score Breakdown Bar */
    .breakdown-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid rgba(48,54,61,0.5);
    }
    .breakdown-row:last-child { border-bottom: none; }
    .breakdown-name {
        font-size: 0.85rem;
        color: #c9d1d9;
        font-weight: 500;
        flex: 1;
    }
    .breakdown-bar-wrap {
        flex: 2;
        height: 6px;
        background: rgba(48,54,61,0.8);
        border-radius: 999px;
        margin: 0 12px;
    }
    .breakdown-bar {
        height: 6px;
        border-radius: 999px;
        background: linear-gradient(90deg, #2ecc71, #27ae60);
    }
    .breakdown-pts {
        font-size: 0.85rem;
        font-weight: 700;
        color: #2ecc71;
        min-width: 42px;
        text-align: right;
    }

    /* Transcript Box */
    .transcript-box {
        background: rgba(13,17,23,0.8);
        border: 1px solid rgba(48,54,61,0.8);
        border-radius: 10px;
        padding: 18px 20px;
        font-size: 0.92rem;
        color: #c9d1d9;
        line-height: 1.7;
        max-height: 200px;
        overflow-y: auto;
    }

    .feedback-box {
        background: rgba(13,17,23,0.6);
        border: 1px solid rgba(48,54,61,0.7);
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 0.88rem;
        color: #e6edf3;
        line-height: 1.5;
    }

    .status-complete {
        background: linear-gradient(90deg, rgba(46,204,113,0.12), rgba(39,174,96,0.06));
        border: 1px solid rgba(46,204,113,0.3);
        border-radius: 10px;
        padding: 14px 20px;
        color: #2ecc71;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Reset / custom styles for Streamlit widgets */
    .stTextArea textarea {
        background: #0d1117 !important;
        border: 1px solid rgba(48,54,61,0.9) !important;
        color: #c9d1d9 !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stFileUploader > div {
        background: rgba(22,27,34,0.7) !important;
        border: 2px dashed rgba(46,204,113,0.35) !important;
        border-radius: 12px !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2ecc71, #27ae60) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 14px 32px !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(46,204,113,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 28px rgba(46,204,113,0.45) !important;
    }
    .stDownloadButton > button {
        background: rgba(46,204,113,0.1) !important;
        border: 1px solid rgba(46,204,113,0.4) !important;
        color: #2ecc71 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(46,204,113,0.2) !important;
    }
    
    /* Sign Up Layout */
    .signup-container {
        max-width: 500px;
        margin: 50px auto;
        background: rgba(22, 27, 34, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(48, 54, 61, 0.8) !important;
        border-radius: 16px !important;
        padding: 35px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    }
    .signup-title {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        text-align: center !important;
        margin-bottom: 8px !important;
    }
    .signup-subtitle {
        font-size: 0.95rem !important;
        color: #8b949e !important;
        text-align: center !important;
        margin-bottom: 25px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session State Init
# --------------------------------------------------------------------------
if "evaluation_results" not in st.session_state:
    st.session_state["evaluation_results"] = None
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "otp" not in st.session_state:
    st.session_state["otp"] = None
if "email_sent" not in st.session_state:
    st.session_state["email_sent"] = False
if "user_info" not in st.session_state:
    st.session_state["user_info"] = {}


# --------------------------------------------------------------------------
# Sign-Up & OTP Verification Page
# --------------------------------------------------------------------------
if not st.session_state["authenticated"]:
    # Resolve SMTP Configuration at runtime
    smtp_host_env = os.getenv("SMTP_HOST", "")
    smtp_port_env = int(os.getenv("SMTP_PORT", "587"))
    smtp_user_env = os.getenv("SMTP_USER", "")
    smtp_password_env = os.getenv("SMTP_PASSWORD", "")
    smtp_from_env = os.getenv("SMTP_FROM", smtp_user_env)
    
    smtp_configured = bool(smtp_host_env and smtp_user_env and smtp_password_env)

    # Render Sign Up Layout in a clean, isolated view
    st.markdown('<div class="signup-container">', unsafe_allow_html=True)
    st.markdown('<div class="signup-title">🎙️ VBCUA Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="signup-subtitle">Verify your identity with OTP to access the Analyser</div>', unsafe_allow_html=True)
    
    if not st.session_state["email_sent"]:
        # Stage 1: Get Details
        auth_mode = st.radio("Choose Access Mode:", ["🔑 Log In", "📝 Sign Up"], horizontal=True, label_visibility="collapsed")
        
        if auth_mode == "🔑 Log In":
            su_email = st.text_input("Registered Email Address", placeholder="e.g. jane.doe@vbcua.edu")
            su_name = ""
            su_role = ""
        else:
            su_name = st.text_input("Full Name", placeholder="e.g. Jane Doe")
            su_email = st.text_input("Email Address", placeholder="e.g. jane.doe@vbcua.edu")
            su_role = st.selectbox("Your Role", ["learner", "educator", "researcher"])
            
        # Accordion UI configuration for SMTP if needed
        smtp_expander = st.expander("⚙️ SMTP Email Server Settings", expanded=not smtp_configured)
        with smtp_expander:
            if smtp_configured:
                st.success("🟢 Loaded SMTP configurations from .env")
            ui_smtp_host = st.text_input("SMTP Host", value=smtp_host_env or "smtp.gmail.com")
            ui_smtp_port = st.number_input("SMTP Port", value=smtp_port_env, min_value=1, max_value=65535)
            ui_smtp_user = st.text_input("SMTP Username (Email)", value=smtp_user_env, placeholder="e.g. sender@gmail.com")
            ui_smtp_password = st.text_input("SMTP App Password", value=smtp_password_env, type="password", placeholder="Provide 16-character App Password")
            ui_smtp_from = st.text_input("SMTP From Email (Optional)", value=smtp_from_env or ui_smtp_user)
            
        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        if st.button("Send Verification OTP", type="primary", use_container_width=True):
            if auth_mode == "📝 Sign Up" and (not su_name.strip() or not su_email.strip()):
                st.error("⚠️ Please fill in all fields.")
            elif auth_mode == "🔑 Log In" and not su_email.strip():
                st.error("⚠️ Please enter your registered email address.")
            elif "@" not in su_email or "." not in su_email:
                st.error("⚠️ Please enter a valid email address.")
            else:
                # Check if email exists in DB if logging in
                user_record = None
                if auth_mode == "🔑 Log In":
                    if LOCAL_FALLBACK_AVAILABLE:
                        try:
                            user_record = db_manager.get_user_by_email(su_email.strip())
                        except Exception as e:
                            logger.exception("Failed to query user record locally.")
                    if not user_record:
                        st.error("❌ Email address not registered. Please Sign Up first.")
                        st.stop()

                # Resolve SMTP details to use
                active_host = smtp_host_env or ui_smtp_host
                active_port = smtp_port_env or ui_smtp_port
                active_user = smtp_user_env or ui_smtp_user
                active_password = smtp_password_env or ui_smtp_password
                active_from = smtp_from_env or ui_smtp_from or active_user
                
                if not active_host or not active_user or not active_password:
                    st.error("⚠️ SMTP Server Settings must be fully configured to dispatch the verification OTP email.")
                else:
                    import random
                    generated_otp = str(random.randint(100000, 999999))
                    
                    with st.spinner("📧 Delivering verification code to your inbox..."):
                        try:
                            email_utils.send_otp_email(
                                to_email=su_email.strip(),
                                otp=generated_otp,
                                smtp_host=active_host,
                                smtp_port=int(active_port),
                                smtp_user=active_user,
                                smtp_password=active_password,
                                smtp_from=active_from
                            )
                            st.session_state["otp"] = generated_otp
                            if auth_mode == "🔑 Log In" and user_record:
                                st.session_state["user_info"] = {
                                    "name": user_record["name"],
                                    "email": user_record["email"],
                                    "role": user_record["role"]
                                }
                            else:
                                st.session_state["user_info"] = {
                                    "name": su_name.strip(),
                                    "email": su_email.strip(),
                                    "role": su_role
                                }
                            st.session_state["email_sent"] = True
                            st.toast(f"📧 Verification OTP sent to {su_email.strip()}!", icon="📧")
                            st.rerun()
                        except Exception as err:
                            st.error(f"❌ Failed to dispatch email: {err}")
                            logger.exception("SMTP email delivery failed.")
    else:
        # Stage 2: OTP Verification
        user_info = st.session_state["user_info"]
        st.info(f"📧 A 6-digit verification code has been dispatched to **{user_info['email']}**.\n\n"
                f"Please check your inbox (and spam folder) and input the code below.")
        
        entered_otp = st.text_input("Enter 6-Digit OTP", max_chars=6, placeholder="######")
        
        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        verify_col1, verify_col2 = st.columns(2)
        with verify_col1:
            if st.button("Verify & Enter Dashboard", type="primary", use_container_width=True):
                if entered_otp.strip() == st.session_state["otp"]:
                    # Save user in the database
                    if LOCAL_FALLBACK_AVAILABLE:
                        try:
                            db_manager.save_user(
                                name=user_info["name"],
                                email=user_info["email"],
                                role=user_info["role"]
                            )
                        except Exception as e:
                            logger.exception("Failed to save user in local database fallback.")
                    st.session_state["authenticated"] = True
                    st.success("✅ OTP verified successfully!")
                    st.rerun()
                else:
                    st.error("❌ Invalid OTP. Please try again.")
        with verify_col2:
            if st.button("Change Details / Go Back", use_container_width=True):
                st.session_state["email_sent"] = False
                st.session_state["otp"] = None
                st.session_state["user_info"] = {}
                st.rerun()
                
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# authenticated user info references
user_name = st.session_state["user_info"]["name"]
user_email = st.session_state["user_info"]["email"]
user_role = st.session_state["user_info"]["role"]


# --------------------------------------------------------------------------
# Sidebar Config
# --------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/microphone.png", width=70)
    st.title("VBCUA Control Panel")
    st.markdown("---")
    
    st.subheader("👤 User Profile")
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")
    st.write(f"**Role:** {user_role.capitalize()}")
    
    st.markdown("---")
    st.subheader("🔑 API Configurations")
    gemini_api_key = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.get("gemini_api_key", ""),
        type="password",
        help="Provide your API key to generate intelligent qualitative recommendations."
    )
    if gemini_api_key:
        st.session_state["gemini_api_key"] = gemini_api_key

    st.markdown("---")
    # Backend Service connection indicator
    try:
        res = requests.get(f"{API_BASE_URL}/api/concepts", timeout=1.5)
        backend_online = res.status_code == 200
    except requests.exceptions.RequestException:
        backend_online = False
        
    if backend_online:
        st.success("🟢 FastAPI Backend: Online")
    else:
        st.warning("🟡 FastAPI Backend: Offline (Running locally)")

    st.markdown("---")
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["otp"] = None
        st.session_state["email_sent"] = False
        st.session_state["user_info"] = {}
        st.rerun()


# --------------------------------------------------------------------------
# Hero Banner
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">✦ VBCUA EDUCATIONAL PLATFORM</div>
        <h1 class="hero-title">🎙️ Voice-Based Concept Understanding Analyser</h1>
        <p class="hero-subtitle">
            Evaluate conceptual explanations, delivery fluency, articulation gaps, and vocal confidence.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Create Tabs
tab_evaluate, tab_history = st.tabs(["🚀 Concept Evaluation", "📜 Assessment History"])


# ==========================================================================
# ██ TAB 1: EVALUATE CONCEPT
# ==========================================================================
with tab_evaluate:
    col_input_left, col_input_right = st.columns([3, 2], gap="large")

    with col_input_left:
        st.markdown('<div class="section-label">Step 1 — Ingest Explanation</div>', unsafe_allow_html=True)
        st.subheader("🎧 Upload Explanation File")
        uploaded_file = st.file_uploader(
            "Upload any explanation file:",
            accept_multiple_files=False,
            label_visibility="collapsed"
        )
        if uploaded_file:
            if uploaded_file.name.lower().endswith(".mp4"):
                st.video(uploaded_file)
            elif uploaded_file.name.lower().endswith((".wav", ".mp3")):
                st.audio(uploaded_file)
            else:
                st.info(f"📂 File '{uploaded_file.name}' staged for processing.")

    with col_input_right:
        st.markdown('<div class="section-label">Step 2 — Reference Standard</div>', unsafe_allow_html=True)
        st.subheader("📖 Select Concept")
        
        concept_choice = st.selectbox(
            "Choose reference standards:",
            options=list(REFERENCE_CONCEPTS.keys()),
            label_visibility="collapsed"
        )
        
        default_concept_text = REFERENCE_CONCEPTS[concept_choice]
        reference_text = st.text_area(
            "Reference Concept ground truth:",
            value=default_concept_text,
            height=200,
            label_visibility="collapsed",
            placeholder="Enter the ideal conceptual explanation to align against..."
        )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    
    # Process Button
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        start_analysis = st.button("🚀 Analyze Concept Understanding", use_container_width=True, type="primary")

    if start_analysis:
        if not uploaded_file:
            st.warning("⚠️ Please upload a file first.")
        elif not reference_text.strip():
            st.warning("⚠️ Please provide reference concept text.")
        else:
            with st.spinner("🔄 Running full pipeline (STT, SBERT, NLTK, Gemini)..."):
                # Scenario A: Backend is Online (FastAPI)
                if backend_online:
                    try:
                        files = {"audio": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {
                            "concept_title": concept_choice if concept_choice != "Custom (Enter below)" else "Custom Concept",
                            "concept_text": reference_text,
                            "user_name": user_name,
                            "user_email": user_email,
                            "user_role": user_role,
                            "gemini_api_key": gemini_api_key
                        }
                        response = requests.post(f"{API_BASE_URL}/api/evaluate", files=files, data=data)
                        if response.status_code == 200:
                            st.session_state["evaluation_results"] = response.json()
                            st.toast("✅ Analysis complete (FastAPI API)!", icon="✅")
                        else:
                            st.error(f"Backend failed: {response.text}")
                    except Exception as e:
                        st.error(f"API Request failed: {e}")
                
                # Scenario B: Single-Process Module Fallback
                elif LOCAL_FALLBACK_AVAILABLE:
                    try:
                        # Save temp upload
                        temp_dir = Path("assets/uploads")
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        temp_path = temp_dir / uploaded_file.name
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Load signal + waveform
                        y, sr = audio_utils.load_audio_signal(temp_path)
                        raw_metrics = audio_utils.extract_raw_audio_metrics(y, sr)
                        
                        wave_dir = Path("assets/waveforms")
                        wave_dir.mkdir(parents=True, exist_ok=True)
                        waveform_path = wave_dir / f"{uploaded_file.name}.png"
                        audio_utils.save_waveform(temp_path, waveform_path)

                        # Transcribe
                        stt = speech_to_text.SpeechToTextEngine(model_size="base")
                        transcript = stt.transcribe(temp_path)

                        # Match
                        se = semantic_eval.SemanticEvaluator()
                        sim = se.compute_similarity(transcript, reference_text)

                        # Filler
                        filler_ratio, filler_count, total_words = scoring_engine.calculate_filler_ratio(transcript)

                        # Score
                        brk = scoring_engine.evaluate_understanding(
                            similarity=sim,
                            filler_ratio=filler_ratio,
                            audio={"pause_ratio": raw_metrics["pause_ratio"], "rms_energy": raw_metrics["rms_energy"]}
                        )

                        # NLP & Gemini feedback
                        nlp_analyzer = nlp_utils.NLPAnalyzer()
                        nlp_results = nlp_analyzer.analyze_text(transcript)
                        
                        feedback = gemini_feedback.generate_qualitative_feedback(
                            student_transcript=transcript,
                            reference_concept=reference_text,
                            overall_score=brk.overall_score,
                            filler_ratio=filler_ratio,
                            pause_ratio=raw_metrics["pause_ratio"],
                            similarity_score=sim,
                            api_key=gemini_api_key
                        )

                        # DB Logging
                        u_id = db_manager.save_user(user_name, user_email, user_role)
                        rc_id = db_manager.save_reference_concept(concept_choice, reference_text)
                        a_id = db_manager.save_audio_file(u_id, uploaded_file.name, str(temp_path), raw_metrics["duration_sec"])
                        t_id = db_manager.save_transcript(a_id, transcript)
                        db_manager.save_audio_features(a_id, raw_metrics["pause_ratio"], raw_metrics["rms_energy"], raw_metrics["zero_crossing_rate"], raw_metrics["duration_sec"])
                        db_manager.save_filler_word_stats(t_id, filler_count, total_words, filler_ratio)
                        db_manager.save_semantic_similarity(t_id, rc_id, sim)
                        
                        notes_dict = {
                            "strengths": feedback["strengths"],
                            "gaps": feedback["gaps"],
                            "tips": feedback["tips"],
                            "sentiment": nlp_results["sentiment"],
                            "lexical_diversity": nlp_results["lexical_diversity"],
                            "sentence_count": nlp_results["sentence_count"]
                        }
                        
                        res_id = db_manager.save_evaluation_result(a_id, rc_id, brk.overall_score, brk.understanding_level, json.dumps(notes_dict))

                        # PDF Report
                        reports_dir = Path("assets/reports")
                        reports_dir.mkdir(parents=True, exist_ok=True)
                        pdf_path = reports_dir / f"{uploaded_file.name}_report.pdf"
                        
                        report_generator.generate_pdf_report(
                            output_filename=str(pdf_path),
                            reference_concept=reference_text,
                            student_transcript=transcript,
                            waveform_img_path=str(waveform_path),
                            metrics_dict={
                                "semantic_similarity": sim,
                                "filler_ratio": filler_ratio,
                                "pause_ratio": raw_metrics["pause_ratio"],
                                "rms_energy": raw_metrics["rms_energy"],
                                "overall_score": brk.overall_score,
                                "understanding_level": brk.understanding_level,
                                "zero_crossing_rate": raw_metrics["zero_crossing_rate"],
                                "lexical_diversity": nlp_results["lexical_diversity"],
                                "sentiment": nlp_results["sentiment"],
                                "strengths": feedback["strengths"],
                                "gaps": feedback["gaps"],
                                "tips": feedback["tips"]
                            }
                        )
                        
                        db_manager.save_report(res_id, str(pdf_path), round(Path(pdf_path).stat().st_size / 1024, 2))
                        db_manager.save_session(u_id, "ended")

                        # Populate state
                        st.session_state["evaluation_results"] = {
                            "transcript": transcript,
                            "overall_score": brk.overall_score,
                            "understanding_level": brk.understanding_level,
                            "tier_color": brk.tier_color,
                            "metrics": {
                                "similarity": sim,
                                "filler_ratio": filler_ratio,
                                "filler_count": filler_count,
                                "total_words": total_words,
                                "pause_ratio": raw_metrics["pause_ratio"],
                                "rms_energy": raw_metrics["rms_energy"],
                                "zero_crossing_rate": raw_metrics["zero_crossing_rate"],
                                "duration_sec": raw_metrics["duration_sec"],
                                "lexical_diversity": nlp_results["lexical_diversity"],
                                "sentence_count": nlp_results["sentence_count"],
                                "sentiment": nlp_results["sentiment"]
                            },
                            "feedback": feedback,
                            "waveform_path": str(waveform_path),
                            "pdf_path": str(pdf_path)
                        }
                        st.toast("✅ Analysis complete (Local Modules)!", icon="✅")
                    except Exception as e:
                        st.error(f"Local pipeline failed: {e}")
                        logger.exception("Local pipeline error.")
                else:
                    st.error("Error: FastAPI is offline and local modules are missing/unimportable.")

    # ── Render Results ──
    if st.session_state["evaluation_results"]:
        results = st.session_state["evaluation_results"]
        
        st.markdown("<hr style='border-top: 1px solid rgba(48,54,61,0.6); margin: 24px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="status-complete">✅ &nbsp; Analysis Completed</div>', unsafe_allow_html=True)
        
        # Transcript + Score Gauge
        res_col_left, res_col_right = st.columns([3, 2], gap="large")
        
        with res_col_left:
            st.markdown('<div class="section-label">Transcribed Speech</div>', unsafe_allow_html=True)
            t_text = results["transcript"] or "(No speech detected in audio.)"
            st.markdown(f'<div class="transcript-box">{t_text}</div>', unsafe_allow_html=True)
            
            # Simple metadata indicators
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            with sub_col1:
                st.metric("Total Words", results["metrics"]["total_words"])
            with sub_col2:
                st.metric("Filler Count", results["metrics"]["filler_count"])
            with sub_col3:
                st.metric("Duration (s)", f"{results['metrics']['duration_sec']:.1f}s")
                
        with res_col_right:
            st.markdown('<div class="section-label">Evaluation Summary</div>', unsafe_allow_html=True)
            score = results["overall_score"]
            level = results["understanding_level"]
            color = results["tier_color"]
            
            st.markdown(
                f"""
                <div style="text-align:center; padding: 22px; background: rgba(13,17,23,0.8);
                            border: 1px solid rgba(48,54,61,0.9); border-radius: 12px;">
                    <div class="score-label">Understanding Score</div>
                    <div class="score-number">{score}<span class="score-denom">/100</span></div>
                    <div style="margin-top:14px;">
                        <span class="tier-badge" style="background-color:{color}20;
                            border: 2px solid {color}; color:{color};">
                            {level}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Waveform + NLP Cards
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        mid_col_left, mid_col_right = st.columns([3, 2], gap="large")
        
        with mid_col_left:
            st.markdown('<div class="section-label">Audio Waveform</div>', unsafe_allow_html=True)
            if results.get("waveform_path") and Path(results["waveform_path"]).is_file():
                st.image(results["waveform_path"], use_container_width=True)
            else:
                st.info("Waveform image unavailable.")
                
        with mid_col_right:
            st.markdown('<div class="section-label">NLP Delivery & Tone Analysis</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="glass-card" style="padding: 18px;">
                    <p style="margin: 0 0 10px 0; font-size:0.85rem; color:#8b949e;">
                        <b>Lexical Diversity (TTR):</b> &nbsp; {results['metrics']['lexical_diversity']:.3f}
                    </p>
                    <p style="margin: 0 0 10px 0; font-size:0.85rem; color:#8b949e;">
                        <b>Delivery Tone:</b> &nbsp; {results['metrics']['sentiment']}
                    </p>
                    <p style="margin: 0; font-size:0.85rem; color:#8b949e;">
                        <b>Sentence Count:</b> &nbsp; {results['metrics']['sentence_count']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Detailed Metric Cards
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Extracted Feature Breakdown</div>', unsafe_allow_html=True)
        
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("🧠 Semantic Similarity", f"{results['metrics']['similarity']:.3f}")
        with m_col2:
            st.metric("🗣️ Filler Word Ratio", f"{results['metrics']['filler_ratio']:.3f}")
        with m_col3:
            st.metric("⏸️ Pause Ratio", f"{results['metrics']['pause_ratio']:.3f}")
        with m_col4:
            st.metric("⚡ RMS Energy", f"{results['metrics']['rms_energy']:.4f}")
        with m_col5:
            st.metric("〰️ Zero-Crossing Rate", f"{results['metrics']['zero_crossing_rate']:.4f}")

        # Gemini Qualitative Feedback Panel
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Qualitative Feedback & Recommendations</div>', unsafe_allow_html=True)
        
        f_col1, f_col2, f_col3 = st.columns(3, gap="medium")
        with f_col1:
            st.markdown("**🌟 Strengths**")
            st.markdown(f'<div class="feedback-box">{results["feedback"]["strengths"]}</div>', unsafe_allow_html=True)
        with f_col2:
            st.markdown("**⚠️ Gaps in Understanding**")
            st.markdown(f'<div class="feedback-box">{results["feedback"]["gaps"]}</div>', unsafe_allow_html=True)
        with f_col3:
            st.markdown("**💡 Delivery & Articulation Tips**")
            st.markdown(f'<div class="feedback-box">{results["feedback"]["tips"]}</div>', unsafe_allow_html=True)

        # Report Downloads
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
        dl_col1, dl_col2 = st.columns([1, 4])
        with dl_col1:
            if results.get("pdf_path") and Path(results["pdf_path"]).is_file():
                with open(results["pdf_path"], "rb") as f:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=f.read(),
                        file_name="vbcua_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.info("PDF Report is currently generation-failed or offline.")
        with dl_col2:
            if st.button("🔄 Analyze Another Concept"):
                st.session_state["evaluation_results"] = None
                st.rerun()


# ==========================================================================
# ██ TAB 2: HISTORY EXPLORER
# ==========================================================================
with tab_history:
    st.markdown('<div class="section-label">Assessment Log</div>', unsafe_allow_html=True)
    st.subheader("📜 Review Previous Submissions")

    # Fetch history
    history_list = []
    if backend_online:
        try:
            res = requests.get(f"{API_BASE_URL}/api/history")
            if res.status_code == 200:
                history_list = res.json()
        except Exception:
            pass
    elif LOCAL_FALLBACK_AVAILABLE:
        try:
            history_list = db_manager.get_evaluation_history()
        except Exception:
            pass
            
    if not history_list:
        st.info("No historical concept submissions recorded in the database yet.")
    else:
        # Create dictionary map for dropdown display
        history_map = {}
        for row in history_list:
            display_str = f"[{row['uploaded_at']}] {row['concept_title']} — User: {row['user_name']} ({row['overall_score']}/100)"
            history_map[display_str] = row
            
        selected_key = st.selectbox("Select previous assessment record:", options=list(history_map.keys()))
        
        if selected_key:
            selected_row = history_map[selected_key]
            
            # Load JSON notes containing qualitative feedback
            try:
                selected_notes = json.loads(selected_row["notes"])
            except Exception:
                selected_notes = {
                    "strengths": selected_row.get("notes", ""),
                    "gaps": "See reference details.",
                    "tips": "Reduce fillers.",
                    "sentiment": "Neutral",
                    "lexical_diversity": 0.0,
                    "sentence_count": 0
                }

            st.markdown("<hr style='border-top: 1px solid rgba(48,54,61,0.4); margin: 20px 0;'>", unsafe_allow_html=True)
            
            h_col1, h_col2 = st.columns([3, 2], gap="large")
            with h_col1:
                st.markdown("**Transcript Explanation**")
                st.code(selected_row["transcript_text"] or "(No speech text recorded.)", language=None)
                
            with h_col2:
                # Render score box
                h_score = selected_row["overall_score"]
                h_level = selected_row["understanding_level"]
                h_color = TIER_COLORS.get(h_level, "#999999")
                
                st.markdown(
                    f"""
                    <div style="text-align:center; padding: 18px; background: rgba(13,17,23,0.8);
                                border: 1px solid rgba(48,54,61,0.9); border-radius: 12px; max-width: 320px; margin: 0 auto;">
                        <div class="score-label">Score Recalled</div>
                        <div class="score-number">{h_score}<span class="score-denom">/100</span></div>
                        <div style="margin-top:10px;">
                            <span class="tier-badge" style="background-color:{h_color}20;
                                border: 2px solid {h_color}; color:{h_color};">
                                {h_level}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Historical qualitative feedback
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            hf_col1, hf_col2, hf_col3 = st.columns(3, gap="medium")
            with hf_col1:
                st.markdown("**🌟 Strengths**")
                st.markdown(f'<div class="feedback-box">{selected_notes.get("strengths", "")}</div>', unsafe_allow_html=True)
            with hf_col2:
                st.markdown("**⚠️ Gaps in Understanding**")
                st.markdown(f'<div class="feedback-box">{selected_notes.get("gaps", "")}</div>', unsafe_allow_html=True)
            with hf_col3:
                st.markdown("**💡 Delivery & Articulation Tips**")
                st.markdown(f'<div class="feedback-box">{selected_notes.get("tips", "")}</div>', unsafe_allow_html=True)

            # Historical PDF download
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
            if selected_row.get("pdf_path") and Path(selected_row["pdf_path"]).is_file():
                with open(selected_row["pdf_path"], "rb") as h_file:
                    st.download_button(
                        label="📄 Download Historical PDF Report",
                        data=h_file.read(),
                        file_name=f"historical_vbcua_{selected_row['result_id']}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.info("PDF report is unavailable or path is invalid for this historical run.")

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; margin-top:40px; color: rgba(139,148,158,0.5);
                font-size:0.75rem; border-top:1px solid rgba(48,54,61,0.3); padding-top:16px;">
        VBCUA &nbsp;·&nbsp; Voice-Based Concept Understanding Analyser &nbsp;·&nbsp;
        Powered by OpenAI Whisper · Sentence-BERT · Librosa · NLTK · ReportLab
    </div>
    """,
    unsafe_allow_html=True,
)
