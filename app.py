import streamlit as st
import json
import tempfile
import time
from pathlib import Path

# Import your main functions
from transcribe.main import transcribe_audio_parallel
from analyse.main import analyse, GOOGLE_LLM_MODEL

# Define model options (can extend easily)
MODEL_OPTIONS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

MODEL_PRICING = {
    "gemini-2.5-flash-lite": {
        "input_text": 0.10,
        "input_audio": 0.30,
        "output": 0.40,
        "cached_text": 0.01,
        "cached_audio": 0.03,
    },
    "gemini-2.5-flash": {
        "input_text": 0.10,
        "input_audio": 0.30,
        "output": 0.40,
        "cached_text": 0.01,
        "cached_audio": 0.03,
    },
    "gemini-2.0-flash": {
        "input_text": 0.10,
        "input_audio": 0.70,
        "output": 0.40,
        "cached_text": 0.025,
        "cached_audio": 0.175,
    },
    "gemini-2.0-flash-lite": {
        "input_text": 0.075,
        "input_audio": 0.075,  # same price for text/audio per spec
        "output": 0.30,
        "cached_text": 0.0,
        "cached_audio": 0.0,
    },
}

# Define keywords and boolean checks (you can move these to config)
KEYWORDS = {
    "intent_keywords": ["product features", "trial demo", "schedule call"]
}

BOOL_CHECKS_INPUT = [
    {"field": "agent_introduced", "type": "boolean", "description": "Did the agent introduce themselves properly?"},
    {"field": "agent_closed_properly", "type": "boolean", "description": "Did the agent close the call properly?"},
    {"field": "issue_resolved", "type": "boolean", "description": "Was the customer issue resolved?"},
    {"field": "feedback_requested", "type": "boolean", "description": "Did the agent request feedback from the customer?"},
    {"field": "verification", "type": "qa", "description": "Did the agent verify customer information correctly?"},
    {"field": "objection_handling", "type": "qa", "description": "Did the agent handle objections properly?"},
    {"field": "acknowledgement_proactiveness", "type": "qa", "description": "Did the agent proactively acknowledge customer's concerns?"},
    {"field": "empathy", "type": "qa", "description": "Did the agent show empathy during the call?"},
    {"field": "effective_listening", "type": "qa", "description": "Did the agent demonstrate effective listening?"},
    {"field": "slang_jargons", "type": "qa", "description": "Did the agent avoid using slang or jargons?"},
    {"field": "unprofessional_speech", "type": "qa", "description": "Did the agent avoid unprofessional speech?"},
    {"field": "abrupt_call_disconnection", "type": "qa", "description": "Was the call disconnected abruptly?"},
    {"field": "rude_on_call", "type": "qa", "description": "Did the agent appear rude on the call?"},
    {"field": "relevant_information", "type": "qa", "description": "Did the agent provide relevant information?"},
    {"field": "probing_general", "type": "qa", "description": "Did the agent ask general probing questions?"},
    {"field": "customer_fcr", "type": "qa", "description": "Was the customer's First Call Resolution achieved?"},
]

# Streamlit App UI
st.set_page_config(page_title="Call Analysis System", layout="wide")
st.title("🎧 AI Call Transcription & Analysis Dashboard")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox("Select LLM Model", MODEL_OPTIONS, index=0)
    
# File uploader
uploaded_file = st.file_uploader("Upload an audio file (.aac, .mp3, .wav)", type=["aac", "mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mpeg")
    if st.button("🚀 Process Audio"):
        with st.spinner("Processing... Please wait (this might take a few minutes)"):
            start_time = time.time()

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Step 1: Transcribe audio
            st.info("🔊 Transcribing audio...")
            transcript_result= transcribe_audio_parallel(tmp_path, model_name= selected_model)
            transcript_json = transcript_result["transcript"]
            usage = transcript_result.get("usage_metadata", {})
            print("+++++++++++++++++++++++++++++", usage)
            st.subheader("📝 Token usage summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Input text tokens", usage.get("input_text_tokens", "N/A"))
            col2.metric("Input audio tokens", usage.get("input_audio_tokens", "N/A"))
            col3.metric("Output tokens", usage.get("output_tokens", "N/A"))
            col4, col5, col6 = st.columns(3)
            col4.metric("Thinking Tokens", usage.get("thinking_tokens", "N/A"))
            col5.metric("chached text tokens", usage.get("cached_text_tokens", "N/A"))
            col6.metric("chached audio tokens", usage.get("cached_audio_tokens", "N/A"))
            total_tokens = sum(usage.values())
            st.metric("Total Tokens Used in Transcription", total_tokens)



            # Step 2: Analyse transcript
            st.info("🧠 Running analysis...")
            result = analyse(transcript_json, selected_model, KEYWORDS, BOOL_CHECKS_INPUT)

            total_time = time.time() - start_time
            st.success(f"✅ Completed in {total_time:.2f} seconds")
            
            # ✅ Display Analysis Token Usage
            if "analysis_usage_metadata" in result:
                st.subheader("📊 Analysis Token Usage")
                analysis_usage = result["analysis_usage_metadata"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Input Tokens", analysis_usage.get("input_tokens", "N/A"))
                col2.metric("Output Tokens", analysis_usage.get("output_tokens", "N/A"))
                col3.metric("Total Analysis Tokens", analysis_usage.get("total_tokens", "N/A"))
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Thinking Tokens", analysis_usage.get("thinking_tokens", "N/A"))
                col5.metric("Cached Text Tokens", analysis_usage.get("cached_text_tokens", "N/A"))
                col6.metric("Cached Audio Tokens", analysis_usage.get("cached_audio_tokens", "N/A"))

            # ✅ Display Combined Total
            st.subheader("💰 Total Token Usage (Transcription + Analysis)")
            total_transcription_tokens = sum(usage.values()) if usage else 0
            total_analysis_tokens = result.get("analysis_usage_metadata", {}).get("total_tokens", 0)
            grand_total = total_transcription_tokens + total_analysis_tokens

            col1, col2, col3 = st.columns(3)
            col1.metric("Transcription Tokens", total_transcription_tokens)
            col2.metric("Analysis Tokens", total_analysis_tokens)
            col3.metric("Grand Total", grand_total)

            # Step 3: Display results
            st.subheader("📜 Transcript Sample")
            st.json(transcript_json[:5])

            st.subheader("📈 Analysis Results")
            st.json(result)

        

            # Step 5: Allow user to download analysis JSON
            st.download_button(
                label="💾 Download Analysis Result",
                data=json.dumps(result, indent=2, ensure_ascii=False),
                file_name="analysis_result.json",
                mime="application/json"
            )
