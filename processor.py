import tempfile, time, json
from pathlib import Path

# Import your modules
from transcribe.main import transcribe_audio_parallel
from analyse.main import analyse
import logging
logger = logging.getLogger("call_analytics_api")
SUPPORTED_MODELS = {
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
}

# -------------------- Configs --------------------
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

# -------------------- Core Processing --------------------
def process_audio_file(uploaded_bytes, filename, selected_model, job_id: str, thinking_budget=100):
    """
    Process a single uploaded or local audio file.
    Handles transcription and analysis with consistent job_id.
    """
    try:
        logger.info(f"[Job {job_id}] Starting processing for file '{filename}' using model {selected_model}")

        if selected_model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {selected_model}")

        # Save to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(uploaded_bytes)
            tmp_path = tmp.name

        start_time = time.time()

        # Step 1 — Transcription
        logger.info(f"[Job {job_id}] Beginning transcription...")
        trans_result = transcribe_audio_parallel(
            tmp_path, model_name=selected_model, job_id=job_id, thinking_budget=thinking_budget
        )
        transcript_json = trans_result.get("transcript", {})
        usage_trans = trans_result.get("usage_metadata", {})

        # Step 2 — Analysis
        logger.info(f"[Job {job_id}] Beginning analysis...")
        analysis_result = analyse(
            transcript_json, selected_model, KEYWORDS, BOOL_CHECKS_INPUT, job_id=job_id, thinking_budget=thinking_budget
        )
        usage_analysis = analysis_result.get("analysis_usage_metadata", {})

        total_time = round(time.time() - start_time, 2)

        logger.info(f"[Job {job_id}] ✅ Processing complete in {total_time}s")

        return {
            "job_id": job_id,
            "file_name": filename,
            "model": selected_model,
            "transcription_usage": usage_trans,
            "analysis_usage": usage_analysis,
            "transcript": transcript_json,
            "analysis_result": analysis_result,
            "total_time_sec": total_time,
        }

    except Exception as e:
        logger.exception(f"[Job {job_id}] ❌ Processing failed: {e}")
        return {"job_id": job_id, "error": str(e)}
