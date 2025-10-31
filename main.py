"""
analysis_main.py

Full analysis pipeline that:
- Calls Gemini via google.genai
- Uses structured response schemas
- Includes robust retry logic (exponential backoff + special 429 handling)
- Attaches job_id (UUID) for tracking and file outputs
- Integrates Langfuse instrumentation (if configured)
- Returns both analysis JSON and usage metadata
"""

import os
import json
import time
import random
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import re
from collections import Counter

import numpy as np
from dotenv import load_dotenv

import google.genai as genai
from google.genai.types import Part, GenerateContentConfig, FinishReason, ThinkingConfig
from google.genai.errors import ServerError

# Import prompts & schemas from your package
from .prompt import (
    combined_components_prompt,
    combined_components_1_prompt,
    combined_components_2_prompt
)
from .response_schema import (
    combined_components_response_schema,
    combined_components_1_response_schema,
    combined_components_2_response_schema
)

# Langfuse instrumentation (optional)
try:
    from langfuse import get_client
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
    langfuse = get_client()
    assert langfuse.auth_check(), "Langfuse authentication failed!"
    GoogleGenAIInstrumentor().instrument()
except Exception as e:
    # If Langfuse is unavailable, continue without instrumentation
    langfuse = None
    # We'll still allow the script to run
    print("Langfuse not configured or failed to initialize:", e)

# Load environment
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Model config & client
GOOGLE_LLM_MODEL = os.getenv("ANALYSIS_MODEL", "gemini-2.0-flash-lite")
FALLBACK_LLM_MODEL_1 = os.getenv("ANALYSIS_MODEL_FALLBACK", "gemini-2.0-flash")
GCP_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GCP_API_KEY)

# CRM Fields (from your snippet)
crm_fields = [
  "Billing Name",
  "Designation",
  "Current Usage",
  "LinkedIn Link",
  "LinkedIn Designation",
  "Engagement Score Over Call",
  "Objections / Churn Reasons",
  "KDM Name",
  "KDM Phone",
  "KDM Email",
  "Source of Activation"
]

# Default boolean checks (from your snippet)
DEFAULT_BOOLEAN_CHECKS = [
    "Verification – Did the agent confirm the customer/business name, or acknowledge when the customer stated their name? Mark NA if call ended before verification was possible.",
    "Objection Handling – Did the agent handle customer objections? Mark NA if no scope/transfer call.",
    "Acknowledgement & Proactiveness – Did the agent address/paraphrase the issue and take ownership?",
    "Empathy – If the issue was company’s fault, did the agent apologise? (Yes/No/NA).",
    "Effective Listening – Was the agent attentive and responsive to the customer?",
    "Slang/Jargons – Did the agent use heavy slang/jargon that confused the customer? (Yes/No/NA).",
    "Unprofessional Speech – Did the agent rush/fumble/use incomplete words?",
    "Abrupt call disconnection – Did the agent end while customer was still speaking? Follow NA/No rules from guidelines.",
    "Rude on call – Was the agent rude/unprofessional? Apply lenient analysis. (Yes/No/NA).",
    "Relevant Information – Did the agent share relevant info when needed? (Yes/No/NA).",
    "Probing – Did the agent probe appropriately when necessary? (Yes/No/NA).",
    "Customer FCR – Did the agent make a genuine effort to resolve the issue on first contact? (Yes/No/NA).",
    "Agent Introduced – Did the agent introduce themselves?",
    "Agent Closed Properly – Did the agent close politely/professionally?",
    "Issue Resolved – Was the issue resolved during the call?",
    "Feedback Requested – Did the agent request feedback?"
]

# -------------------- Utilities --------------------

def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"

def safe_json_parse(text: str, save_path: str = "bad_response.json") -> Union[dict, list]:
    """
    Safely parse JSON out of a possibly fenced response. Save raw response on failure.
    """
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            if len(parts) >= 2:
                cleaned = parts[1].strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"❌ JSON parse failed: {e}")
        try:
            Path(save_path).write_text(text, encoding="utf-8")
            logger.warning(f"Raw response saved → {save_path}")
        except Exception:
            logger.exception("Failed to save raw response to disk.")
        return {}

def safe_extract_usage_metadata(response) -> Dict[str, int]:
    """Safely extract token usage info from Gemini response."""
    usage_meta = {}
    try:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            logger.debug("No usage_metadata found in response")
            return usage_meta

        input_text_tokens = 0
        input_audio_tokens = 0
        cached_text_tokens = 0
        cached_audio_tokens = 0

        prompt_details = getattr(usage, "prompt_tokens_details", []) or []
        cache_details = getattr(usage, "cache_tokens_details", []) or []
        thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        if prompt_details:
            input_text_tokens = sum(
                i.token_count for i in prompt_details if getattr(i, "modality", None) == "TEXT"
            )
            input_audio_tokens = sum(
                i.token_count for i in prompt_details if getattr(i, "modality", None) == "AUDIO"
            )
        else:
            input_text_tokens = getattr(usage, "prompt_token_count", 0)

        if cache_details:
            cached_text_tokens = sum(
                i.token_count for i in cache_details if getattr(i, "modality", None) and getattr(i.modality, "name", "").upper() == "TEXT"
            )
            cached_audio_tokens = sum(
                i.token_count for i in cache_details if getattr(i, "modality", None) and getattr(i.modality, "name", "").upper() == "AUDIO"
            )
        if thinking_tokens is None:
            thinking_tokens =0
        usage_meta = {
            "input_text_tokens": input_text_tokens,
            "input_audio_tokens": input_audio_tokens,
            "output_tokens": getattr(usage, "candidates_token_count", 0),
            "thinking_tokens": thinking_tokens,
            "cached_text_tokens": cached_text_tokens,
            "cached_audio_tokens": cached_audio_tokens,
            "total_token_count": getattr(usage, "total_token_count", 0),
        }

        # Fallback: if everything zero, try total_token_count
        if sum(int(v or 0) for v in usage_meta.values()) == 0:
            total = getattr(usage, "total_token_count", 0)
            if total:
                usage_meta["input_text_tokens"] = total
    except Exception as e:
        logger.warning(f"⚠️ Error extracting usage metadata: {e}")
    return usage_meta

def count_usage(input_text: str, model_name: str) -> Tuple[int, int]:
    """Return token count and char length for input_text."""
    result = client.models.count_tokens(
        model=model_name,
        contents=[{"role": "user", "parts": [{"text": input_text}]}]
    )
    return result.total_tokens, len(input_text)

# -------------------- Text & Schema Helpers --------------------

def build_bool_checks_text(boolean_checks: List[Dict[str, str]]) -> str:
    lines = []
    for check in boolean_checks:
        if check.get("type") == "boolean":
            lines.append(f"- {check['field']}: {check['description']} Provide True/False with evidence.")
        else:
            lines.append(f"- {check['field']}: {check['description']} Provide Yes/No/NA with justification.")
    return "\n".join(lines)

def build_boolean_checks_schema(bool_checks_input: List[Dict[str, str]]) -> Dict[str, Any]:
    properties = {}
    required_fields = []
    for check in bool_checks_input:
        field = check["field"]
        required_fields.append(field)
        if check.get("type") == "boolean":
            properties[field] = {
                "type": "object",
                "properties": {
                    "value": {"type": "boolean"},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamps": {"type": "string"},
                                "speaker": {"type": "string"},
                                "utterance": {"type": "string"},
                            },
                            "required": ["timestamps", "speaker", "utterance"]
                        }
                    },
                    "justification": {"type": "string"}
                },
                "required": ["value", "evidence"]
            }
        else:
            properties[field] = {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "enum": ["Yes", "No", "NA"]},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamps": {"type": "string"},
                                "speaker": {"type": "string"},
                                "utterance": {"type": "string"},
                            },
                            "required": ["timestamps", "speaker", "utterance"]
                        }
                    },
                    "justification": {"type": "string"}
                },
                "required": ["value", "justification"],
            }
    return {
        "type": "object",
        "properties": {"booleans": {"type": "object", "properties": properties, "required": required_fields}},
        "required": ["booleans"]
    }

# -------------------- Analysis Utility Functions --------------------

def extract_transcript(response) -> str:
    """Return textual content from Gemini response object (first candidate part text)."""
    try:
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and len(candidate.content.parts) > 0:
                return candidate.content.parts[0].text
        logger.warning("Empty or malformed response from Gemini.")
        return ""
    except Exception as e:
        logger.error(f"Failed to extract transcript: {e}")
        return ""

def average_sentiment(full_transcript: List[dict]) -> Tuple[float, float]:
    customer_sentiments = [d.get("sentiment", 0) for d in full_transcript if d.get("speaker") == "customer" and "sentiment" in d]
    agent_sentiments = [d.get("sentiment", 0) for d in full_transcript if d.get("speaker") == "agent" and "sentiment" in d]
    avg_customer_sentiment = sum(customer_sentiments) / len(customer_sentiments) if customer_sentiments else 0.0
    avg_agent_sentiment = sum(agent_sentiments) / len(agent_sentiments) if agent_sentiments else 0.0
    return avg_agent_sentiment, avg_customer_sentiment

def update_analysis_with_counts(transcript_json: List[dict], analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    entities_list = analysis_dict.get("entity_filler_count", {}).get("entities", [])
    filler_list = analysis_dict.get("entity_filler_count", {}).get("filler_words", [])
    entities_sorted = sorted(entities_list, key=len, reverse=True)
    entity_pattern = re.compile(r'\b(' + '|'.join(re.escape(e) for e in entities_sorted) + r')\b', re.IGNORECASE) if entities_sorted else None
    filler_pattern = re.compile(r'\b(' + '|'.join(re.escape(f) for f in filler_list) + r')\b', re.IGNORECASE) if filler_list else None
    entity_counter = Counter()
    filler_counter = Counter()

    for utterance_obj in transcript_json:
        utterance = utterance_obj.get("utterance", "")
        if entity_pattern:
            entity_matches = entity_pattern.findall(utterance)
            for match in entity_matches:
                for e in entities_list:
                    if match.lower() == e.lower():
                        entity_counter[e] += 1
                        break
        if filler_pattern:
            filler_matches = filler_pattern.findall(utterance)
            for match in filler_matches:
                for f in filler_list:
                    if match.lower() == f.lower():
                        filler_counter[f] += 1
                        break

    analysis_dict.setdefault('entity_filler_count', {})
    analysis_dict['entity_filler_count']['entities'] = dict(entity_counter)
    analysis_dict['entity_filler_count']['filler_words'] = dict(filler_counter)
    return analysis_dict

def total_words_spoken(full_transcript: list) -> Dict[str, int]:
    agent_total = 0
    customer_total = 0
    for dialogue in full_transcript:
        utterance = dialogue.get("utterance", "")
        word_count = len(utterance.split())
        speaker = (dialogue.get("speaker") or "").lower()
        if speaker == "agent":
            agent_total += word_count
        elif speaker == "customer":
            customer_total += word_count
    return {"agent": agent_total, "customer": customer_total}

def match_previous_next_timestamps(full_transcript: list) -> list:
    for i, dialogue in enumerate(full_transcript[:-1]):
        prev_time = full_transcript[i]['end_time']
        next_time = full_transcript[i+1]['start_time']
        if prev_time != next_time:
            full_transcript[i]['end_time'] = next_time
    return full_transcript

def speaking_non_speech_percentage(full_transcript: list) -> dict:
    speaker_time = {}
    non_speech_time = {"hold_time": 0, "noise": 0, "dead_air": 0}
    total_time = 0.0

    def time_to_seconds(time_str):
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(float, parts)
            return m * 60 + s
        else:
            return float(time_str)

    for dialogue in full_transcript:
        start_seconds = time_to_seconds(dialogue["start_time"])
        end_seconds = time_to_seconds(dialogue["end_time"])
        duration = max(0, end_seconds - start_seconds)
        speaker = dialogue.get("speaker")
        if speaker == "non_speech":
            utterance = dialogue.get("utterance", "").lower()
            for key in non_speech_time.keys():
                if key in utterance:
                    non_speech_time[key] += duration
        else:
            speaker_time[speaker] = speaker_time.get(speaker, 0) + duration
        total_time += duration

    if total_time == 0:
        return {"speakers": {}, "non_speech": non_speech_time}

    speaker_percentage = {s: round((t / total_time) * 100, 2) for s, t in speaker_time.items()}
    non_speech_percentage = {k: round((t / total_time) * 100, 2) for k, t in non_speech_time.items()}
    return {"speakers": speaker_percentage, "non_speech": non_speech_percentage}

def loudness_analysis(full_transcript: list):
    speaker_loudness = {}
    speaker_counts = {}
    for d in full_transcript:
        sp = d.get("speaker")
        loud = d.get("loudness")
        if loud is None or sp is None:
            continue
        speaker_loudness[sp] = speaker_loudness.get(sp, 0) + loud
        speaker_counts[sp] = speaker_counts.get(sp, 0) + 1
    if not speaker_loudness:
        return [], {}
    avg = {sp: round(speaker_loudness[sp] / speaker_counts[sp], 2) for sp in speaker_loudness}
    stds = {sp: float(np.std([d["loudness"] for d in full_transcript if d.get("speaker") == sp and d.get("loudness") is not None])) for sp in speaker_loudness}
    deviations = []
    for d in full_transcript:
        sp = d.get("speaker")
        loud = d.get("loudness")
        if loud is None or sp not in avg:
            continue
        if abs(loud - avg[sp]) > 1.5 * stds[sp]:
            deviations.append(d)
    return deviations, avg

# -------------------- LLM Call Wrapper (with retries) --------------------

def llm_generate_with_retries(prompt_text: str,
                              model_name: str,
                              response_schema: dict,
                              job_id: str,
                              max_retries: int = 4,
                              base_backoff: float = 2.0,
                              thinking_budget: int = 100) -> Tuple[str, dict]:
    """
    Generic LLM-call wrapper with retries and special 429 handling.
    Returns (response_text, usage_metadata)
    """
    attempt = 0
    last_exception = None
    total_usage_meta = {
    "input_text_tokens": 0, "input_audio_tokens": 0,
    "output_tokens": 0, "thinking_tokens": 0,
    "cached_text_tokens": 0, "cached_audio_tokens": 0,
    }
    while attempt < max_retries:
        attempt += 1
        try:
            config_kwargs = dict(
                temperature=0,
                top_p=0.9,
                response_mime_type="application/json",
                response_schema=response_schema
            )
            if any(keyword in model_name for keyword in ["2.5-flash", "2.5-pro"]) and "gemini-2.5-flash-lite" not in model_name:
                config_kwargs["thinking_config"] = ThinkingConfig(thinking_budget=thinking_budget)

            response = client.models.generate_content(
                model=model_name,
                contents=[prompt_text],
                config=GenerateContentConfig(**config_kwargs)
            )

            print(f"LLM response finish reason: {response.candidates[0].finish_reason}")
            response_text = extract_transcript(response)
            finish_reason = response.candidates[0].finish_reason
            print(f"LLM response finish reason: {finish_reason}")
            usage = safe_extract_usage_metadata(response)
            for k, v in usage.items():
                total_usage_meta[k] = total_usage_meta.get(k, 0) + v
            # Optionally log to langfuse here if configured
            logger.debug(f"[Job {job_id}] LLM call success (model={model_name}) usage={usage}")
            if finish_reason in (FinishReason.STOP, None):
                logger.debug(f"[Job {job_id}] Successful LLM response.")
                return response_text, total_usage_meta
            # if hit MAX_TOKENS or other reason, force retry
            logger.warning(f"[Job {job_id}] FinishReason={finish_reason}, retrying...")

        except ServerError as e:
            last_exception = e
            # If ServerError contains 429 or RESOURCE_EXHAUSTED, use longer backoff
            err_str = str(e)
            is_rate_limited = ("429" in err_str) or ("RESOURCE_EXHAUSTED" in err_str) or ("rate limit" in err_str.lower())
            if is_rate_limited:
                # Use significantly longer wait on rate limit hits
                wait = 30 * attempt + random.uniform(0, 5)
                logger.warning(f"[Job {job_id}] LLM ServerError -> rate limited. Sleeping {wait:.1f}s before retry {attempt}/{max_retries}. Error: {err_str}")
            else:
                wait = base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.warning(f"[Job {job_id}] LLM ServerError. Sleeping {wait:.1f}s before retry {attempt}/{max_retries}. Error: {err_str}")

            if attempt >= max_retries:
                logger.error(f"[Job {job_id}] LLM call failed after {attempt} attempts. Raising.")
                raise
            time.sleep(wait)

        except Exception as e:
            last_exception = e
            err_str = str(e)
            # Generic exponential backoff
            wait = base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 1)
            logger.warning(f"[Job {job_id}] LLM call exception. Sleeping {wait:.1f}s before retry {attempt}/{max_retries}. Error: {err_str}")
            if attempt >= max_retries:
                logger.error(f"[Job {job_id}] LLM call failed after {attempt} attempts. Raising.")
                raise
            time.sleep(wait)

    # If we exit loop unexpectedly
    logger.error(f"[Job {job_id}] llm_generate_with_retries exhausted: {last_exception}")
    raise last_exception

# -------------------- Combined Components Analysis Functions --------------------

llm_call_count = 0

def combined_components_analysis(full_transcript, model_name, intent_keywords, bool_checks_input, job_id=None, max_retries=3, thinking_budget=100) -> Tuple[dict, dict]:
    """
    Primary combined analysis (single LLM call) that returns parsed JSON and usage metadata.
    """
    global llm_call_count
    llm_call_count += 1
    job_id = job_id or str(uuid.uuid4())
    logger.info(f"[Job {job_id}] Starting combined_components_analysis (model={model_name})")
    start_time = time.time()

    bool_checks_schema = build_boolean_checks_schema(bool_checks_input)
    # inject boolean checks into the response schema copy to avoid mutating global schema
    schema_copy = json.loads(json.dumps(combined_components_response_schema))
    schema_copy["properties"]["boolean_checks"] = bool_checks_schema

    bool_checks_text = build_bool_checks_text(bool_checks_input)
    analysis_prompt = combined_components_prompt.format(
        full_transcript=full_transcript,
        intent_keywords=", ".join(intent_keywords),
        boolean_checks=bool_checks_text,
        crm_fields=", ".join(crm_fields)
    )

    # Use llm_generate_with_retries wrapper
    response_text, usage_metadata = llm_generate_with_retries(
        prompt_text=analysis_prompt,
        model_name=model_name,
        response_schema=schema_copy,
        job_id=job_id,
        max_retries=max_retries,
        thinking_budget=thinking_budget
    )

    # Validate finish reason via response object if needed - here we rely on safe parsing
    parsed = safe_json_parse(response_text, save_path=f"bad_response_{job_id}.json")
    if not parsed:
        # If parse failed, try fallback model (once) then raise
        logger.warning(f"[Job {job_id}] combined_components_analysis: initial JSON parse failed.")
        if model_name != FALLBACK_LLM_MODEL_1:
            logger.info(f"[Job {job_id}] Retrying with fallback model {FALLBACK_LLM_MODEL_1}")
            response_text, usage_metadata_2 = llm_generate_with_retries(
                prompt_text=analysis_prompt,
                model_name=FALLBACK_LLM_MODEL_1,
                response_schema=schema_copy,
                job_id=job_id,
                max_retries=max_retries,
                thinking_budget=100
            )
            parsed = safe_json_parse(response_text, save_path=f"bad_response_fallback_{job_id}.json")
            # Merge usage metadata
            usage_metadata = {k: usage_metadata.get(k, 0) + usage_metadata_2.get(k, 0) for k in set(list(usage_metadata.keys()) + list(usage_metadata_2.keys()))}

    end_time = time.time()
    logger.info(f"[Job {job_id}] combined_components_analysis finished in {end_time - start_time:.2f}s")
    return parsed, usage_metadata

def combined_components_1_analysis(full_transcript, model_name, intent_keywords, job_id=None, max_retries=3, thinking_budget=100):
    global llm_call_count
    llm_call_count += 1
    job_id = job_id or str(uuid.uuid4())
    logger.info(f"[Job {job_id}] Starting combined_components_1_analysis (model={model_name})")
    analysis_prompt = combined_components_1_prompt.format(
        full_transcript=full_transcript,
        intent_keywords=", ".join(intent_keywords),
    )

    schema_copy = json.loads(json.dumps(combined_components_1_response_schema))
    response_text, usage_metadata = llm_generate_with_retries(
        prompt_text=analysis_prompt,
        model_name=model_name,
        response_schema=schema_copy,
        job_id=job_id,
        max_retries=max_retries
    )
    parsed = safe_json_parse(response_text, save_path=f"bad_response_c1_{job_id}.json")
    if not parsed:
        logger.warning(f"[Job {job_id}] combined_components_1_analysis JSON parse failed.")
    return parsed, usage_metadata

def combined_components_2_analysis(full_transcript, model_name, bool_checks_input, job_id=None, max_retries=3, thinking_budget=100):
    global llm_call_count
    llm_call_count += 1
    job_id = job_id or str(uuid.uuid4())
    logger.info(f"[Job {job_id}] Starting combined_components_2_analysis (model={model_name})")

    bool_checks_text = build_bool_checks_text(bool_checks_input)
    analysis_prompt = combined_components_2_prompt.format(
        full_transcript=full_transcript,
        boolean_checks=bool_checks_text
    )

    schema_copy = json.loads(json.dumps(combined_components_2_response_schema))
    bool_checks_schema = build_boolean_checks_schema(bool_checks_input)
    schema_copy["properties"]["boolean_checks"] = bool_checks_schema

    response_text, usage_metadata = llm_generate_with_retries(
        prompt_text=analysis_prompt,
        model_name=model_name,
        response_schema=schema_copy,
        job_id=job_id,
        max_retries=max_retries
    )

    parsed = safe_json_parse(response_text, save_path=f"bad_response_c2_{job_id}.json")
    if not parsed:
        logger.warning(f"[Job {job_id}] combined_components_2_analysis JSON parse failed.")
        return parsed, usage_metadata

    # Post-process entity/filler counts if present
    transcript_text = ". ".join([item.get("utterance", "") for item in full_transcript if "utterance" in item])
    def get_counts(keyword_list, transcript_text):
        return [{"keyword": kw, "count": transcript_text.lower().count(kw.lower())} for kw in keyword_list]

    if "entity_filler_count" in parsed:
        if "entities" in parsed["entity_filler_count"]:
            parsed["entity_filler_count"]["entities"] = get_counts(parsed["entity_filler_count"]["entities"], transcript_text)
        if "filler_words" in parsed["entity_filler_count"]:
            parsed["entity_filler_count"]["filler_words"] = get_counts(parsed["entity_filler_count"]["filler_words"], transcript_text)

    return parsed, usage_metadata

# -------------------- Main analyse() Pipeline --------------------

def analyse(full_transcript: List[dict],
            model_name: str = GOOGLE_LLM_MODEL,
            keywords: Dict[str, List[str]] = None,
            bool_checks_input: List[Dict[str, str]] = None,
            job_id: str = None,
            thinking_budget: int = 100) -> Dict[str, Any]:
    """
    Main analysis pipeline. Returns analysis JSON which includes aggregated usage metadata.
    """
    job_id = job_id or str(uuid.uuid4())
    logger.info(f"[Job {job_id}] Starting analyse pipeline")
    keywords = keywords or {"intent_keywords": []}
    bool_checks_input = bool_checks_input or []

    token_count, _ = count_usage(json.dumps(full_transcript), model_name)
    logger.debug(f"[Job {job_id}] token_count for transcript: {token_count}")

    response_json: Dict[str, Any] = {}
    analysis_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "thinking_tokens": 0,
        "cached_text_tokens": 0,
        "cached_audio_tokens": 0
    }

    # Choose single-call or split approach
    if token_count <= 18000:
        logger.info(f"[Job {job_id}] Using combined analysis (single LLM call)")
        analysis, usage = combined_components_analysis(full_transcript, model_name, keywords.get("intent_keywords", []), bool_checks_input, job_id=job_id, thinking_budget=thinking_budget)
        response_json.update(analysis or {})
        # Update usage safely
        analysis_usage["input_tokens"] += usage.get("input_text_tokens", 0) + usage.get("input_audio_tokens", 0)
        analysis_usage["output_tokens"] += usage.get("output_tokens", 0)
        analysis_usage["total_tokens"] += usage.get("total_token_count", 0)
        t = usage.get("thoughts_token_count")
        t = 0 if t is None else t
        print(f"thinking tokens: {t}")
        analysis_usage["thinking_tokens"] += t 
        analysis_usage["cached_text_tokens"] += usage.get("cached_text_tokens", 0)
        analysis_usage["cached_audio_tokens"] += usage.get("cached_audio_tokens", 0)
    else:
        logger.info(f"[Job {job_id}] Using split analysis (two LLM calls)")
        analysis_1, usage_1 = combined_components_1_analysis(full_transcript, model_name, keywords.get("intent_keywords", []), job_id=job_id, thinking_budget=thinking_budget)
        analysis_2, usage_2 = combined_components_2_analysis(full_transcript, model_name, bool_checks_input, job_id=job_id, thinking_budget=thinking_budget)
        response_json.update(analysis_1 or {})
        response_json.update(analysis_2 or {})
        analysis_usage["input_tokens"] += usage_1.get("input_text_tokens", 0) + usage_2.get("input_text_tokens", 0)
        analysis_usage["output_tokens"] += usage_1.get("output_tokens", 0) + usage_2.get("output_tokens", 0)
        analysis_usage["total_tokens"] += usage_1.get("total_token_count", 0) + usage_2.get("total_token_count", 0)
        t = usage_1.get("thoughts_token_count")
        print(f"thinking tokens for comp 1: {t}")
        analysis_usage["thinking_tokens"] += t if isinstance(t, (int, float)) else 0
        t = usage_2.get("thoughts_token_count")
        print(f"thinking tokens for comp 2: {t}")
        analysis_usage["thinking_tokens"] += t if isinstance(t, (int, float)) else 0
        analysis_usage["cached_text_tokens"] += usage_1.get("cached_text_tokens", 0) + usage_2.get("cached_text_tokens", 0)
        analysis_usage["cached_audio_tokens"] += usage_1.get("cached_audio_tokens", 0) + usage_2.get("cached_audio_tokens", 0)

    # Post-analysis metrics
    response_json["speaking_time"] = speaking_non_speech_percentage(full_transcript)
    deviations, avg_loudness = loudness_analysis(full_transcript)
    response_json["loudness_analysis"] = deviations
    response_json["avg_speaker_loudness"] = avg_loudness
    avg_agent_sentiment, avg_customer_sentiment = average_sentiment(full_transcript)
    response_json["average_sentiment"] = {
        "agent": round(avg_agent_sentiment, 2),
        "customer": round(avg_customer_sentiment, 2)
    }
    response_json["total_words_spoken"] = total_words_spoken(full_transcript)
    response_json["analysis_usage_metadata"] = analysis_usage
    response_json["job_id"] = job_id
    logger.info(f"[Job {job_id}] analyse pipeline complete")
    with open(f"outputs/{job_id}/analysis.json", "w", encoding="utf-8") as f:
        json.dump(response_json, f, indent=2, ensure_ascii=False)
        logger.info(f"[Job {job_id}] Full analysis response saved to {job_id}_analysis.json")
    return response_json

# # -------------------- CLI / Script Entrypoint --------------------

# if __name__ == "__main__":
#     # Example usage for local debugging
#     sample_transcript_path = os.getenv("SAMPLE_TRANSCRIPT_FILE", "/home/runo/final-test/data/output/audio30 (1)_transcript.json")
#     if not os.path.exists(sample_transcript_path):
#         print("Please set SAMPLE_TRANSCRIPT_FILE env or ensure the sample file exists:", sample_transcript_path)
#         raise SystemExit(1)

#     with open(sample_transcript_path, "r", encoding="utf-8") as fh:
#         full_transcript = json.load(fh)

#     keywords = {"intent_keywords": ["product features", "trial demo", "schedule call"]}
#     bool_checks_input = [
#     {"field": "agent_introduced", "type": "boolean", "description": "Did the agent introduce themselves properly?"},
#     {"field": "agent_closed_properly", "type": "boolean", "description": "Did the agent close the call properly?"},
#     {"field": "issue_resolved", "type": "boolean", "description": "Was the customer issue resolved?"},
#     {"field": "feedback_requested", "type": "boolean", "description": "Did the agent request feedback from the customer?"},
#     {"field": "verification", "type": "qa", "description": "Did the agent verify customer information correctly?"},
#     {"field": "objection_handling", "type": "qa", "description": "Did the agent handle objections properly?"},
#     {"field": "acknowledgement_proactiveness", "type": "qa", "description": "Did the agent proactively acknowledge customer's concerns?"},
#     {"field": "empathy", "type": "qa", "description": "Did the agent show empathy during the call?"},
#     {"field": "effective_listening", "type": "qa", "description": "Did the agent demonstrate effective listening?"},
#     {"field": "slang_jargons", "type": "qa", "description": "Did the agent avoid using slang or jargons?"},
#     {"field": "unprofessional_speech", "type": "qa", "description": "Did the agent avoid unprofessional speech?"},
#     {"field": "abrupt_call_disconnection", "type": "qa", "description": "Was the call disconnected abruptly?"},
#     {"field": "rude_on_call", "type": "qa", "description": "Did the agent appear rude on the call?"},
#     {"field": "relevant_information", "type": "qa", "description": "Did the agent provide relevant information?"},
#     {"field": "probing_general", "type": "qa", "description": "Did the agent ask general probing questions?"},
#     {"field": "customer_fcr", "type": "qa", "description": "Was the customer's First Call Resolution achieved?"},
#     ]

#     job_id = str(uuid.uuid4())
#     result = analyse(full_transcript, model_name=GOOGLE_LLM_MODEL, keywords=keywords, bool_checks_input=bool_checks_input, job_id=job_id)

#     out_path = Path(sample_transcript_path).with_name(f"analysis_result_{job_id}.json")
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)

#     print(f"Analysis saved → {out_path}")
