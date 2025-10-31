import os
import io
import json
import time
import random
import uuid
import logging
import ffmpeg
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.genai as genai
from google.genai.types import Part, GenerateContentConfig, ThinkingConfig, FinishReason
from pydub import AudioSegment
from .prompt import transcript_prompt as prompt
from .response_schema import response_schema

# ------------------ Setup Logging ------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ------------------ Load API Key ------------------
load_dotenv()
GCP_API_KEY = os.getenv("GOOGLE_API_KEY")
print(f"GCP_API_KEY: {GCP_API_KEY}")
client = genai.Client(api_key=GCP_API_KEY)

# ------------------ Utilities ------------------
def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"

def safe_json_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None

# ------------------ Audio Splitting ------------------
def split_audio_ffmpeg(input_file, chunk_length_sec=360):
    """
    Splits an audio file into fixed-length chunks using FFmpeg.
    Returns a list of tuples: (chunk_path, start_time_sec, end_time_sec)
    and the total duration of the original file.
    """
    out_chunks = []
    audio = AudioSegment.from_file(input_file)
    duration = len(audio) / 1000  # seconds
    base_name, ext = os.path.splitext(os.path.basename(input_file))

    os.makedirs("temp_chunks", exist_ok=True)

    for i, start in enumerate(range(0, int(duration), chunk_length_sec)):
        remaining = duration - start
        this_chunk_len = min(chunk_length_sec, remaining)
        end_time = start + this_chunk_len

        out_file = os.path.join("temp_chunks", f"{base_name}_chunk{i+1}.mp3")

        (
            ffmpeg
            .input(input_file, ss=start, t=this_chunk_len)
            .output(out_file)
            .run(overwrite_output=True, quiet=True)
        )

        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            out_chunks.append((out_file, float(start), float(end_time)))
        else:
            logger.warning(f"Chunk {i+1} not created or empty → skipped")

    return out_chunks, duration


# ------------------ Response Extractors ------------------
def extract_transcript(response):
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

def safe_extract_usage_metadata(response):
    usage_meta = {}
    try:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            logger.warning("⚠️ No usage_metadata found in response")
            return usage_meta

        input_text_tokens = 0
        input_audio_tokens = 0
        cached_text_tokens = 0
        cached_audio_tokens = 0

        prompt_details = getattr(usage, "prompt_tokens_details", [])
        cache_details = getattr(usage, "cache_tokens_details", [])
        thinking_tokens = getattr(usage, "thoughts_token_count", 0)
        print("thinking tokens:", thinking_tokens)
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
                i.token_count for i in cache_details if getattr(i, "modality", None) and i.modality.name == "TEXT"
            )
            cached_audio_tokens = sum(
                i.token_count for i in cache_details if getattr(i, "modality", None) and i.modality.name == "AUDIO"
            )
        if thinking_tokens is None:
            thinking_tokens=0
        usage_meta = {
            "input_text_tokens": input_text_tokens,
            "input_audio_tokens": input_audio_tokens,
            "output_tokens": getattr(usage, "candidates_token_count", 0),
            "thinking_tokens": thinking_tokens,
            "cached_text_tokens": cached_text_tokens,
            "cached_audio_tokens": cached_audio_tokens,
        }

    except Exception as e:
        logger.warning(f"⚠️ Error extracting usage metadata safely: {e}")
    return usage_meta

# ------------------ Transcription with Retry ------------------
def transcribe_chunk(audio_file_path, chunk_number, chunk_length_sec, total_duration, start_time, end_time, model_name="gemini-2.5-flash", job_id=None, max_retries=3, thinking_budget=100):
    with open(audio_file_path, "rb") as f:
        byte_data = f.read()

    audio_content = Part.from_bytes(
        data=byte_data,
        mime_type="audio/mpeg" if audio_file_path.endswith(".wav") else "audio/mp3"
    )
    total_usage_meta = {
        "input_text_tokens": 0, "input_audio_tokens": 0,
        "output_tokens": 0, "thinking_tokens": 0,
        "cached_text_tokens": 0, "cached_audio_tokens": 0,
    }
    attempt = 0
    while attempt < max_retries:
        try:
            logger.debug(f"[Job {job_id}] [Chunk {chunk_number}] Sending to Gemini ({attempt+1}/{max_retries})")

            config_kwargs = dict(
                temperature=0,
                top_p=0.9,
                response_mime_type="application/json",
                response_schema=response_schema,
                #max_output_tokens=8192,
            )
            if any(keyword in model_name for keyword in ["2.5-flash", "2.5-pro"]) and "gemini-2.5-flash-lite" not in model_name:
                print("============Adding thinking config================")
                config_kwargs["thinking_config"] = ThinkingConfig(thinking_budget=thinking_budget)

            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, audio_content],
                config=GenerateContentConfig(**config_kwargs),
            )
            finish_reason = getattr(response.candidates[0], "finish_reason", None)
            if finish_reason != FinishReason.STOP:
                print("Response:", response)
                raise Exception(f"Non-STOP finish reason: {finish_reason}")

            text_data = response.candidates[0].content.parts[0].text
            transcript_json = json.loads(text_data) if text_data else []
            chunk_start_seconds = start_time
            for dialogue in transcript_json:
                for timestamp_key in ["start_time", "end_time"]:
                    time_str: str = dialogue[timestamp_key]
                    parts = time_str.split(':')
                    if len(parts) == 3:  # HH:MM:SS
                        timestamp_secs = (int(parts[0]) * 3600) + (int(parts[1]) * 60) + int(parts[2]) + chunk_start_seconds
                    elif len(parts) == 2:  # MM:SS
                        timestamp_secs = int(parts[0]) * 60 + int(parts[1]) + chunk_start_seconds
                    else:
                        raise ValueError("Invalid timestamp format")
                    
                    dialogue[timestamp_key] = f"{int(timestamp_secs/3600):02d}:{int((timestamp_secs % 3600)/60):02d}:{int(timestamp_secs % 60):02d}"


            usage_meta = safe_extract_usage_metadata(response)
            for k, v in usage_meta.items():
                total_usage_meta[k] = total_usage_meta.get(k, 0) + v
            output_dir = os.path.join("outputs", job_id)
            os.makedirs(output_dir, exist_ok=True)

            chunk_file = os.path.join(output_dir, f"chunk_{chunk_number}.json")
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(transcript_json, f, indent=2, ensure_ascii=False)

            logger.info(f"[Job {job_id}] [Chunk {chunk_number}] Transcript saved → {chunk_file}")
            return {
                "chunk_result": {
                    "chunk_number": chunk_number,
                    "start_time": format_time(start_time),
                    "end_time": format_time(end_time),
                    "entries": transcript_json,
                },
                "usage_metadata": total_usage_meta,
            }

        except Exception as e:
            attempt += 1
            error_message = str(e)
            print(f"[Job {job_id}] [Chunk {chunk_number}] Error: {error_message}")
            try:
                if "response" in locals() and response:
                    usage = safe_extract_usage_metadata(response)
                    for k, v in usage.items():
                        total_usage_meta[k] = total_usage_meta.get(k, 0) + v
            except Exception:
                pass  # ignore if response is not available 
            is_rate_limited = "429" in error_message or "RESOURCE_EXHAUSTED" in error_message
            sleep_time = (30 if is_rate_limited else 2 ** attempt) + random.uniform(0, 5)
            logger.warning(f"[Job {job_id}] [Chunk {chunk_number}] Retry {attempt}/{max_retries} after {sleep_time:.1f}s ({error_message})")
            if attempt < max_retries:
                time.sleep(sleep_time)
            
            else:
                logger.error(f"[Job {job_id}] [Chunk {chunk_number}] ❌ Failed after {max_retries} attempts.")
                return {
                    "chunk_result": {
                        "chunk_number": chunk_number,
                        "start_time": format_time(start_time),
                        "end_time": format_time(end_time),
                        "entries": [],
                    },
                    "usage_metadata": total_usage_meta,
                }


# ------------------ Adjust Final Timestamps ------------------
def adjust_chunk_timestamps(transcripts):
    corrected_entries = []
    for chunk in transcripts:
        if "entries" in chunk and chunk["entries"]:
            for e in chunk["entries"]:
                corrected_entries.append({
                    "start_time": e.get("start_time", chunk["start_time"]),
                    "end_time": e.get("end_time", chunk["end_time"]),
                    "speaker": e.get("speaker"),
                    "utterance": e.get("utterance") or e.get("content") or "",
                    "loudness": e.get("loudness"),
                    "sentiment": e.get("sentiment")
                })
        else:
            corrected_entries.append({
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "speaker": None,
                "utterance": chunk.get("raw_text", ""),
                "loudness": None,
                "sentiment": None
            })
    return corrected_entries

# ------------------ Main Transcription Function ------------------
def transcribe_audio_parallel(audio_path, model_name, job_id, chunk_length_sec=360, max_workers=4, thinking_budget=100):
    
    logger.info(f"🎧 Starting transcription job: {job_id}")

    output_dir = os.path.join("outputs", job_id)
    os.makedirs(output_dir, exist_ok=True)

    chunk_files, duration = split_audio_ffmpeg(audio_path, chunk_length_sec)
    logger.info(f"Created {len(chunk_files)} chunks (total duration {format_time(duration)}).")

    transcripts = []
    input_text_tokens = input_audio_tokens = total_output_tokens = 0
    cached_text_tokens = cached_audio_tokens = thinking_tokens = 0
    results_with_time = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(
                transcribe_chunk,
                chunk_path, idx+1, chunk_length_sec, duration, start_time, end_time, model_name, job_id, thinking_budget=thinking_budget
            ): (chunk_path, start_time, end_time)
            for idx, (chunk_path, start_time, end_time) in enumerate(chunk_files)
        }
        results_with_time = []

        for future in as_completed(future_to_chunk):
            chunk_number, start_time, end_time = future_to_chunk[future]
            try:
                result = future.result()
                if not result:
                    logger.warning(f"[Job {job_id}] [Chunk {chunk_number}] Empty result — skipped.")
                    continue
                results_with_time.append((result, start_time, end_time))
            except Exception as e:
                logger.error(f"[Job {job_id}] Error in chunk {chunk_number}: {e}")
                continue
            
            usage = result.get("usage_metadata") or {}
            print(usage)
            input_text_tokens += usage.get("input_text_tokens", 0)
            input_audio_tokens += usage.get("input_audio_tokens", 0)
            total_output_tokens += usage.get("output_tokens", 0)
            t = usage.get("thinking_tokens")
            thinking_tokens += 0 if t is None else t
            cached_text_tokens += usage.get("cached_text_tokens", 0)
            cached_audio_tokens += usage.get("cached_audio_tokens", 0)

        results_with_time.sort(key=lambda x: x[1])
        transcripts = [t[0] for t in results_with_time]
        full_transcript_entries = []
        for chunk in transcripts:
            full_transcript_entries.extend(chunk["chunk_result"]["entries"])
        full_transcript_entries = sorted(
            full_transcript_entries,
            key=lambda e: e.get("start_time", "00:00:00")
        )
        final_json = {
            "job_id": job_id,
            "model": model_name,
            "audio_metadata": {
                "duration_sec": duration,
                "num_chunks": len(chunk_files),
            },
            "usage_metadata": {
                "input_text_tokens": input_text_tokens,
                "input_audio_tokens": input_audio_tokens,
                "output_tokens": total_output_tokens,
                "thinking_tokens": thinking_tokens,
                "cached_text_tokens": cached_text_tokens,
                "cached_audio_tokens": cached_audio_tokens,
            },
            "chunks": [c["chunk_result"] for c in transcripts],
            "transcript": full_transcript_entries,
        }
        final_file = os.path.join(output_dir, "final_transcript.json")
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Final transcript saved → {final_file}")
        return final_json

# # ------------------ Script Entry Point ------------------
# if __name__ == "__main__":
#     audio_path = "/path/to/your/audiofile.mp3"
#     model = "gemini-2.5-flash"
#     result = transcribe_audio_parallel(audio_path, model)
#     print(json.dumps(result, indent=2, ensure_ascii=False))
