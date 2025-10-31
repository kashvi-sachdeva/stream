import uuid
import tempfile
import time
import json
import logging
from pathlib import Path
from processor import process_audio_file
from fastapi import FastAPI, Form, HTTPException, UploadFile, File

 #-------------------- Setup --------------------
app = FastAPI(title="Call Analytics API", version="1.0.0")

AUDIO_FILES_DIR = Path("audios")
AUDIO_FILES_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("call_analytics_api")

SUPPORTED_MODELS = {
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro"
}

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg', '.wma'}

# -------------------- FastAPI Endpoints --------------------

@app.post("/process_audio/")
async def process_audio(
    filename: str = Form(...),
    model: str = Form("gemini-2.5-flash"),
    thinking_budget: int = Form(100)
):
    """
    Process an audio file from local directory by filename.
    """
    job_id = str(uuid.uuid4())
    logger.info(f"[Job {job_id}] Request received for file '{filename}'")

    try:
        file_path = AUDIO_FILES_DIR / filename
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found in {AUDIO_FILES_DIR}")

        if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_path.suffix}")

        with open(file_path, "rb") as f:
            contents = f.read()

        result = process_audio_file(contents, filename, model, job_id=job_id, thinking_budget=thinking_budget)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[Job {job_id}] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_audio_upload/")
async def process_audio_upload(
    file: UploadFile = File(...),
    model: str = Form("gemini-2.5-flash"),
    thinking_budget: int = Form(100)
):
    """
    Upload and process an audio file directly.
    """
    job_id = str(uuid.uuid4())
    logger.info(f"[Job {job_id}] Upload received: {file.filename}")

    try:
        contents = await file.read()

        # Save uploaded file for traceability
        save_path = AUDIO_FILES_DIR / file.filename
        with open(save_path, "wb") as f:
            f.write(contents)
        logger.info(f"[Job {job_id}] File saved at {save_path}")

        result = process_audio_file(contents, file.filename, model, job_id=job_id, thinking_budget=thinking_budget)
        return result

    except Exception as e:
        logger.exception(f"[Job {job_id}] Upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check if API is running"""
    return {
        "status": "ok",
        "audio_dir": str(AUDIO_FILES_DIR),
        "audio_dir_exists": AUDIO_FILES_DIR.exists(),
        "audio_dir_is_dir": AUDIO_FILES_DIR.is_dir(),
        "supported_models": list(SUPPORTED_MODELS)
    }


@app.get("/list_files")
async def list_audio_files():
    """List available audio files"""
    try:
        if not AUDIO_FILES_DIR.exists():
            return {"error": f"Directory {AUDIO_FILES_DIR} does not exist"}

        files = [
            f.name for f in AUDIO_FILES_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        ]

        return {
            "directory": str(AUDIO_FILES_DIR),
            "file_count": len(files),
            "files": sorted(files)
        }

    except Exception as e:
        logger.error(f"File listing failed: {e}")
        return {"error": str(e)}
