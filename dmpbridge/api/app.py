"""HTTP API for the DMP Bridge web app: a DMP PDF in, the merged DMP JSON out.

    POST /jobs        upload a PDF (form field ``file``); answers at once with the job id
    GET  /jobs/{id}   the job's status, and its result once it is done
    GET  /up          health check: the server is up and Ollama has both models

Run it:

    pip install -e ".[server]"
    uvicorn dmpbridge.api.app:app --port 8000

Settings are environment variables (or a ``.env`` file at the repository root):

    DMPBRIDGE_HOST              Ollama URL                        http://localhost:11434
    DMPBRIDGE_NARRATIVE_MODEL   model for stages 2-4              gemma4:e4b
    DMPBRIDGE_RDA_MODEL         model for stage 5                 gemma4:e4b
    DMPBRIDGE_FALLBACK          extractor for an unreadable PDF   auto (LightOnOCR); none to turn off
    DMPBRIDGE_ATTEMPTS          tries per model step              3
    DMPBRIDGE_TIMEOUT_SECONDS   longest wait for one model answer 180
    DMPBRIDGE_MAX_UPLOAD_MB     largest PDF accepted              20
    DMPBRIDGE_JOB_TTL_MINUTES   how long a finished job is kept   60
    DMPBRIDGE_CORS_ORIGINS      web app origins, comma-separated  http://localhost:3000
"""
import os
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core import config
from ..models.ollama import OllamaModel
from ..utils import setup_logging
from .jobs import JobQueue, QueueFull
from .pipeline import make_runner

setup_logging()

HOST = config.HOST
NARRATIVE_MODEL = os.getenv("DMPBRIDGE_NARRATIVE_MODEL", "gemma4:e4b")
RDA_MODEL = os.getenv("DMPBRIDGE_RDA_MODEL", "gemma4:e4b")
FALLBACK = os.getenv("DMPBRIDGE_FALLBACK", "auto")
ATTEMPTS = int(os.getenv("DMPBRIDGE_ATTEMPTS", "3"))
TIMEOUT_SECONDS = int(os.getenv("DMPBRIDGE_TIMEOUT_SECONDS", "180"))
MAX_UPLOAD_BYTES = int(os.getenv("DMPBRIDGE_MAX_UPLOAD_MB", "20")) * 1024 * 1024
JOB_TTL_SECONDS = int(os.getenv("DMPBRIDGE_JOB_TTL_MINUTES", "60")) * 60
CORS_ORIGINS = [o.strip() for o in os.getenv("DMPBRIDGE_CORS_ORIGINS", "http://localhost:3000").split(",")
                if o.strip()]

# One model answer normally takes under a minute. The command line waits up to an hour;
# here a stuck answer is cut off and retried.
OllamaModel.timeout = TIMEOUT_SECONDS

jobs = JobQueue(
    run=make_runner(NARRATIVE_MODEL, RDA_MODEL, HOST,
                    fallback=None if FALLBACK == "none" else FALLBACK, attempts=ATTEMPTS),
    ttl=JOB_TTL_SECONDS,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    jobs.start()
    yield


app = FastAPI(title="DMP Bridge API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"])


@app.post("/jobs", status_code=202)
async def create_job(file: UploadFile = File(...)):
    # Read one byte past the limit, so a larger file is caught without reading all of it.
    pdf = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(pdf) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"The PDF is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not pdf.startswith(b"%PDF-"):
        raise HTTPException(415, "This file isn't a PDF.")
    try:
        job = jobs.submit(pdf)
    except QueueFull:
        raise HTTPException(503, "The server is busy. Please try again in a few minutes.") from None
    return {"id": job.id, "status": job.status}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "This job doesn't exist or has expired. Please upload the PDF again.")
    return {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "position": jobs.position(job),
        "error": job.error,
        "result": job.result,
    }


@app.get("/up")
def up():
    try:
        tags = requests.get(f"{HOST}/api/tags", timeout=5).json()
    except (requests.exceptions.RequestException, ValueError):
        return JSONResponse({"status": "Ollama isn't reachable", "ollama": HOST}, status_code=503)

    pulled = {m.get("name") for m in tags.get("models", [])}
    missing = [m for m in {NARRATIVE_MODEL, RDA_MODEL}
               if (m if ":" in m else f"{m}:latest") not in pulled]
    if missing:
        return JSONResponse({"status": f"Model not pulled in Ollama: {', '.join(sorted(missing))}"},
                            status_code=503)
    return {"status": "ok", "waiting": jobs.waiting()}
