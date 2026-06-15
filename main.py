import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

app = FastAPI(title="dmpbridge", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

TEMPLATE = Path("templates/index.html")


@app.get("/", response_class=HTMLResponse)
async def index():
    return TEMPLATE.read_text(encoding="utf-8")


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")
    dest = UPLOAD_DIR / "current.pdf"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"url": "/uploads/current.pdf", "name": file.filename}


@app.post("/upload/json")
async def upload_json(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".json"):
        raise HTTPException(400, "File must be JSON")
    dest = UPLOAD_DIR / "current.json"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"url": "/uploads/current.json", "name": file.filename}


@app.get("/uploads/{filename}")
async def serve_upload(filename: str):
    path = UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path)
