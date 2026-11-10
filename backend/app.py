"""音乐实验室工作站 — backend.

Serves the studio frontend at ``/`` and the AI tools under ``/api``. Long jobs run
one at a time in a worker thread (they share one GPU); the frontend polls
``/api/jobs/{id}``.

Run:  uvicorn backend.app:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import engines

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("MUSICLAB_DATA", ROOT / "data"))
JOBS_DIR = DATA / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = int(os.environ.get("MUSICLAB_MAX_UPLOAD_MB", "200"))
JOB_TTL_S = int(os.environ.get("MUSICLAB_JOB_TTL", str(6 * 3600)))

app = FastAPI(title="音乐实验室工作站 API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("MUSICLAB_CORS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pool = ThreadPoolExecutor(max_workers=int(os.environ.get("MUSICLAB_WORKERS", "1")))
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _public(job: dict) -> dict:
    return {k: job[k] for k in ("id", "kind", "status", "message", "result", "error", "created")}


def _submit(kind: str, fn, *args) -> dict:
    jid = uuid.uuid4().hex[:12]
    out_dir = JOBS_DIR / jid
    out_dir.mkdir(parents=True)
    job = {"id": jid, "kind": kind, "status": "queued", "message": "排队中…", "result": None, "error": None,
           "created": time.time(), "dir": out_dir}
    with _lock:
        _jobs[jid] = job

    def progress(msg: str):
        job["message"] = msg

    def run():
        job["status"] = "running"
        job["message"] = "处理中…"
        try:
            res = fn(*args, out_dir=out_dir, progress=progress)
            # turn bare file names into download URLs
            for item in res.get("stems", []):
                item["url"] = f"/api/files/{jid}/{item['file']}"
            if "file" in res:
                res["url"] = f"/api/files/{jid}/{res['file']}"
            job["result"] = res
            job["status"] = "done"
            job["message"] = "完成"
        except Exception as e:  # noqa: BLE001 - report every failure to the client
            traceback.print_exc()
            job["status"] = "error"
            job["error"] = str(e) or e.__class__.__name__
            job["message"] = "失败"

    _pool.submit(run)
    _cleanup()
    return _public(job)


def _cleanup():
    now = time.time()
    with _lock:
        for jid in [j for j, v in _jobs.items() if now - v["created"] > JOB_TTL_S and v["status"] in ("done", "error")]:
            shutil.rmtree(_jobs[jid]["dir"], ignore_errors=True)
            del _jobs[jid]


async def _save_upload(up: UploadFile) -> Path:
    suffix = Path(up.filename or "audio.wav").suffix or ".wav"
    dest = DATA / "uploads" / f"{uuid.uuid4().hex}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dest.open("wb") as f:
        while chunk := await up.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"文件超过 {MAX_UPLOAD_MB} MB")
            f.write(chunk)
    return dest


def _require(feature: str):
    if not engines.features().get(feature):
        raise HTTPException(503, f"服务器没有安装「{feature}」所需的模型，请看 README 的部署说明")


# ---------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    return {"ok": True, "features": engines.features()}


@app.post("/api/separate")
async def separate(file: UploadFile = File(...), mode: str = Form("vocals")):
    _require("separate")
    if mode not in ("vocals", "4stems"):
        raise HTTPException(400, "mode 只能是 vocals 或 4stems")
    src = await _save_upload(file)
    return _submit("separate", lambda out_dir, progress: engines.separate(src, out_dir, mode, progress))


@app.post("/api/align")
async def align(file: UploadFile = File(...), lyrics: str = Form(...), language: str = Form("zh"),
                vocals_first: bool = Form(True)):
    _require("align")
    lines = json.loads(lyrics) if lyrics.strip().startswith("[") else lyrics.splitlines()
    src = await _save_upload(file)
    return _submit("align", lambda out_dir, progress: engines.align(src, lines, language, vocals_first, out_dir, progress))


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    _require("transcribe")
    src = await _save_upload(file)
    return _submit("transcribe", lambda out_dir, progress: engines.transcribe(src, progress))


class GenerateReq(BaseModel):
    prompt: str = ""
    lyrics: str = ""
    duration: float = 30
    bpm: int | None = None
    key: str | None = None
    language: str = "zh"


@app.post("/api/generate")
def generate(req: GenerateReq):
    _require("generate")
    return _submit("generate", lambda out_dir, progress: engines.generate(req.model_dump(), out_dir, progress))


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = _jobs.get(jid)
    if not j:
        raise HTTPException(404, "找不到这个任务（可能已过期）")
    return _public(j)


@app.get("/api/files/{jid}/{name}")
def files(jid: str, name: str):
    j = _jobs.get(jid)
    if not j or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(404, "文件不存在")
    path = j["dir"] / name
    if not path.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(path, media_type="audio/wav", filename=name)


# the studio itself
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
