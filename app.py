"""音乐实验室工作站 — Hugging Face Space entry point (also runs locally).

Serves the studio page at ``/`` and the AI tools as Gradio API endpoints
(``/separate``, ``/align``, ``/transcribe``, ``/generate``, ``/health``). The
endpoints go through Gradio's queue, so on a ZeroGPU Space each call borrows a
GPU only while it runs and is billed to the caller's Hugging Face quota.

    python app.py                    # http://localhost:7860
    MUSICLAB_MOCK=1 python app.py    # demo mode, no models needed
"""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from gradio import Server  # noqa: E402
from gradio.data_classes import FileData  # noqa: E402

from backend import engines  # noqa: E402

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
MAX_STEMS = 6

try:  # present on Hugging Face Spaces
    import spaces

    GPU = spaces.GPU
except ImportError:  # local machine: run the function as is

    def GPU(fn=None, **_):
        return fn if fn is not None else (lambda f: f)


app = Server(title="音乐实验室工作站")


def _job_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="musiclab-"))


def _path(f: FileData | dict) -> Path:
    return Path(f["path"] if isinstance(f, dict) else f.path)


# --- GPU work: each runs inside one ZeroGPU allocation --------------------------
@GPU(duration=150)
def _gpu_separate(src: str, out_dir: str, mode: str) -> dict:
    return engines.separate(Path(src), Path(out_dir), mode)


@GPU(duration=180)
def _gpu_align(src: str, lines: list[str], language: str, vocals_first: bool, out_dir: str) -> dict:
    return engines.align(Path(src), lines, language, vocals_first, Path(out_dir))


@GPU(duration=180)
def _gpu_generate(opts: dict, out_dir: str) -> dict:
    return engines.generate(opts, Path(out_dir))


# --- API endpoints ------------------------------------------------------------------
@app.api(name="health")
def health() -> dict:
    """Which AI tools this server can run."""
    return {"ok": True, "features": engines.features()}


@app.api(name="separate", concurrency_limit=1)
def separate(
    audio: FileData, mode: str = "vocals"
) -> tuple[dict, FileData | None, FileData | None, FileData | None, FileData | None, FileData | None, FileData | None]:
    """Split audio into stems (mode "vocals" → 人声+伴奏, "4stems" → 人声/鼓/贝斯/其他, "6stems" → 再加吉他/钢琴).

    Returns stem info followed by up to six WAV files in the same order.
    """
    if mode not in ("vocals", "4stems", "6stems"):
        raise ValueError("mode 只能是 vocals、4stems 或 6stems")
    print(f"[separate] mode={mode} file={_path(audio).stat().st_size // 1024} KB", flush=True)
    out = _job_dir()
    res = _gpu_separate(str(_path(audio)), str(out), mode)
    stems = res["stems"][:MAX_STEMS]
    files = [FileData(path=str(out / s["file"])) for s in stems]
    files += [None] * (MAX_STEMS - len(files))
    return ({"stems": [{"name": s["name"], "label": s["label"]} for s in stems]}, *files)


@app.api(name="align", concurrency_limit=1)
def align(audio: FileData, lyrics: str, language: str = "zh", vocals_first: bool = True) -> dict:
    """Time each lyric line. ``lyrics`` is a JSON list of lines or plain text, one line per row."""
    lines = json.loads(lyrics) if lyrics.strip().startswith("[") else lyrics.splitlines()
    return _gpu_align(str(_path(audio)), lines, language, vocals_first, str(_job_dir()))


@app.api(name="transcribe", concurrency_limit=2)
def transcribe(audio: FileData) -> dict:
    """Audio → MIDI notes [{p, s, d, v}] (pitch, start s, duration s, velocity 0–1)."""
    return engines.transcribe(_path(audio))


@app.api(name="generate", concurrency_limit=1)
def generate(prompt: str, lyrics: str = "", duration: float = 30, bpm: int = 0, language: str = "zh") -> tuple[dict, FileData]:
    """Make music from a style prompt (and lyrics, when ACE-Step is connected)."""
    out = _job_dir()
    res = _gpu_generate({"prompt": prompt, "lyrics": lyrics, "duration": duration, "bpm": bpm or None, "language": language}, str(out))
    return {k: v for k, v in res.items() if k != "file"}, FileData(path=str(out / res["file"]))


# --- the studio page ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return (FRONTEND / "index.html").read_text(encoding="utf-8")


@app.get("/vendor/gradio-client.js")
def gradio_client_js():
    return FileResponse(FRONTEND / "vendor" / "gradio-client.min.js", media_type="text/javascript")


if __name__ == "__main__":
    app.launch(server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", "7860")), show_error=True,
               ssr_mode=False)  # Spaces turn SSR on; its Node proxy breaks the API calls from the studio page
