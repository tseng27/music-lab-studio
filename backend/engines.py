"""AI engines behind the studio's 「AI 工具」 panel.

Every engine takes an input file path plus options, writes any audio it makes into
``out_dir`` and returns a JSON-serialisable result. Heavy libraries are imported
lazily so the server starts (and the frontend works) even when some are missing.

Set ``MUSICLAB_MOCK=1`` to replace every engine with a fast synthetic stand-in;
that is how the test suite and a GPU-less demo run.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

Progress = Callable[[str], None]

MOCK = os.environ.get("MUSICLAB_MOCK") == "1"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")
MUSICGEN_MODEL = os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-small")
ACESTEP_URL = os.environ.get("ACESTEP_URL", "").rstrip("/")
ACESTEP_API_KEY = os.environ.get("ACESTEP_API_KEY", "")


def has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def features() -> dict:
    if MOCK:
        return {"separate": True, "align": True, "transcribe": True, "generate": "mock", "mock": True}
    gen = "acestep" if ACESTEP_URL else ("musicgen" if has("transformers") and has("torch") else None)
    return {
        "separate": has("demucs"),
        "align": has("stable_whisper"),
        "transcribe": has("basic_pitch"),
        "generate": gen,
        "mock": False,
    }


# ---------------------------------------------------------------- separation
STEM_LABELS = {"vocals": "人声", "no_vocals": "伴奏", "drums": "鼓", "bass": "贝斯", "other": "其他乐器", "guitar": "吉他", "piano": "钢琴"}


def separate(src: Path, out_dir: Path, mode: str = "vocals", progress: Progress = print) -> dict:
    """Split a mix into stems. mode = "vocals" (人声 + 伴奏) or "4stems" (人声/鼓/贝斯/其他)."""
    if MOCK:
        return _mock_separate(src, out_dir, mode)
    progress(f"Demucs ({DEMUCS_MODEL}) 正在分离…")
    cmd = [sys.executable, "-m", "demucs", "-n", DEMUCS_MODEL, "-o", str(out_dir / "_demucs"), "--filename", "{stem}.{ext}"]
    if mode == "vocals":
        cmd += ["--two-stems", "vocals"]
    cmd.append(str(src))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("Demucs 失败：" + proc.stderr[-800:])
    stems = []
    for wav in sorted((out_dir / "_demucs").rglob("*.wav")):
        dest = out_dir / wav.name
        wav.replace(dest)
        stems.append({"name": wav.stem, "label": STEM_LABELS.get(wav.stem, wav.stem), "file": dest.name})
    if not stems:
        raise RuntimeError("Demucs 没有产出任何分轨")
    return {"stems": stems}


# ---------------------------------------------------------------- lyric alignment
_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def _norm(s: str) -> str:
    return _PUNCT.sub("", s).lower()


def lines_from_words(lines: list[str], words: list[tuple[float, float, str]]) -> list[dict]:
    """Map word timings from the aligner back onto the user's lyric lines.

    Walks the words in order, consuming normalised characters until each line's
    characters are used up; a line starts at the first word that contributes to it.
    """
    out: list[dict] = []
    wi, carry = 0, ""  # carry = characters of the current word not yet consumed
    cur_start = cur_end = None
    for line in lines:
        need = len(_norm(line))
        start = end = None
        while need > 0 and (carry or wi < len(words)):
            if not carry:
                ws, we, wt = words[wi]
                wi += 1
                carry = _norm(wt)
                cur_start, cur_end = ws, we
                if not carry:
                    continue
            if start is None:
                start = cur_start
            take = min(need, len(carry))
            carry, need = carry[take:], need - take
            end = cur_end
        out.append({"text": line, "start": None if start is None else round(float(start), 3),
                    "end": None if end is None else round(float(end), 3)})
    return out


_whisper = None


def align(src: Path, lyrics: list[str], language: str = "zh", vocals_first: bool = True, out_dir: Path | None = None,
          progress: Progress = print) -> dict:
    lyrics = [l.strip() for l in lyrics if l.strip()]
    if not lyrics:
        raise ValueError("没有收到歌词")
    if MOCK:
        dur = sf.info(str(src)).duration
        step = dur / len(lyrics)
        return {"lines": [{"text": l, "start": round(i * step, 3), "end": round((i + 1) * step - 0.1, 3)} for i, l in enumerate(lyrics)]}
    audio = src
    if vocals_first and has("demucs") and out_dir is not None:
        progress("先分离人声，让对齐更准…")
        res = separate(src, out_dir, "vocals", progress)
        audio = out_dir / next(s["file"] for s in res["stems"] if s["name"] == "vocals")
    global _whisper
    import stable_whisper  # type: ignore

    if _whisper is None:
        progress(f"载入 Whisper {WHISPER_MODEL} 模型…")
        _whisper = stable_whisper.load_model(WHISPER_MODEL)
    progress("Whisper 正在对齐歌词…")
    result = _whisper.align(str(audio), "\n".join(lyrics), language=language)
    words = [(w.start, w.end, w.word) for w in result.all_words()]
    return {"lines": lines_from_words(lyrics, words)}


# ---------------------------------------------------------------- audio -> MIDI
def transcribe(src: Path, progress: Progress = print) -> dict:
    if MOCK:
        dur = sf.info(str(src)).duration
        scale = [60, 62, 64, 67, 69, 72]
        notes = [{"p": scale[i % len(scale)], "s": round(i * 0.5, 3), "d": 0.45, "v": 0.8} for i in range(int(dur / 0.5))]
        return {"notes": notes}
    from basic_pitch.inference import predict  # type: ignore

    progress("basic-pitch 正在识别音符…")
    _, _, events = predict(str(src))
    notes = [{"p": int(p), "s": round(float(s), 3), "d": round(float(e - s), 3), "v": round(float(min(1.0, max(0.1, a))), 2)}
             for s, e, p, a, *_ in events]
    notes.sort(key=lambda n: n["s"])
    return {"notes": notes}


# ---------------------------------------------------------------- generation
def generate(opts: dict, out_dir: Path, progress: Progress = print) -> dict:
    prompt = (opts.get("prompt") or "").strip()
    lyrics = (opts.get("lyrics") or "").strip()
    duration = float(opts.get("duration") or 30)
    duration = max(5.0, min(duration, 240.0))
    if not prompt and not lyrics:
        raise ValueError("请写一段风格描述或歌词")
    if MOCK:
        return _mock_generate(out_dir, duration, opts)
    if ACESTEP_URL:
        return _acestep(opts, prompt, lyrics, duration, out_dir, progress)
    return _musicgen(prompt or lyrics, duration, out_dir, progress)


def _acestep(opts, prompt, lyrics, duration, out_dir, progress):
    import httpx

    headers = {"Authorization": f"Bearer {ACESTEP_API_KEY}"} if ACESTEP_API_KEY else {}
    body = {"prompt": prompt, "lyrics": lyrics, "audio_duration": duration, "vocal_language": opts.get("language") or "zh",
            "audio_format": "wav", "batch_size": 1, "thinking": True}
    if opts.get("bpm"):
        body["bpm"] = int(opts["bpm"])
    if opts.get("key"):
        body["key_scale"] = opts["key"]
    with httpx.Client(base_url=ACESTEP_URL, headers=headers, timeout=60) as c:
        progress("已提交给 ACE-Step…")
        r = c.post("/release_task", json=body)
        r.raise_for_status()
        task_id = r.json()["data"]["task_id"]
        deadline = time.time() + 1800
        while time.time() < deadline:
            time.sleep(3)
            q = c.post("/query_result", json={"task_id_list": [task_id]})
            q.raise_for_status()
            item = q.json()["data"][0]
            if item["status"] == 2:
                raise RuntimeError("ACE-Step 生成失败")
            if item["status"] == 1:
                res = item["result"]
                res = json.loads(res) if isinstance(res, str) else res
                first = res[0] if isinstance(res, list) else res
                audio = c.get(first["file"], timeout=300)
                audio.raise_for_status()
                dest = out_dir / "generated.wav"
                dest.write_bytes(audio.content)
                return {"file": dest.name, "engine": "acestep", "metas": first.get("metas")}
            progress("ACE-Step 生成中…")
    raise RuntimeError("ACE-Step 超时")


_musicgen_pipe = None


def _musicgen(prompt, duration, out_dir, progress):
    global _musicgen_pipe
    from transformers import pipeline  # type: ignore

    if _musicgen_pipe is None:
        progress(f"载入 {MUSICGEN_MODEL}…")
        _musicgen_pipe = pipeline("text-to-audio", model=MUSICGEN_MODEL)
    progress("MusicGen 生成中…")
    out = _musicgen_pipe(prompt, forward_params={"do_sample": True, "max_new_tokens": int(min(duration, 30) * 50)})
    audio = np.asarray(out["audio"]).squeeze()
    if audio.ndim == 2:
        audio = audio.T
    dest = out_dir / "generated.wav"
    sf.write(dest, audio, int(out["sampling_rate"]))
    return {"file": dest.name, "engine": "musicgen"}


# ---------------------------------------------------------------- mocks
def _tone(dur, sr, freq, amp=0.2):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _mock_separate(src, out_dir, mode):
    info = sf.info(str(src))
    names = ["vocals", "no_vocals"] if mode == "vocals" else ["vocals", "drums", "bass", "other"]
    stems = []
    for i, n in enumerate(names):
        dest = out_dir / f"{n}.wav"
        sf.write(dest, _tone(info.duration, info.samplerate, 220 * (i + 1)), info.samplerate)
        stems.append({"name": n, "label": STEM_LABELS[n], "file": dest.name})
    return {"stems": stems}


def _mock_generate(out_dir, duration, opts):
    sr = 44100
    bpm = float(opts.get("bpm") or 100)
    beat = 60 / bpm
    y = np.zeros(int(duration * sr), np.float32)
    for i in range(int(duration / beat)):
        s = int(i * beat * sr)
        n = _tone(beat * 0.9, sr, [262, 330, 392, 330][i % 4], 0.25)
        n *= np.linspace(1, 0, len(n), dtype=np.float32)
        y[s:s + len(n)] += n[: len(y) - s]
    dest = out_dir / "generated.wav"
    sf.write(dest, y, sr)
    return {"file": dest.name, "engine": "mock"}
