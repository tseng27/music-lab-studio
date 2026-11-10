"""AI engines behind the studio's 「AI 工具」 panel.

Every engine takes an input file path plus options, writes any audio it makes into
``out_dir`` and returns a JSON-serialisable result. Heavy libraries are imported
lazily so the server starts (and the frontend works) even when some are missing.
Models stay on the CPU between calls and borrow the GPU only while they run,
which is what Hugging Face ZeroGPU expects.

Set ``MUSICLAB_MOCK=1`` to replace every engine with a fast synthetic stand-in;
that is how the test suite and a GPU-less demo run.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
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
MUSICGEN_MAX = 120.0  # seconds; longer pieces are made window by window
MUSICGEN_WINDOW = 30.0  # what the model was trained on
MUSICGEN_CONTEXT = 10.0  # seconds fed back as the prompt for the next window
ACESTEP_URL = os.environ.get("ACESTEP_URL", "").rstrip("/")
ACESTEP_API_KEY = os.environ.get("ACESTEP_API_KEY", "")


def has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def device() -> str:
    """"cuda" inside a ZeroGPU call or on a GPU machine, else "cpu"."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class on_device:
    """Move a cached CPU model to the GPU for one call, then back.

    ZeroGPU only lends a GPU for the duration of a @spaces.GPU call, so models
    live on the CPU between calls.
    """

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.dev = device()
        self.model.to(self.dev)
        return self.dev

    def __exit__(self, *exc):
        if self.dev != "cpu":
            self.model.to("cpu")


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


_demucs: dict = {}  # model name -> loaded Demucs model


def _load_audio(path: Path, sr: int):
    """Read any soundfile-supported file as a (2, samples) float tensor at ``sr``."""
    import torch
    import torchaudio.functional as AF

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T.copy())
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    wav = wav[:2]
    if file_sr != sr:
        wav = AF.resample(wav, file_sr, sr)
    return wav


def separate(src: Path, out_dir: Path, mode: str = "vocals", progress: Progress = print) -> dict:
    """Split a mix into stems.

    mode = "vocals" (人声 + 伴奏), "4stems" (人声/鼓/贝斯/其他) or "6stems" (adds 吉他/钢琴, model htdemucs_6s).
    """
    if MOCK:
        return _mock_separate(src, out_dir, mode)
    import torch
    from demucs.apply import apply_model  # type: ignore
    from demucs.pretrained import get_model  # type: ignore

    name = "htdemucs_6s" if mode == "6stems" else DEMUCS_MODEL
    if name not in _demucs:
        progress(f"载入 Demucs {name}…")
        _demucs[name] = get_model(name).eval()
    model = _demucs[name]
    wav = _load_audio(src, model.samplerate)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    progress("Demucs 正在分离…")
    with on_device(model) as dev, torch.no_grad():
        out = apply_model(model, ((wav - mean) / std)[None].to(dev), device=dev, shifts=1, split=True, overlap=0.25)[0]
    out = (out * std + mean).cpu()
    parts = dict(zip(model.sources, out))
    if mode == "vocals":
        parts = {"vocals": parts["vocals"], "no_vocals": sum(v for k, v in parts.items() if k != "vocals")}
    order = ["vocals", "no_vocals", "drums", "bass", "guitar", "piano", "other"]
    stems = []
    for name in sorted(parts, key=lambda n: order.index(n) if n in order else 99):
        dest = out_dir / f"{name}.wav"
        sf.write(dest, parts[name].numpy().T, model.samplerate)
        stems.append({"name": name, "label": STEM_LABELS.get(name, name), "file": dest.name})
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
        _whisper = stable_whisper.load_model(WHISPER_MODEL, device="cpu")
    progress("Whisper 正在对齐歌词…")
    with on_device(_whisper):
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


_musicgen_model = None  # (processor, model) once loaded


def prefetch_generate_model(progress: Progress = print) -> None:
    """Download the MusicGen weights ahead of time, outside the GPU allocation."""
    if MOCK or ACESTEP_URL:
        return
    from huggingface_hub import snapshot_download

    progress(f"下载 {MUSICGEN_MODEL}…")
    snapshot_download(MUSICGEN_MODEL)


def _musicgen(prompt, duration, out_dir, progress):
    global _musicgen_model
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration  # type: ignore

    if _musicgen_model is None:
        progress(f"载入 {MUSICGEN_MODEL}…")
        _musicgen_model = (AutoProcessor.from_pretrained(MUSICGEN_MODEL),
                           MusicgenForConditionalGeneration.from_pretrained(MUSICGEN_MODEL).eval())
    proc, model = _musicgen_model
    sr = model.config.audio_encoder.sampling_rate
    total = min(duration, MUSICGEN_MAX)
    with on_device(model) as dev, torch.no_grad():
        # first window, then keep extending: the last few seconds become the audio prompt
        # for the next window, so tempo, key and instruments carry on
        progress("MusicGen 生成中…")
        inputs = proc(text=[prompt], padding=True, return_tensors="pt").to(dev)
        first = min(total, MUSICGEN_WINDOW)
        out = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=int(first * 50))
        audio = np.asarray(out[0].cpu().numpy(), dtype=np.float32).reshape(-1)[: int(first * sr)]
        while len(audio) < int(total * sr) - sr // 10:
            progress(f"MusicGen 接续生成… {len(audio) / sr:.0f}/{total:.0f} 秒")
            ctx = audio[-int(MUSICGEN_CONTEXT * sr):]
            step = min(total - len(audio) / sr, MUSICGEN_WINDOW - MUSICGEN_CONTEXT)
            inputs = proc(audio=ctx, sampling_rate=sr, text=[prompt], padding=True, return_tensors="pt").to(dev)
            out = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=int(step * 50))
            new = np.asarray(out[0].cpu().numpy(), dtype=np.float32).reshape(-1)[len(ctx):]
            if len(new) < sr // 10:
                break
            audio = np.concatenate([audio, new[: int(step * sr)]])
    dest = out_dir / "generated.wav"
    sf.write(dest, audio, sr)
    return {"file": dest.name, "engine": "musicgen"}


# ---------------------------------------------------------------- mocks
def _tone(dur, sr, freq, amp=0.2):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _mock_separate(src, out_dir, mode):
    info = sf.info(str(src))
    names = {"vocals": ["vocals", "no_vocals"], "4stems": ["vocals", "drums", "bass", "other"],
             "6stems": ["vocals", "drums", "bass", "guitar", "piano", "other"]}[mode]
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
