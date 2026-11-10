import io
import os
import time

import numpy as np
import soundfile as sf

os.environ["MUSICLAB_MOCK"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from backend import engines  # noqa: E402
from backend.app import app  # noqa: E402

client = TestClient(app)


def wav_bytes(seconds=6.0, sr=22050):
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def wait(job_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.1)
    raise AssertionError("job timed out")


def test_health_and_frontend():
    assert client.get("/api/health").json()["features"]["mock"] is True
    assert "音乐实验室" in client.get("/").text


def test_separate_returns_downloadable_stems():
    r = client.post("/api/separate", files={"file": ("a.wav", wav_bytes(), "audio/wav")}, data={"mode": "4stems"})
    j = wait(r.json()["id"])
    assert j["status"] == "done"
    stems = j["result"]["stems"]
    assert [s["name"] for s in stems] == ["vocals", "drums", "bass", "other"]
    assert client.get(stems[0]["url"]).status_code == 200


def test_align_transcribe_generate():
    r = client.post("/api/align", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                    data={"lyrics": '["第一句", "第二句"]', "language": "zh"})
    lines = wait(r.json()["id"])["result"]["lines"]
    assert [l["text"] for l in lines] == ["第一句", "第二句"] and lines[0]["start"] == 0

    r = client.post("/api/transcribe", files={"file": ("a.wav", wav_bytes(), "audio/wav")})
    assert wait(r.json()["id"])["result"]["notes"]

    r = client.post("/api/generate", json={"prompt": "lofi piano", "duration": 10})
    j = wait(r.json()["id"])
    assert client.get(j["result"]["url"]).status_code == 200


def test_bad_requests():
    assert client.post("/api/separate", files={"file": ("a.wav", wav_bytes(), "audio/wav")}, data={"mode": "x"}).status_code == 400
    assert client.get("/api/jobs/nope").status_code == 404
    r = client.post("/api/generate", json={"prompt": "", "lyrics": ""})
    assert wait(r.json()["id"])["status"] == "error"


def test_lines_from_words_maps_chinese_and_english():
    words = [(0.0, 0.3, "夜色"), (0.3, 0.6, "慢慢"), (1.0, 1.2, "落下,"), (2.0, 2.4, " Hello"), (2.4, 2.8, " world!")]
    out = engines.lines_from_words(["夜色慢慢", "落下", "Hello world"], words)
    assert [l["start"] for l in out] == [0.0, 1.0, 2.0]
    assert out[2]["end"] == 2.8


def test_lines_from_words_word_spanning_two_lines():
    out = engines.lines_from_words(["我爱", "你"], [(0.0, 1.0, "我爱你")])
    assert [l["start"] for l in out] == [0.0, 0.0]
