"""API tests in demo mode (MUSICLAB_MOCK=1): start the Space app, call it like the studio does."""
import json
import os
import socket

import numpy as np
import pytest
import soundfile as sf

os.environ["MUSICLAB_MOCK"] = "1"

import httpx  # noqa: E402
from gradio_client import Client, handle_file  # noqa: E402

import app as space  # noqa: E402
from backend import engines  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    space.app.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True, show_error=True)
    url = f"http://127.0.0.1:{port}/"
    yield url, Client(url, verbose=False)
    space.app.get_blocks().close()


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "in.wav"
    t = np.arange(int(6 * 22050)) / 22050
    sf.write(p, (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 22050)
    return handle_file(str(p))


def test_page_and_client_bundle(server):
    url, _ = server
    assert "音乐实验室" in httpx.get(url).text
    js = httpx.get(url + "vendor/gradio-client.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]


def test_health(server):
    assert server[1].predict(api_name="/health")["features"]["mock"] is True


def test_separate_four_stems(server, wav):
    info, *files = server[1].predict(wav, "4stems", api_name="/separate")
    assert [s["name"] for s in info["stems"]] == ["vocals", "drums", "bass", "other"]
    assert all(f and sf.info(f).duration > 5 for f in files)


def test_separate_two_stems_leaves_empty_slots(server, wav):
    info, *files = server[1].predict(wav, "vocals", api_name="/separate")
    assert [s["label"] for s in info["stems"]] == ["人声", "伴奏"]
    assert files[2] is None and files[3] is None


def test_align(server, wav):
    res = server[1].predict(wav, json.dumps(["第一句", "第二句"]), "zh", True, api_name="/align")
    assert [l["text"] for l in res["lines"]] == ["第一句", "第二句"] and res["lines"][0]["start"] == 0


def test_transcribe_and_generate(server, wav):
    assert server[1].predict(wav, api_name="/transcribe")["notes"]
    info, path = server[1].predict("lofi piano", "", 10, 90, "zh", api_name="/generate")
    assert info["engine"] == "mock" and abs(sf.info(path).duration - 10) < 0.1


def test_errors_reach_the_client(server, wav):
    with pytest.raises(Exception, match="mode"):
        server[1].predict(wav, "x", api_name="/separate")
    with pytest.raises(Exception, match="风格描述"):
        server[1].predict("", "", 10, 0, "zh", api_name="/generate")


def test_lines_from_words_maps_chinese_and_english():
    words = [(0.0, 0.3, "夜色"), (0.3, 0.6, "慢慢"), (1.0, 1.2, "落下,"), (2.0, 2.4, " Hello"), (2.4, 2.8, " world!")]
    out = engines.lines_from_words(["夜色慢慢", "落下", "Hello world"], words)
    assert [l["start"] for l in out] == [0.0, 1.0, 2.0]
    assert out[2]["end"] == 2.8


def test_lines_from_words_word_spanning_two_lines():
    out = engines.lines_from_words(["我爱", "你"], [(0.0, 1.0, "我爱你")])
    assert [l["start"] for l in out] == [0.0, 0.0]
