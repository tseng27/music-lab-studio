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
    assert all(f and sf.info(f).duration > 5 for f in files[:4]) and files[4:] == [None, None]


def test_separate_six_stems(server, wav):
    info, *files = server[1].predict(wav, "6stems", api_name="/separate")
    assert [s["label"] for s in info["stems"]] == ["人声", "鼓", "贝斯", "吉他", "钢琴", "其他乐器"]
    assert len(files) == 6 and all(files)


def test_separate_two_stems_leaves_empty_slots(server, wav):
    info, *files = server[1].predict(wav, "vocals", api_name="/separate")
    assert [s["label"] for s in info["stems"]] == ["人声", "伴奏"]
    assert files[2:] == [None] * 4


def test_align(server, wav):
    res = server[1].predict(wav, json.dumps(["第一句", "第二句"]), "zh", True, api_name="/align")
    assert [l["text"] for l in res["lines"]] == ["第一句", "第二句"] and res["lines"][0]["start"] == 0


def test_lyrics(server, wav):
    res = server[1].predict(wav, "zh-TW", True, api_name="/lyrics")
    assert [l["text"] for l in res["lines"]][:1] == ["第一句歌词"]
    assert all(l["start"] < l["end"] for l in res["lines"])


def test_clean_lyric_lines():
    segs = [(0, 2, " 我把春天带着走，"), (2, 3, "我把春天带着走"), (3, 4, "字幕由 xx 提供"), (5, 6, "  ")]
    lines = engines.clean_lyric_lines(segs, "zh-TW")
    assert [l["start"] for l in lines] == [0, 2]
    assert lines[0]["text"] == "我把春天帶著走"
    assert engines.clean_lyric_lines([(0, 1, "Hello.")], "en")[0]["text"] == "Hello"
    assert engines.whisper_language("zh-TW") == "zh"


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


def test_cloud_needs_login(server):
    assert server[1].predict(api_name="/cloud_list") == {"login": False, "available": False}
    with pytest.raises(Exception, match="登录"):
        server[1].predict("x", api_name="/cloud_load")


class FakeHub:
    def __init__(self):
        self.files, self.repos = {}, set()

    def whoami(self, token):
        assert token == "tok"
        return {"name": "alice"}

    def repo_exists(self, rid, repo_type, token):
        return rid in self.repos

    def create_repo(self, rid, repo_type, private, exist_ok, token):
        assert private and repo_type == "dataset"
        self.repos.add(rid)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, token, commit_message):
        self.files[path_in_repo] = open(path_or_fileobj).read()

    def list_repo_files(self, rid, repo_type, token):
        return list(self.files) + [".gitattributes"]

    def hf_hub_download(self, rid, filename, repo_type, token):
        return self.files[filename]


def test_cloud_roundtrip_with_fake_hub(tmp_path):
    from backend import cloud

    hub = FakeHub()
    assert cloud.list_projects("tok", hub) == {"repo": "alice/musiclab-projects", "projects": []}
    f = tmp_path / "p.json"
    f.write_text('{"tracks":[]}')
    assert cloud.save_project("tok", "我的 歌/v1", f, hub) == {"repo": "alice/musiclab-projects", "name": "我的_歌_v1"}
    assert cloud.list_projects("tok", hub)["projects"] == ["我的_歌_v1"]
    assert cloud.load_project("tok", "我的 歌/v1", hub) == '{"tracks":[]}'


def test_collab_room_sync(server):
    import asyncio

    import websockets

    url, _ = server
    base = url.replace("http", "ws")

    async def run():
        async with websockets.connect(base + "collab/room1234/ws?cid=a&name=A") as a:
            assert json.loads(await a.recv())["type"] == "welcome"
            r = httpx.post(url + "collab/room1234/audio/b1", content=b"RIFFdata")
            assert r.status_code == 200
            await a.send(json.dumps({"type": "audio", "bid": "b1", "name": "鼓"}))
            await a.send(json.dumps({"type": "state", "state": '{"tracks":[1]}'}))
            await asyncio.sleep(0.2)
            async with websockets.connect(base + "collab/room1234/ws?cid=b&name=B") as b:
                w = json.loads(await b.recv())
                assert w["state"] == '{"tracks":[1]}' and w["audio"] == [{"bid": "b1", "name": "鼓"}]
                assert [m["name"] for m in w["members"]] == ["A", "B"]
                assert json.loads(await a.recv())["type"] == "members"
                await b.send(json.dumps({"type": "state", "state": '{"tracks":[2]}'}))
                assert json.loads(await a.recv()) == {"type": "state", "state": '{"tracks":[2]}', "from": "b"}
            assert httpx.get(url + "collab/room1234/audio/b1").content == b"RIFFdata"

    asyncio.run(run())


def test_musicgen_loads_once_with_fake_transformers(tmp_path, monkeypatch):
    """The real MusicGen path (no GPU, no download): load once, then reuse the cached model."""
    import contextlib
    import sys
    import types

    loads = []

    class Out:
        def __init__(self, a):
            self.a = a

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    SR = 1000
    calls = []

    class Inputs(dict):
        def to(self, dev):
            return self

    class Proc:
        @classmethod
        def from_pretrained(cls, name):
            loads.append(name)
            return cls()

        def __call__(self, audio=None, **kw):
            return Inputs(prompt=0 if audio is None else len(audio))

    class Model:
        config = types.SimpleNamespace(audio_encoder=types.SimpleNamespace(sampling_rate=SR))

        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def eval(self):
            return self

        def generate(self, max_new_tokens, prompt=0, **kw):
            # like MusicGen: the returned audio starts with the (re-decoded) prompt
            calls.append((prompt, max_new_tokens))
            n = prompt + max_new_tokens * SR // 50
            return [Out(np.full((1, n), 0.1 * len(calls), dtype=np.float32))]

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoProcessor=Proc, MusicgenForConditionalGeneration=Model))
    monkeypatch.setattr(engines, "on_device", lambda m: contextlib.nullcontext("cpu"))
    monkeypatch.setattr(engines, "_musicgen_model", None)
    for _ in range(2):
        res = engines._musicgen("strings", 5, tmp_path, lambda *_: None)
        assert res == {"file": "generated.wav", "engine": "musicgen"}
    assert loads == [engines.MUSICGEN_MODEL]
    assert sf.info(tmp_path / "generated.wav").samplerate == SR

    # 70 s = one 30 s window + two 20 s continuations, each prompted with the last 10 s
    calls.clear()
    engines._musicgen("strings", 70, tmp_path, lambda *_: None)
    assert calls == [(0, 1500), (10 * SR, 1000), (10 * SR, 1000)]
    out, sr = sf.read(tmp_path / "generated.wav")
    assert len(out) == 70 * SR
    # no repeated prompt audio: each window contributes only its new part
    assert np.allclose(out[: 30 * SR], 0.1, atol=1e-3) and np.allclose(out[30 * SR: 50 * SR], 0.2, atol=1e-3) and np.allclose(out[50 * SR:], 0.3, atol=1e-3)
    # capped at two minutes
    calls.clear()
    engines._musicgen("strings", 240, tmp_path, lambda *_: None)
    assert len(sf.read(tmp_path / "generated.wav")[0]) == 120 * SR
