"""Collaboration rooms: everyone with the room link edits the same project live.

A room keeps the latest project snapshot (JSON text) and the audio files it uses.
Clients send their whole snapshot when something changes (last write wins) and
upload each new audio buffer once over HTTP; the WebSocket only carries small
messages. Rooms live in memory and in a temp folder, and are dropped when the
last member leaves.
"""

import asyncio
import json
import re
import shutil
import tempfile
from pathlib import Path

from fastapi import HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

ROOM_RE = re.compile(r"^[A-Za-z0-9_-]{4,40}$")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MAX_AUDIO = 300 * 1024 * 1024
MAX_ROOMS = 50


class Room:
    def __init__(self, name: str):
        self.name = name
        self.clients: dict[WebSocket, dict] = {}
        self.state: str | None = None
        self.audio: dict[str, str] = {}  # bid -> display name
        self.dir = Path(tempfile.mkdtemp(prefix=f"collab-{name}-"))

    def members(self) -> list[dict]:
        return [{"cid": m["cid"], "name": m["name"]} for m in self.clients.values()]

    async def send_all(self, msg: dict, skip: WebSocket | None = None):
        text = json.dumps(msg, ensure_ascii=False)
        for ws in list(self.clients):
            if ws is skip:
                continue
            try:
                await ws.send_text(text)
            except Exception:
                self.clients.pop(ws, None)


ROOMS: dict[str, Room] = {}


def _room(name: str, create: bool = False) -> Room:
    if not ROOM_RE.match(name):
        raise HTTPException(400, "房间名称不对")
    room = ROOMS.get(name)
    if room is None:
        if not create:
            raise HTTPException(404, "房间不存在")
        if len(ROOMS) >= MAX_ROOMS:
            raise HTTPException(503, "房间太多了，请稍后再试")
        room = ROOMS[name] = Room(name)
    return room


def attach(app) -> None:
    @app.websocket("/collab/{name}/ws")
    async def collab_ws(ws: WebSocket, name: str):
        if not ROOM_RE.match(name):
            await ws.close(code=1008)
            return
        await ws.accept()
        room = _room(name, create=True)
        cid = ws.query_params.get("cid") or "anon"
        nick = (ws.query_params.get("name") or "访客")[:24]
        room.clients[ws] = {"cid": cid, "name": nick}
        await ws.send_text(json.dumps({"type": "welcome", "state": room.state,
                                       "audio": [{"bid": b, "name": n} for b, n in room.audio.items()],
                                       "members": room.members()}, ensure_ascii=False))
        await room.send_all({"type": "members", "members": room.members()}, skip=ws)
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                kind = msg.get("type")
                if kind == "state" and isinstance(msg.get("state"), str) and len(msg["state"]) < 20_000_000:
                    room.state = msg["state"]
                    await room.send_all({"type": "state", "state": room.state, "from": cid}, skip=ws)
                elif kind == "audio" and ID_RE.match(str(msg.get("bid", ""))) and (room.dir / msg["bid"]).exists():
                    room.audio[msg["bid"]] = str(msg.get("name", ""))[:80]
                    await room.send_all({"type": "audio", "bid": msg["bid"], "name": room.audio[msg["bid"]]}, skip=ws)
                elif kind == "pos":
                    await room.send_all({"type": "pos", "cid": cid, "name": nick, "pos": float(msg.get("pos", 0))}, skip=ws)
        except (WebSocketDisconnect, ValueError, RuntimeError):
            pass
        finally:
            room.clients.pop(ws, None)
            if room.clients:
                await room.send_all({"type": "members", "members": room.members()})
            else:
                ROOMS.pop(name, None)
                await asyncio.to_thread(shutil.rmtree, room.dir, True)

    @app.post("/collab/{name}/audio/{bid}")
    async def collab_put(name: str, bid: str, request: Request):
        room = _room(name)
        if not ID_RE.match(bid):
            raise HTTPException(400, "bad id")
        body = await request.body()
        if len(body) > MAX_AUDIO:
            raise HTTPException(413, "音频太大")
        (room.dir / bid).write_bytes(body)
        return {"ok": True}

    @app.get("/collab/{name}/audio/{bid}")
    def collab_get(name: str, bid: str):
        room = _room(name)
        path = room.dir / bid
        if not ID_RE.match(bid) or not path.exists():
            raise HTTPException(404, "没有这个音频")
        return FileResponse(path, media_type="audio/wav")
