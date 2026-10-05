import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from seera_tts import server
from seera_tts.config import TTSConfig
from seera_tts.engines import TTSEngine
from seera_tts.pipeline import NarrationPipeline
from seera_tts.text import PronunciationLexicon, TextPreparer
from seera_tts.text.diacritizer import NoOpDiacritizer


class FakeEngine(TTSEngine):
    sample_rate = 24_000

    def __init__(self):
        self.calls = []

    def load(self):
        pass

    def voices(self):
        return ["narrator"]

    def stream(self, text, voice):
        self.calls.append(text)
        for _ in range(3):
            yield np.full(240, 0.1, dtype=np.float32)


@pytest.fixture
def client(monkeypatch):
    """A test client whose app uses the fake engine instead of loading XTTS."""
    engine = FakeEngine()
    cfg = TTSConfig(default_voice="narrator")
    preparer = TextPreparer(PronunciationLexicon({}), NoOpDiacritizer(), 150)
    pipeline = NarrationPipeline(engine, preparer, cfg)
    monkeypatch.setattr(NarrationPipeline, "from_config", classmethod(lambda cls, c: pipeline))
    with TestClient(server.app) as c:  # "with" runs the lifespan (startup/shutdown)
        c.engine = engine
        yield c


def collect(ws):
    """Read messages until 'done' or 'error'. Returns (json events, audio bytes)."""
    events, audio = [], b""
    while True:
        msg = ws.receive()
        if msg.get("bytes"):
            audio += msg["bytes"]
            continue
        event = json.loads(msg["text"])
        events.append(event)
        if event["event"] in ("done", "error"):
            return events, audio


# ------------------------------------------------------------------ HTTP
def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["sample_rate"] == 24000 and body["voices"] == ["narrator"]


def test_http_tts_returns_wav(client):
    r = client.post("/tts", json={"text": "بدأ الوحي في غار حراء."})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"


def test_http_rejects_bad_requests(client):
    assert client.post("/tts", json={"text": ""}).status_code == 422  # empty text
    assert client.post("/tts", json={"text": "نص", "voice": "nobody"}).status_code == 404


# ------------------------------------------------------------- WebSocket
def test_ws_speak(client):
    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"type": "speak", "text": "الجملة الأولى. قال تعالى ﴿آية﴾ الجملة الثانية."})
        events, audio = collect(ws)
    assert [e["event"] for e in events] == ["start", "segment", "segment", "quran", "segment", "done"]
    assert events[0] == {"event": "start", "sample_rate": 24000, "format": "pcm_s16le", "channels": 1}
    assert len(audio) > 0 and len(audio) % 2 == 0  # whole 16-bit samples


def test_ws_quran_event_has_recitation_urls(client):
    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"type": "speak", "text": "قال تعالى: ﴿اقرأ﴾ [العلق: 1-2] ثم رجع."})
        events, _ = collect(ws)
    quran = next(e for e in events if e["event"] == "quran")
    assert quran["audio_urls"] == [
        "https://everyayah.com/data/Alafasy_128kbps/096001.mp3",
        "https://everyayah.com/data/Alafasy_128kbps/096002.mp3",
    ]
    assert all("اقرأ" not in text for text in client.engine.calls)


def test_ws_llm_streaming_mode(client):
    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"type": "begin"})
        for delta in ["خرج النبي ", "من مكة. ", "ووصل إلى ", "المدينة."]:
            ws.send_json({"type": "append", "text": delta})
        ws.send_json({"type": "end"})
        events, _ = collect(ws)

        ws.send_json({"type": "speak", "text": "جملة."})  # same connection, next session
        events2, _ = collect(ws)

    segments = [e["text"] for e in events if e["event"] == "segment"]
    assert segments == ["خرج النبي من مكة.", "ووصل إلى المدينة."]
    assert events2[-1]["event"] == "done"


def test_ws_unknown_voice(client):
    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"type": "speak", "text": "نص", "voice": "nobody"})
        assert ws.receive_json()["event"] == "error"


def test_ws_unknown_message_type(client):
    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"type": "dance"})
        assert ws.receive_json()["event"] == "error"
        ws.send_json({"type": "speak", "text": "جملة."})  # connection still usable
        assert collect(ws)[0][-1]["event"] == "done"