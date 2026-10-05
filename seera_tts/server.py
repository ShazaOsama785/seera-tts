"""HTTP + WebSocket API.

Run:  uvicorn seera_tts.server:app --host 0.0.0.0 --port 8000

WebSocket /ws/tts protocol
--------------------------
Client -> server (JSON text frames):
    {"type": "speak", "text": "...", "voice": "narrator"}     one-shot
    {"type": "begin", "voice": "narrator"}                     start LLM-streaming mode
    {"type": "append", "text": "<delta>"}                      ...send deltas as they arrive
    {"type": "end"}                                            no more text
Server -> client:
    {"event": "start", "sample_rate": 24000, "format": "pcm_s16le", "channels": 1}
    {"event": "segment", "index": 0, "text": "..."}            highlight this text now
    <binary frames: raw PCM audio for that segment>
    {"event": "quran", "index": 3, "text": "﴿...﴾", "audio_urls": [...]}
                                                               show verse, play the recitation
    {"event": "done"}
    {"event": "error", "message": "..."}
A connection can run many sessions one after another.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from contextlib import asynccontextmanager
from typing import TypeVar

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .config import TTSConfig
from .pipeline import AudioChunk, NarrationPipeline, QuranSegment, SegmentStarted

logger = logging.getLogger("seera_tts")
T = TypeVar("T")


# ------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    config = TTSConfig.from_env()
    app.state.config = config
    app.state.pipeline = await run_in_threadpool(NarrationPipeline.from_config, config)
    yield


app = FastAPI(title="Darb Al-Seerah TTS", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=TTSConfig.max_text_chars)
    voice: str | None = None


def _pipeline(request: Request | WebSocket) -> NarrationPipeline:
    return request.app.state.pipeline


# ------------------------------------------------------------------ HTTP
@app.get("/health")
async def health(request: Request) -> dict:
    p = _pipeline(request)
    return {"status": "ok", "sample_rate": p.sample_rate, "voices": p.engine.voices()}


@app.get("/voices")
async def voices(request: Request) -> list[str]:
    return _pipeline(request).engine.voices()


@app.post("/tts", response_class=Response)
async def tts(body: TTSRequest, request: Request) -> Response:
    """Non-streaming fallback: full narration as one WAV."""
    pipeline = _pipeline(request)
    _check_voice(pipeline, body.voice)
    wav = await run_in_threadpool(pipeline.synthesize_wav, body.text, body.voice)
    return Response(wav, media_type="audio/wav")


def _check_voice(pipeline: NarrationPipeline, voice: str | None) -> None:
    if voice and voice not in pipeline.engine.voices():
        raise HTTPException(404, f"Unknown voice {voice!r}")


# ------------------------------------------------------------- WebSocket
class TextFeed:
    """Thread-safe iterable of text deltas, fed from the async receive loop."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()

    def put(self, text: str) -> None:
        self._queue.put(text)

    def close(self) -> None:
        self._queue.put(None)

    def __iter__(self) -> Iterator[str]:
        while (item := self._queue.get()) is not None:
            yield item


@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket) -> None:
    await ws.accept()
    pipeline = _pipeline(ws)
    try:
        while True:
            msg = await ws.receive_json()
            kind, voice = msg.get("type"), msg.get("voice")
            if voice and voice not in pipeline.engine.voices():
                await ws.send_json({"event": "error", "message": f"Unknown voice {voice!r}"})
                continue

            if kind == "speak":
                await _run_session(ws, pipeline, msg.get("text", ""), voice)
            elif kind == "begin":
                await _run_streaming_session(ws, pipeline, voice)
            else:
                await ws.send_json({"event": "error", "message": f"Unexpected message type {kind!r}"})
    except WebSocketDisconnect:
        pass


async def _run_streaming_session(ws: WebSocket, pipeline: NarrationPipeline, voice: str | None) -> None:
    """Synthesize while text deltas are still arriving."""
    feed = TextFeed()
    session = asyncio.create_task(_run_session(ws, pipeline, feed, voice))
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "append":
                feed.put(msg.get("text", ""))
            elif msg.get("type") == "end":
                break
    finally:
        feed.close()  # also unblocks synthesis if the client disconnected
    await session


async def _run_session(
    ws: WebSocket, pipeline: NarrationPipeline, text: str | Iterable[str], voice: str | None
) -> None:
    await ws.send_json(
        {"event": "start", "sample_rate": pipeline.sample_rate, "format": "pcm_s16le", "channels": 1}
    )
    try:
        async for event in iterate_in_thread(lambda: pipeline.stream(text, voice)):
            match event:
                case AudioChunk(pcm=pcm):
                    await ws.send_bytes(pcm)
                case SegmentStarted(index=i, text=t):
                    await ws.send_json({"event": "segment", "index": i, "text": t})
                case QuranSegment(index=i, text=t, audio_urls=urls):
                    await ws.send_json({"event": "quran", "index": i, "text": t, "audio_urls": list(urls)})
    except Exception as exc:  # report to client instead of silently killing the socket
        logger.exception("TTS session failed")
        await ws.send_json({"event": "error", "message": str(exc)})
        return
    await ws.send_json({"event": "done"})


async def iterate_in_thread(make_iterator: Callable[[], Iterator[T]]) -> AsyncIterator[T]:
    """Run a blocking (GPU) iterator in a worker thread and consume it asynchronously.

    Stops the worker promptly if the consumer goes away (e.g. client disconnects).
    """
    loop = asyncio.get_running_loop()
    items: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    done = object()

    def worker() -> None:
        iterator = make_iterator()
        try:
            for item in iterator:
                if stop.is_set():
                    break
                loop.call_soon_threadsafe(items.put_nowait, item)
        except Exception as exc:
            loop.call_soon_threadsafe(items.put_nowait, exc)
        finally:
            close = getattr(iterator, "close", None)
            if close:
                close()  # releases the engine's GPU lock immediately
            loop.call_soon_threadsafe(items.put_nowait, done)

    worker_future = loop.run_in_executor(None, worker)
    try:
        while (item := await items.get()) is not done:
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
        await asyncio.shield(worker_future)