"""Smoke-test a running server from anywhere: time to first audio, then save output.wav.

    python examples/ws_client.py wss://<your-tunnel>.trycloudflare.com/ws/tts
"""

import asyncio
import json
import sys
import time
import wave

import websockets

TEXT = (
    "ثم أرسلني فقال: { اقْرَأْ بِاسْمِ رَبِّكَ الَّذِي خَلَقَ } [ العلق: 1 ] "
    "فرجع بها النبي ﷺ يرجف فؤاده، حتى دخل على خديجة رضي الله عنها."
)


async def main(url: str) -> None:
    async with websockets.connect(url, max_size=None) as ws:
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "speak", "text": TEXT}))
        pcm, first_audio, sample_rate = bytearray(), None, 24000
        async for msg in ws:
            if isinstance(msg, bytes):
                first_audio = first_audio or time.perf_counter() - t0
                pcm += msg
                continue
            event = json.loads(msg)
            print(event)
            if event["event"] == "start":
                sample_rate = event["sample_rate"]
            if event["event"] in ("done", "error"):
                break

    total = time.perf_counter() - t0
    duration = len(pcm) / 2 / sample_rate
    print(f"first audio: {first_audio:.2f}s | total: {total:.2f}s | audio: {duration:.2f}s")
    with wave.open("output.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm)
    print("saved output.wav")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))