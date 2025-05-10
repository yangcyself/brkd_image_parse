import os
import asyncio
import json
import base64
import requests
import sounddevice as sd
import websockets

API_KEY = os.getenv("OPENAI_API_KEY")
REST_URL = "https://api.openai.com/v1/realtime/transcription_sessions"
WS_URL   = "wss://api.openai.com/v1/realtime?intent=transcription"

RATE       = 16000
CHANNELS   = 1
FRAME_SIZE = 1024  # frames per block

def get_token():
    resp = requests.post(REST_URL,
                         headers={"Authorization":f"Bearer {API_KEY}"})
    resp.raise_for_status()
    return resp.json()["client_secret"]

def audio_generator():
    """Yields raw int16 PCM chunks from the default microphone."""
    with sd.RawInputStream(samplerate=RATE,
                           blocksize=FRAME_SIZE,
                           dtype='int16',
                           channels=CHANNELS) as stream:
        while True:
            yield stream.read(FRAME_SIZE)[0]  # returns (data, overflow)

async def run():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Beta":   "realtime=v1"
    }
    async with websockets.connect(WS_URL, extra_headers=headers) as ws:
        # now you’ll get a successful handshake…
        await ws.send(json.dumps({
            "type": "transcription_session.update",
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "gpt-4o-transcribe",
                "prompt": "",
                "language": ""
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500
            },
            "input_audio_noise_reduction": {"type": "near_field"},
            "include": ["item.input_audio_transcription.logprobs"]
        }))
        print("► Session started with beta header!")

        async def send_audio():
            for chunk in audio_generator():
                msg = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii")
                }
                await ws.send(json.dumps(msg))
                await asyncio.sleep(0)

        async def recv_transcripts():
            async for raw in ws:
                evt = json.loads(raw)
                if evt.get("type","").startswith("transcription_event"):
                    p = evt.get("payload", {})
                    text = p.get("alternatives",[{}])[0].get("transcript","")
                    flag = "[✔]" if p.get("is_final") else "[…]"
                    print(f"{flag} {text}")

        await asyncio.gather(send_audio(), recv_transcripts())

if __name__=="__main__":
    asyncio.run(run())
