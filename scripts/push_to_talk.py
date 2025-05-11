#!/usr/bin/env python3
import threading
import io
import wave
import time
import json

from openai import OpenAI
from pydub.generators import Sine
from pydub.playback import play
import sounddevice as sd
from pynput import keyboard
from scipy.io.wavfile import write

class PushToTalk:
    def __init__(self, on_transcript):
        """
        on_transcript: callback taking (text:str) once transcription is done
        """
        self.on_transcript = on_transcript

        # recording params
        self.RATE = 24000
        self.CHANNELS = 1
        self.FRAMES_PER_BUFFER = 1024
        self.dtype = 'int16'

        self._stream = None
        self._frames = []
        self._recording = False
        self._lock = threading.Lock()

        # start keyboard listener
        listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        listener.daemon = True
        listener.start()

        self._openai = OpenAI()

    def _beep(self, freq=1000, duration_ms=150):
        tone = Sine(freq).to_audio_segment(duration=duration_ms)
        play(tone)

    def _sd_callback(self, indata, frames, time_info, status):
        if status:
            # you could log the overflow here
            return
        with self._lock:
            if self._recording:
                # store raw bytes
                self._frames.append(indata.copy())

    def _on_press(self, key):
        if key == keyboard.Key.space:
            # print("on press")
            with self._lock:
                if not self._recording:
                    self._recording = True
                    self._frames = []
                    self._beep()
                    # self._stream = sd.InputStream(
                    #     samplerate=self.RATE,
                    #     channels=self.CHANNELS,
                    #     dtype=self.dtype,
                    #     blocksize=self.FRAMES_PER_BUFFER,
                    #     callback=self._sd_callback
                    # )
                    # self._stream.start()
                    self.recording = sd.rec(99999999, samplerate=self.RATE, channels=self.CHANNELS, dtype=self.dtype)

    def _on_release(self, key):
        if key == keyboard.Key.space:
            print("on release")
            transcribe = False
            print("wait for lock")
            with self._lock:
                print("got lock")
                print("recording", self._recording)
                if self._recording:
                    self._recording = False
                    # stop & close stream
                    # self._stream.stop()
                    # print("stop stream done")
                    # self._stream.abort()

                    # self._stream.close()
                    # self._stream = None
                    sd.stop()
                    transcribe = True
                    # transcribe in background
                    # threading.Thread(target=self._transcribe, daemon=True).start()
            print("transcribe", transcribe)
            if transcribe:

                # play a beep to indicate recording stopped
                self._transcribe()

    def _transcribe(self):
        # stitch together into one NumPy array
        import numpy as np
        # data = np.concatenate(self._frames, axis=0)
        data = self.recording 
        write('output.wav', self.RATE, data)

        print("write to in-memory WAV")
        # write to in-memory WAV
        buf = io.BytesIO()
        wf = wave.open(buf, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(np.dtype(self.dtype).itemsize)
        wf.setframerate(self.RATE)
        wf.writeframes(data.tobytes())
        wf.close()
        buf.seek(0)
        buf.name = "audio.wav"

        # save to a file for debugging
        # with open("audio.wav", "wb") as f:
        #     f.write(buf.getbuffer())

        print("call Whisper")
        # call Whisper
        resp = self._openai.audio.transcriptions.create(
            model="whisper-1",
            file=buf,

            response_format="json"
        )
        print("transcription done")
        print(resp)
        text = resp.text.strip()
        if text:
            self.on_transcript(text)


def main():
    def on_transcript(text: str):
        print(f"[Whisper] {text}")

    ptt = PushToTalk(on_transcript)
    print("Push-to-talk ready. Hold SPACE to record; release to transcribe.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    main()