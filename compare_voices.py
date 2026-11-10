"""Compare cloned vs stock Gnani TTS outputs for listening tests.

Usage:
  python compare_voices.py my_voice.wav "Text to speak"

Outputs (in current directory):
  out_clone_hi.wav  - cloned voice, Hindi
  out_clone_te.wav  - cloned voice, Telugu
  out_stock_te.wav  - stock voice, Telugu

Prints timings only. No secrets are printed.
"""
import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

GNANI_BASE = "https://api.vachana.ai"
GNANI_KEY = os.getenv("GNANI_API_KEY", "")
FALLBACK_VOICE = os.getenv("FALLBACK_VOICE", "Nalini")
TTS_MODEL = os.getenv("TTS_MODEL", "timbre-v2.5")
AUDIO_CFG = {"sample_rate": 24000, "num_channels": 1, "sample_width": 2,
             "encoding": "linear_pcm", "container": "wav"}


def die(msg):
    print(msg)
    raise SystemExit(1)


def extract_embedding(j):
    if isinstance(j, dict):
        if isinstance(j.get("embedding"), str):
            return {"embedding": j["embedding"], "shape": j.get("shape", [1, 768]),
                    "dtype": j.get("dtype", "torch.bfloat16")}
        for v in j.values():
            found = extract_embedding(v)
            if found:
                return found
    elif isinstance(j, list):
        for v in j:
            found = extract_embedding(v)
            if found:
                return found
    return None


def main():
    if not GNANI_KEY:
        die("GNANI_API_KEY is not set.")
    if len(sys.argv) < 3:
        die("Usage: python compare_voices.py my_voice.wav \"Text to speak\"")
    path, text = sys.argv[1], sys.argv[2]
    with open(path, "rb") as f:
        audio = f.read()
    headers = {"X-API-Key-ID": GNANI_KEY}
    with httpx.Client(timeout=60.0) as client:
        t0 = time.time()
        r = client.post(f"{GNANI_BASE}/api/v1/tts/voice-clone/embeddings", headers=headers,
                        files={"audio_file": (os.path.basename(path), audio)})
        if r.status_code >= 400:
            die(f"Embedding failed: HTTP {r.status_code}")
        emb = extract_embedding(r.json())
        if not emb:
            die("Embedding not found in response.")
        print(f"embedding_s={time.time() - t0:.2f}")

        jobs = [
            ("out_clone_hi.wav", {"text": text, "model": "vachana-vc-v1", "audio_config": AUDIO_CFG, "speaker_embedding": emb}),
            ("out_clone_te.wav", {"text": text, "model": "vachana-vc-v1", "audio_config": AUDIO_CFG, "speaker_embedding": emb}),
            ("out_stock_te.wav", {"text": text, "voice": FALLBACK_VOICE, "model": TTS_MODEL, "language": "te-IN", "speed": 1.0, "audio_config": AUDIO_CFG}),
        ]
        for name, body in jobs:
            t1 = time.time()
            r = client.post(f"{GNANI_BASE}/api/v1/tts/inference", headers=headers, json=body)
            if r.status_code >= 400:
                die(f"{name} failed: HTTP {r.status_code}")
            with open(name, "wb") as f:
                f.write(r.content)
            print(f"{name} tts_s={time.time() - t1:.2f} bytes={len(r.content)}")


if __name__ == "__main__":
    main()
