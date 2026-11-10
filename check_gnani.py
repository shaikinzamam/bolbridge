"""20-minute feasibility test. Run:  python check_gnani.py my_voice.wav
Use a clear 8-20 s recording of YOUR voice (WAV or MP3 for cloning, under 30 s)."""
import sys, time, os, httpx
from dotenv import load_dotenv
load_dotenv()
KEY = os.getenv("GNANI_API_KEY", "")
B = "https://api.vachana.ai"
H = {"X-API-Key-ID": KEY}
if not KEY or len(sys.argv) < 2:
    sys.exit("Set GNANI_API_KEY in .env and pass an audio file: python check_gnani.py my_voice.wav")
path = sys.argv[1]
audio = open(path, "rb").read()
cfg = {"sample_rate": 24000, "num_channels": 1, "sample_width": 2, "encoding": "linear_pcm", "container": "wav"}

def show(r):
    if r.status_code >= 400:
        sys.exit(f"  FAILED {r.status_code}: {r.text[:300]}")

def find_embedding(j):
    if isinstance(j, dict):
        if isinstance(j.get("embedding"), str):
            return {"embedding": j["embedding"], "shape": j.get("shape", [1, 768]), "dtype": j.get("dtype", "torch.bfloat16")}
        for v in j.values():
            f = find_embedding(v)
            if f: return f
    elif isinstance(j, list):
        for v in j:
            f = find_embedding(v)
            if f: return f
    return None

def outline(j, d=0):
    """Print response structure without dumping the huge embedding string."""
    if isinstance(j, dict):
        return "{" + ", ".join(f"{k}: {outline(v, d+1)}" for k, v in j.items()) + "}"
    if isinstance(j, list):
        return f"[list of {len(j)}]"
    return f"<{type(j).__name__} len={len(j)}>" if isinstance(j, str) and len(j) > 60 else repr(j)

print("1) STT (Prisma v3.0, auto language)")
t = time.time()
r = httpx.post(f"{B}/stt/v3", headers=H, timeout=60,
               files={"audio_file": (os.path.basename(path), audio)},
               data={"model": "prisma-v3.0", "language_code": "indic-auto"})
show(r); j = r.json()
print(f"  {time.time()-t:.2f}s  lang={j.get('detected_language_code')}  text={j.get('transcript')}")

print("2) Voice clone embedding")
t = time.time()
r = httpx.post(f"{B}/api/v1/tts/voice-clone/embeddings", headers=H, timeout=60,
               files={"audio_file": (os.path.basename(path), audio)})
show(r); raw = r.json()
emb = find_embedding(raw)
if not emb:
    sys.exit("  Could not find an embedding. Response structure: " + outline(raw))
print(f"  {time.time()-t:.2f}s  OK  shape={emb['shape']}  (structure: {outline(raw)})")

print("3) Cloned-voice TTS (Hindi sentence)")
t = time.time()
r = httpx.post(f"{B}/api/v1/tts/inference", headers=H, timeout=60, json={
    "text": "नमस्ते, मैं आपकी बात समझ सकती हूँ।", "model": "vachana-vc-v1", "audio_config": cfg,
    "speaker_embedding": emb})
show(r); open("out_cloned.wav", "wb").write(r.content)
print(f"  {time.time()-t:.2f}s  saved out_cloned.wav  (does it sound like you?)")

print("4) Stock TTS (Timbre v2.5)")
t = time.time()
r = httpx.post(f"{B}/api/v1/tts/inference", headers=H, timeout=60, json={
    "text": "नमस्ते, आप कैसे हैं?", "voice": "Nalini", "model": "timbre-v2.5", "language": "hi-IN", "speed": 1.0, "audio_config": cfg})
show(r); open("out_stock.wav", "wb").write(r.content)
print(f"  {time.time()-t:.2f}s  saved out_stock.wav")
print("\nAll four steps worked. Send me the timings and tell me whether out_cloned.wav sounds like you.")