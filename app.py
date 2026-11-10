"""BolBridge backend.

Pipeline for one spoken turn:
  audio -> Gnani Prisma (STT, language auto-detect) -> LLM translation -> Gnani Timbre (voice-cloned TTS)
"""
import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

logger = logging.getLogger("bolbridge")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

GNANI_BASE = "https://api.vachana.ai"
GNANI_KEY = os.getenv("GNANI_API_KEY", "")
LLM_BASE = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")


def _llm_reasoning_effort():
    return os.getenv("LLM_REASONING_EFFORT", "").strip().lower()


def _word_count(text: str) -> int:
    return len(re.sub(r"[^\w\s]", "", text or "").split())


def _normalized_translation(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text or "").casefold().split())


async def _idempotent_turn(
    endpoint: str,
    client_turn_id: str,
    operation,
) -> dict:
    if not client_turn_id:
        return await operation()
    if len(client_turn_id) > 120 or not re.fullmatch(r"[A-Za-z0-9._:-]+", client_turn_id):
        raise HTTPException(400, "Invalid client turn id.")

    key = (endpoint, client_turn_id)
    lock = turn_locks.setdefault(key, asyncio.Lock())
    async with lock:
        now = time.monotonic()
        for cached_key, (created, _) in list(turn_cache.items()):
            if now - created > TURN_CACHE_TTL:
                turn_cache.pop(cached_key, None)
        for stale_key, stale_lock in list(turn_locks.items()):
            if stale_key not in turn_cache and not stale_lock.locked():
                turn_locks.pop(stale_key, None)
        cached = turn_cache.get(key)
        if cached:
            turn_cache.move_to_end(key)
            return cached[1]

        result = await operation()
        turn_cache[key] = (time.monotonic(), result)
        turn_cache.move_to_end(key)
        while len(turn_cache) > TURN_CACHE_SIZE:
            evicted, _ = turn_cache.popitem(last=False)
            evicted_lock = turn_locks.get(evicted)
            if evicted_lock and not evicted_lock.locked():
                turn_locks.pop(evicted, None)
        return result


STT_MODEL = os.getenv("STT_MODEL", "prisma-v3.0")
TTS_MODEL = os.getenv("TTS_MODEL", "timbre-v2.5")
FALLBACK_VOICE = os.getenv("FALLBACK_VOICE", "Nalini")

ROOT = Path(__file__).parent
VOICES_FILE = ROOT / "voices.json"
MAX_AUDIO_BYTES = 5 * 1024 * 1024  # ~30s 16kHz mono WAV with headroom
MIN_ENROLL_BYTES = 10000  # ~0.3s WAV floor; client enforces 10s quality checks

# Codes follow Gnani's BCP-47 style (hi-IN). Verify against docs.gnani.ai if a language fails.
LANGS = {
    "hi-IN": "Hindi",
    "en-IN": "English",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "pa-IN": "Punjabi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
}
SCRIPTS = {
    "Hindi": "Devanagari", "Marathi": "Devanagari", "Tamil": "Tamil", "Telugu": "Telugu",
    "Kannada": "Kannada", "Malayalam": "Malayalam", "Punjabi": "Gurmukhi",
    "Bengali": "Bengali", "Gujarati": "Gujarati", "English": "Latin",
}
PREVIEW_TEXT = {
    "hi-IN": "नमस्ते, यह मेरी आवाज़ का नमूना है।",
    "en-IN": "Hello, this is a sample of my voice.",
    "ta-IN": "வணக்கம், இது என் குரலின் மாதிரி.",
    "te-IN": "నమస్తే, ఇది నా స్వరం యొక్క నమూనా.",
    "kn-IN": "ನಮಸ್ಕಾರ, ಇದು ನನ್ನ ಧ್ವನಿಯ ಮಾದರಿ.",
    "ml-IN": "നമസ്കാരം, ഇത് എന്റെ ശബ്ദത്തിന്റെ മാതൃകയാണ്.",
    "mr-IN": "नमस्कार, हा माझ्या आवाजाचा नमुना आहे.",
    "pa-IN": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਇਹ ਮੇਰੀ ਆਵਾਜ਼ ਦਾ ਨਮੂਨਾ ਹੈ।",
    "bn-IN": "নমস্কার, এটি আমার কণ্ঠের একটি নমুনা।",
    "gu-IN": "નમસ્તે, આ મારા અવાજનો નમૂનો છે.",
}

app = FastAPI(title="BolBridge")
client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
# Some Gnani accounts allow only ONE active request at a time, so we queue every call.
gate = asyncio.Semaphore(1)
TURN_CACHE_TTL = 300
TURN_CACHE_SIZE = 32
turn_cache: OrderedDict[tuple[str, str], tuple[float, dict]] = OrderedDict()
turn_locks: dict[tuple[str, str], asyncio.Lock] = {}

voices: dict = {}
if VOICES_FILE.exists():
    try:
        voices = json.loads(VOICES_FILE.read_text(encoding="utf-8"))
    except Exception:
        voices = {}
logger.info("voices loaded: %d", len(voices))


def save_voices():
    VOICES_FILE.write_text(json.dumps(voices, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- Gnani helpers
def explain_gnani_error(r: httpx.Response) -> str:
    try:
        body = r.json()
    except Exception:
        body = {}
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        code, msg = detail.get("error_code", ""), detail.get("message", "")
    else:
        code = ""
        msg = (body.get("message") if isinstance(body, dict) else "") or r.text[:200]
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            msg = body["error"].get("message", msg)
    hints = {
        401: "Gnani API key missing or invalid. Check GNANI_API_KEY in .env.",
        402: "Out of Gnani credits.",
        403: "This API key does not have access to that service (scope).",
        429: "Gnani rate limit hit. Wait a moment and try again.",
    }
    return f"Gnani {r.status_code}{' ' + code if code else ''}: {hints.get(r.status_code, msg)}"


async def gnani(method: str, path: str, **kw) -> httpx.Response:
    if not GNANI_KEY:
        raise HTTPException(500, "GNANI_API_KEY is not set. Add it to .env and restart.")
    headers = {"X-API-Key-ID": GNANI_KEY}
    async with gate:
        r = None
        for attempt in range(4):
            try:
                r = await client.request(method, GNANI_BASE + path, headers=headers, **kw)
            except httpx.TransportError:
                await asyncio.sleep(0.5)
                try:
                    r = await client.request(method, GNANI_BASE + path, headers=headers, **kw)
                except httpx.TransportError as exc:
                    raise HTTPException(502, "Could not reach the service. Please try again.") from exc
            if r.status_code != 429:
                break
            await asyncio.sleep(0.7 * (attempt + 1))
    if r.status_code >= 400:
        raise HTTPException(502, explain_gnani_error(r))
    return r


async def stt(wav: bytes, lang: str) -> tuple[str, str]:
    code = "indic-auto" if lang == "auto" else lang
    r = await gnani(
        "POST", "/stt/v3",
        files={"audio_file": ("turn.wav", wav, "audio/wav")},
        data={"model": STT_MODEL, "language_code": code},
    )
    try:
        j = r.json()
    except Exception:
        raise HTTPException(502, "Transcription failed: invalid response.")
    if isinstance(j, dict) and j.get("success") is False:
        raise HTTPException(502, f"Transcription failed: {j}")
    transcript = ((j.get("transcript") if isinstance(j, dict) else "") or "").strip()
    detected = ((j.get("detected_language_code") if isinstance(j, dict) else "") or "").strip()
    if not detected and lang != "auto":
        detected = lang
    return transcript, detected


async def translate(text: str, src_name: str, tgt_name: str, strict: bool = False) -> str:
    if not (LLM_BASE and LLM_MODEL):
        raise HTTPException(500, "Translation LLM not configured. Set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in .env.")
    script = SCRIPTS.get(tgt_name, "the native script")
    system = (
        "You are the translation engine of a live two-person phone conversation. "
        f"Translate the speaker's words into natural, spoken {tgt_name}, written in {script} script. "
        "Keep it conversational, keep the meaning and tone, keep names and numbers exact. "
        "The input may mix languages (for example Hinglish); handle it as one sentence. "
        "The transcript may be written in a different script than the language actually spoken (for example English words written in Devanagari or Telugu letters); work out the real meaning and translate that into the target language. "
        "Never repeat the input unchanged. "
        + ("The previous translation was unchanged; translate the meaning into the target language, changing every word that has a target-language equivalent. " if strict else "")
        + "Output ONLY the translation, no quotes or notes. "
        "If the input is unintelligible or just noise, output exactly: [UNCLEAR]"
    )
    headers = {"Authorization": f"Bearer {LLM_KEY}"} if LLM_KEY else {}
    body = {
        "model": LLM_MODEL, "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Source language: {src_name or 'unknown'}\n\n{text}"},
        ],
    }
    effort = _llm_reasoning_effort()
    if effort in {"low", "medium", "high"}:
        body["reasoning_effort"] = effort
    try:
        r = await client.post(f"{LLM_BASE}/chat/completions", headers=headers, json=body)
    except httpx.TransportError:
        await asyncio.sleep(0.5)
        try:
            r = await client.post(f"{LLM_BASE}/chat/completions", headers=headers, json=body)
        except httpx.TransportError as exc:
            raise HTTPException(502, "Could not reach the service. Please try again.") from exc
    if r.status_code >= 400:
        raise HTTPException(502, f"Translation LLM error {r.status_code}: {r.text[:200]}")
    try:
        j = r.json()
        return j["choices"][0]["message"]["content"].strip()
    except Exception:
        raise HTTPException(502, "Translation LLM error: invalid response.")


async def speak(text: str, speaker: str, tgt_code: str) -> tuple[bytes, str]:
    audio_cfg = {"sample_rate": 24000, "num_channels": 1, "sample_width": 2,
                 "encoding": "linear_pcm", "container": "wav"}
    emb = voices.get(speaker)
    if emb:
        body = {"text": text, "model": "vachana-vc-v1", "audio_config": audio_cfg, "speaker_embedding": emb}
        kind = "cloned"
    else:
        body = {"text": text, "voice": FALLBACK_VOICE, "model": TTS_MODEL, "language": tgt_code,
                "speed": 1.0, "audio_config": audio_cfg}
        kind = "stock"
    r = await gnani("POST", "/api/v1/tts/inference", json=body)
    return r.content, kind


# ---------------------------------------------------------------- routes
def extract_embedding(j):
    """Find the embedding anywhere in Gnani's response (docs show several layouts)."""
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


@app.get("/api/state")
async def state():
    return {
        "languages": LANGS,
        "enrolled": {"A": "A" in voices, "B": "B" in voices},
        "gnani_key": bool(GNANI_KEY),
        "llm": bool(LLM_BASE and LLM_MODEL),
    }


@app.get("/api/health")
async def health():
    return {
        "gnani_key_configured": bool(GNANI_KEY),
        "llm_base_configured": bool(LLM_BASE),
        "llm_key_configured": bool(LLM_KEY),
        "llm_model_configured": bool(LLM_MODEL),
        "llm_configured": bool(LLM_BASE and LLM_MODEL),
    }


@app.post("/api/enroll")
async def enroll(speaker: str = Form(...), consent: str = Form("false"), audio_file: UploadFile = File(...)):
    if speaker not in ("A", "B"):
        raise HTTPException(400, "speaker must be A or B")
    if consent.lower() != "true":
        raise HTTPException(400, "Consent is required: only clone your own voice, or someone who agreed.")
    data = await audio_file.read()
    if len(data) < 320000:
        raise HTTPException(400, "Recording too short. Speak clearly for at least 10 seconds.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(400, "Recording too large. Use a 10-30 second clip.")
    r = await gnani("POST", "/api/v1/tts/voice-clone/embeddings",
                    files={"audio_file": ("voice.wav", data, "audio/wav")})
    try:
        j = r.json()
    except Exception:
        raise HTTPException(502, "Voice cloning returned an invalid response.")
    e = extract_embedding(j)
    if not e:
        raise HTTPException(502, f"Voice cloning returned no embedding: {str(j)[:200]}")
    voices[speaker] = e
    save_voices()
    return {"ok": True}


@app.post("/api/reset-voices")
async def reset_voices():
    voices.clear()
    save_voices()
    return {"ok": True}


@app.post("/api/remove-voice")
async def remove_voice(speaker: str = Form(...)):
    if speaker not in ("A", "B"):
        raise HTTPException(400, "speaker must be A or B")
    if speaker in voices:
        voices.pop(speaker, None)
        save_voices()
    return {"ok": True}


@app.post("/api/voice-preview")
async def voice_preview(speaker: str = Form(...), tgt_lang: str = Form("en-IN")):
    if speaker not in ("A", "B"):
        raise HTTPException(400, "speaker must be A or B")
    if tgt_lang not in LANGS:
        raise HTTPException(400, "Unknown target language.")
    text = PREVIEW_TEXT.get(tgt_lang, PREVIEW_TEXT["en-IN"])
    audio, kind = await speak(text, speaker, tgt_lang)
    logger.info("voice-preview speaker=%s voice_path=%s audio_bytes=%s", speaker, kind, len(audio))
    return {
        "voice_used": kind,
        "embedding_present": speaker in voices,
        "audio_b64": base64.b64encode(audio).decode(),
    }


@app.post("/api/turn")
async def turn(
    speaker: str = Form(...),
    src_lang: str = Form("auto"),
    tgt_lang: str = Form(...),
    audio_file: UploadFile = File(...),
    client_turn_id: str = Form(""),
):
    _validate_turn_languages(speaker, src_lang, tgt_lang)

    async def process():
        turn_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        wav = await audio_file.read()
        if len(wav) < 1500:
            logger.info("turn id=%s speaker=%s wav_bytes=%s src_lang=%s status=rejected_empty", turn_id, speaker, len(wav), src_lang)
            raise HTTPException(400, "Recording was empty. Hold the button and speak.")
        if len(wav) > MAX_AUDIO_BYTES:
            logger.info("turn id=%s speaker=%s wav_bytes=%s src_lang=%s status=rejected_too_large", turn_id, speaker, len(wav), src_lang)
            raise HTTPException(400, "Recording too large. Keep turns under 30 seconds.")

        transcript, detected = await stt(wav, src_lang)
        stt_ms = round((time.perf_counter() - t0) * 1000)
        return await _finish_turn(
            turn_id, speaker, src_lang, tgt_lang, transcript, detected, len(wav), stt_ms, t0
        )

    return await _idempotent_turn("audio", client_turn_id, process)


@app.post("/api/turn-text")
async def turn_text(
    speaker: str = Form(...),
    src_lang: str = Form("auto"),
    tgt_lang: str = Form(...),
    text: str = Form(...),
    client_turn_id: str = Form(""),
):
    _validate_turn_languages(speaker, src_lang, tgt_lang)
    transcript = text.strip()
    if not transcript:
        raise HTTPException(400, "Text cannot be empty.")

    async def process():
        turn_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        detected = src_lang if src_lang != "auto" else "unknown"
        return await _finish_turn(
            turn_id, speaker, src_lang, tgt_lang, transcript, detected, 0, 0, t0
        )

    return await _idempotent_turn("text", client_turn_id, process)


def _validate_turn_languages(speaker: str, src_lang: str, tgt_lang: str):
    if speaker not in ("A", "B"):
        raise HTTPException(400, "speaker must be A or B")
    if tgt_lang not in LANGS:
        raise HTTPException(400, "Unknown target language.")
    if src_lang != "auto" and src_lang not in LANGS:
        raise HTTPException(400, "Unknown source language.")


async def _finish_turn(
    turn_id: str,
    speaker: str,
    src_lang: str,
    tgt_lang: str,
    transcript: str,
    detected: str,
    wav_bytes: int,
    stt_ms: int,
    started: float,
) -> dict:
    src_name = LANGS.get(detected, detected)
    tgt_name = LANGS[tgt_lang]
    translation_skipped = src_lang != "auto" and src_lang == tgt_lang
    skip_reason = "explicit_same_language" if translation_skipped else "none"

    if _word_count(transcript) < 2:
        logger.info(
            "turn id=%s speaker=%s wav_bytes=%s src_lang=%s detected=%s transcript_chars=%s "
            "translation_skipped=true skip_reason=too_few_words voice_path=none stt_ms=%s "
            "translate_ms=0 tts_ms=0 total_ms=%s status=unclear",
            turn_id, speaker, wav_bytes, src_lang, detected, len(transcript),
            stt_ms, round((time.perf_counter() - started) * 1000),
        )
        return {
            "status": "unclear", "turn_id": turn_id, "transcript": transcript,
            "message": "Could not catch that. Please say it again.",
            "timings": {"stt_ms": stt_ms},
        }

    translate_started = time.perf_counter()
    if translation_skipped:
        translation = transcript
    else:
        translation = await translate(transcript, src_name, tgt_name)
        if _normalized_translation(translation) == _normalized_translation(transcript):
            translation = await translate(transcript, src_name, tgt_name, strict=True)
            if _normalized_translation(translation) == _normalized_translation(transcript):
                translate_ms = round((time.perf_counter() - translate_started) * 1000)
                total_ms = round((time.perf_counter() - started) * 1000)
                logger.info(
                    "turn id=%s speaker=%s wav_bytes=%s src_lang=%s detected=%s transcript_chars=%s "
                    "translation_skipped=false skip_reason=identical_output voice_path=none stt_ms=%s "
                    "translate_ms=%s tts_ms=0 total_ms=%s status=unclear",
                    turn_id, speaker, wav_bytes, src_lang, detected, len(transcript),
                    stt_ms, translate_ms, total_ms,
                )
                return {
                    "status": "unclear", "turn_id": turn_id, "transcript": transcript,
                    "message": "The translation matched the original. Please try again.",
                    "timings": {"stt_ms": stt_ms, "translate_ms": translate_ms, "total_ms": total_ms},
                }

    translate_ms = round((time.perf_counter() - translate_started) * 1000)
    if translation.strip().casefold() == "[unclear]":
        total_ms = round((time.perf_counter() - started) * 1000)
        logger.info(
            "turn id=%s speaker=%s wav_bytes=%s src_lang=%s detected=%s transcript_chars=%s "
            "translation_skipped=%s skip_reason=%s voice_path=none stt_ms=%s translate_ms=%s "
            "tts_ms=0 total_ms=%s status=unclear_llm",
            turn_id, speaker, wav_bytes, src_lang, detected, len(transcript),
            str(translation_skipped).lower(), skip_reason, stt_ms, translate_ms, total_ms,
        )
        return {
            "status": "unclear", "turn_id": turn_id, "transcript": transcript,
            "message": "That sounded unclear. Please say it again.",
            "timings": {"stt_ms": stt_ms, "translate_ms": translate_ms, "total_ms": total_ms},
        }

    tts_started = time.perf_counter()
    audio, kind = await speak(translation, speaker, tgt_lang)
    t3 = time.perf_counter()
    timings = {
        "stt_ms": stt_ms,
        "translate_ms": translate_ms,
        "tts_ms": round((t3 - tts_started) * 1000),
        "total_ms": round((t3 - started) * 1000),
    }
    logger.info(
        "turn id=%s speaker=%s wav_bytes=%s src_lang=%s detected=%s transcript_chars=%s "
        "translation_skipped=%s skip_reason=%s voice_path=%s stt_ms=%s translate_ms=%s "
        "tts_ms=%s total_ms=%s status=ok",
        turn_id, speaker, wav_bytes, src_lang, detected, len(transcript),
        str(translation_skipped).lower(), skip_reason, kind,
        timings["stt_ms"], timings["translate_ms"], timings["tts_ms"], timings["total_ms"],
    )
    return {
        "status": "ok",
        "turn_id": turn_id,
        "transcript": transcript,
        "detected_lang": detected,
        "detected_name": src_name,
        "translation": translation,
        "voice": kind,
        "voice_used": kind,
        "embedding_present": speaker in voices,
        "audio_b64": base64.b64encode(audio).decode(),
        "timings": timings,
    }


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.on_event("shutdown")
async def _close():
    await client.aclose()
