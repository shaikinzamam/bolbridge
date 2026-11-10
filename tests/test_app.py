import asyncio
import base64
import importlib
import json
import os
from unittest.mock import AsyncMock, call

import httpx
import pytest
from fastapi.testclient import TestClient


def load_app(monkeypatch):
    monkeypatch.setenv("GNANI_API_KEY", "test-gnani")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-llm")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    import app as app_module
    importlib.reload(app_module)
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    app_module.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(500)), timeout=5.0)
    app_module.voices.clear()
    return app_module


@pytest.fixture
def appmod(monkeypatch):
    return load_app(monkeypatch)


def wav_bytes(n=4000):
    return b"RIFF" + b"\x00" * 4 + b"WAVEfmt " + b"\x00" * (n - 16)


def api_client(appmod):
    return TestClient(appmod.app)


def test_gnani_and_translate_retry_transport_error(appmod, monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(appmod.asyncio, "sleep", sleep)
    appmod.client.request = AsyncMock(side_effect=[
        httpx.ReadError("connection closed"),
        httpx.Response(200),
    ])
    appmod.client.post = AsyncMock(side_effect=[
        httpx.ReadError("connection closed"),
        httpx.Response(200, json={"choices": [{"message": {"content": "Hello"}}]}),
    ])

    async def exercise():
        response = await appmod.gnani("GET", "/test")
        translation = await appmod.translate("नमस्ते", "Hindi", "English")
        return response, translation

    response, translation = asyncio.run(exercise())

    assert response.status_code == 200
    assert translation == "Hello"
    assert appmod.client.request.await_count == 2
    assert appmod.client.post.await_count == 2
    assert sleep.await_args_list == [call(0.5), call(0.5)]


def test_gnani_and_translate_raise_502_after_two_transport_errors(appmod, monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(appmod.asyncio, "sleep", sleep)
    appmod.client.request = AsyncMock(side_effect=[
        httpx.ReadError("connection closed"),
        httpx.ConnectError("connection refused"),
    ])
    appmod.client.post = AsyncMock(side_effect=[
        httpx.ReadError("connection closed"),
        httpx.ConnectError("connection refused"),
    ])

    async def exercise():
        for operation in (
            appmod.gnani("GET", "/test"),
            appmod.translate("नमस्ते", "Hindi", "English"),
        ):
            with pytest.raises(appmod.HTTPException) as exc_info:
                await operation
            assert exc_info.value.status_code == 502
            assert exc_info.value.detail == "Could not reach the service. Please try again."

    asyncio.run(exercise())
    assert sleep.await_args_list == [call(0.5), call(0.5)]


def test_health_reports_booleans_only(appmod):
    with api_client(appmod) as ac:
        r = ac.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {
        "gnani_key_configured": True,
        "llm_base_configured": True,
        "llm_key_configured": True,
        "llm_model_configured": True,
        "llm_configured": True,
    }
    assert "test-gnani" not in r.text
    assert "test-llm" not in r.text


def test_successful_turn(appmod):
    async def handler(request: httpx.Request):
        url = str(request.url)
        if url.endswith("/stt/v3"):
            return httpx.Response(200, json={"transcript": "नमस्ते आप कैसे हैं", "detected_language_code": "hi-IN"})
        if url.endswith("/chat/completions"):
            body = json.loads(request.content)
            assert "reasoning_effort" not in body
            return httpx.Response(200, json={"choices": [{"message": {"content": "Hello"}}]})
        if url.endswith("/tts/inference"):
            return httpx.Response(200, content=b"WAVDATA")
        return httpx.Response(404)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn", data={"speaker": "A", "src_lang": "auto", "tgt_lang": "en-IN"},
                    files={"audio_file": ("t.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert j["translation"] == "Hello"
    assert j["detected_lang"] == "hi-IN"
    assert base64.b64decode(j["audio_b64"]) == b"WAVDATA"
    assert j["voice"] == "stock"
    assert set(j["timings"]) == {"stt_ms", "translate_ms", "tts_ms", "total_ms"}


def test_unclear_audio_from_stt(appmod):
    async def handler(request: httpx.Request):
        if str(request.url).endswith("/stt/v3"):
            return httpx.Response(200, json={"transcript": "", "detected_language_code": "hi-IN"})
        return httpx.Response(500)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn", data={"speaker": "A", "src_lang": "auto", "tgt_lang": "en-IN"},
                    files={"audio_file": ("t.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    assert r.json()["status"] == "unclear"
    assert "translate_ms" not in r.json()["timings"]


def test_explicit_same_language_skips_translation(appmod):
    calls = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        if str(request.url).endswith("/stt/v3"):
            return httpx.Response(200, json={"transcript": "hello world", "detected_language_code": "en-IN"})
        if str(request.url).endswith("/tts/inference"):
            return httpx.Response(200, content=b"WAVDATA")
        return httpx.Response(500)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn", data={"speaker": "A", "src_lang": "en-IN", "tgt_lang": "en-IN"},
                    files={"audio_file": ("t.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert j["translation"] == "hello world"
    assert j["voice_used"] == "stock"
    assert j["embedding_present"] is False
    assert not any(u.endswith("/chat/completions") for u in calls)


def test_auto_detect_same_as_target_still_translates(appmod):
    calls = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        if str(request.url).endswith("/stt/v3"):
            return httpx.Response(200, json={"transcript": "hello world", "detected_language_code": "en-IN"})
        if str(request.url).endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "hello translated"}}]})
        if str(request.url).endswith("/tts/inference"):
            return httpx.Response(200, content=b"WAVDATA")
        return httpx.Response(500)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn", data={"speaker": "A", "src_lang": "auto", "tgt_lang": "en-IN"},
                    files={"audio_file": ("t.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    assert r.json()["translation"] == "hello translated"
    assert any(u.endswith("/chat/completions") for u in calls)


def test_voice_used_cloned_when_enrolled(appmod):
    appmod.voices["A"] = {"embedding": "abc", "shape": [1, 768], "dtype": "torch.bfloat16"}
    tts_bodies = []

    async def handler(request: httpx.Request):
        url = str(request.url)
        if url.endswith("/stt/v3"):
            return httpx.Response(200, json={"transcript": "नमस्ते जी", "detected_language_code": "hi-IN"})
        if url.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "Hello"}}]})
        if url.endswith("/tts/inference"):
            tts_bodies.append(json.loads(request.content))
            return httpx.Response(200, content=b"WAVDATA")
        return httpx.Response(404)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn", data={"speaker": "A", "src_lang": "auto", "tgt_lang": "en-IN"},
                    files={"audio_file": ("t.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert j["voice_used"] == "cloned"
    assert j["voice"] == "cloned"
    assert j["embedding_present"] is True
    assert tts_bodies[0]["model"] == "vachana-vc-v1"
    assert "speaker_embedding" in tts_bodies[0]


def test_voice_preview_stock_and_cloned(appmod):
    async def stock_handler(request: httpx.Request):
        if str(request.url).endswith("/tts/inference"):
            body = json.loads(request.content)
            assert body["model"] == "timbre-v2.5"
            assert body["language"] == "te-IN"
            return httpx.Response(200, content=b"STOCK")
        return httpx.Response(404)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(stock_handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/voice-preview", data={"speaker": "A", "tgt_lang": "te-IN"})
    assert r.status_code == 200
    j = r.json()
    assert j["voice_used"] == "stock"
    assert j["embedding_present"] is False
    assert base64.b64decode(j["audio_b64"]) == b"STOCK"

    appmod.voices["A"] = {"embedding": "abc", "shape": [1, 768], "dtype": "torch.bfloat16"}

    async def cloned_handler(request: httpx.Request):
        if str(request.url).endswith("/tts/inference"):
            body = json.loads(request.content)
            assert body["model"] == "vachana-vc-v1"
            assert body["speaker_embedding"]["embedding"] == "abc"
            return httpx.Response(200, content=b"CLONED")
        return httpx.Response(404)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(cloned_handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/voice-preview", data={"speaker": "A", "tgt_lang": "en-IN"})
    assert r.status_code == 200
    j = r.json()
    assert j["voice_used"] == "cloned"
    assert j["embedding_present"] is True
    assert base64.b64decode(j["audio_b64"]) == b"CLONED"


def test_remove_voice(appmod):
    appmod.voices["A"] = {"embedding": "abc", "shape": [1, 768], "dtype": "torch.bfloat16"}
    appmod.save_voices()
    with api_client(appmod) as ac:
        r = ac.post("/api/remove-voice", data={"speaker": "A"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    assert "A" not in appmod.voices
    assert "A" not in json.loads(appmod.VOICES_FILE.read_text(encoding="utf-8"))


def test_enrollment_replaces_existing_embedding(appmod, tmp_path):
    appmod.VOICES_FILE = tmp_path / "voices.json"
    appmod.save_voices()
    calls = {"n": 0}

    async def handler(request: httpx.Request):
        if str(request.url).endswith("/api/v1/tts/voice-clone/embeddings"):
            calls["n"] += 1
            emb = "first" if calls["n"] == 1 else "second"
            return httpx.Response(200, json={"data": {"voice_clone_embedding": {"embedding": emb, "shape": [1, 768], "dtype": "torch.bfloat16"}}})
        return httpx.Response(404)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r1 = ac.post("/api/enroll", data={"speaker": "A", "consent": "true"},
                     files={"audio_file": ("v.wav", wav_bytes(400000), "audio/wav")})
        r2 = ac.post("/api/enroll", data={"speaker": "A", "consent": "true"},
                     files={"audio_file": ("v.wav", wav_bytes(400000), "audio/wav")})
    assert r1.status_code == 200 and r2.status_code == 200
    assert appmod.voices["A"]["embedding"] == "second"
    saved = json.loads(appmod.VOICES_FILE.read_text(encoding="utf-8"))
    assert saved["A"]["embedding"] == "second"



def test_gnani_429_then_success(appmod, monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(appmod.asyncio, "sleep", fake_sleep)
    attempts = {"n": 0}

    async def handler(request: httpx.Request):
        if str(request.url).endswith("/stt/v3"):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(429, json={"message": "rate"})
            return httpx.Response(200, json={"transcript": "नमस्ते जी", "detected_language_code": "hi-IN"})
        if str(request.url).endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "Hello"}}]})
        if str(request.url).endswith("/tts/inference"):
            return httpx.Response(200, content=b"WAVDATA")
        return httpx.Response(404)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn", data={"speaker": "A", "src_lang": "auto", "tgt_lang": "en-IN"},
                    files={"audio_file": ("t.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    assert attempts["n"] == 2
    assert sleeps == [0.7]


@pytest.mark.parametrize("code,msg", [
    (401, "Gnani API key"),
    (402, "credits"),
    (500, "Gnani 500"),
])
def test_gnani_error_messages(appmod, code, msg):
    async def handler(request: httpx.Request):
        return httpx.Response(code, json={"message": "boom"})

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn", data={"speaker": "A", "src_lang": "auto", "tgt_lang": "en-IN"},
                    files={"audio_file": ("t.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 502
    assert msg in r.json()["detail"]


def test_llm_error(appmod):
    async def handler(request: httpx.Request):
        if str(request.url).endswith("/stt/v3"):
            return httpx.Response(200, json={"transcript": "नमस्ते जी", "detected_language_code": "hi-IN"})
        if str(request.url).endswith("/chat/completions"):
            return httpx.Response(500, text="fail")
        return httpx.Response(404)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn", data={"speaker": "A", "src_lang": "auto", "tgt_lang": "en-IN"},
                    files={"audio_file": ("t.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 502
    assert "Translation LLM error 500" in r.json()["detail"]


def test_enrollment_without_consent(appmod):
    with api_client(appmod) as ac:
        r = ac.post("/api/enroll", data={"speaker": "A", "consent": "false"},
                    files={"audio_file": ("v.wav", wav_bytes(400000), "audio/wav")})
    assert r.status_code == 400
    assert "Consent" in r.json()["detail"]


def test_enrollment_too_short(appmod):
    with api_client(appmod) as ac:
        r = ac.post("/api/enroll", data={"speaker": "A", "consent": "true"},
                    files={"audio_file": ("v.wav", wav_bytes(1000), "audio/wav")})
    assert r.status_code == 400
    assert "too short" in r.json()["detail"].lower()


@pytest.mark.parametrize("payload", [
    {"embedding": "abc", "shape": [1, 768], "dtype": "torch.bfloat16"},
    {"data": {"voice_clone_embedding": {"embedding": "abc", "shape": [1, 768], "dtype": "torch.bfloat16"}}},
    {"result": [{"embedding": "abc", "shape": [1, 768], "dtype": "torch.bfloat16"}]},
])
def test_embedding_extraction_layouts(appmod, payload):
    e = appmod.extract_embedding(payload)
    assert e and e["embedding"] == "abc"
    assert e["shape"] == [1, 768]


def test_reasoning_effort_sent_only_when_set(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    appmod = load_app(monkeypatch)
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    bodies = []

    async def handler(request: httpx.Request):
        if str(request.url).endswith("/stt/v3"):
            return httpx.Response(200, json={"transcript": "नमस्ते जी", "detected_language_code": "hi-IN"})
        if str(request.url).endswith("/chat/completions"):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "Hello"}}]})
        if str(request.url).endswith("/tts/inference"):
            return httpx.Response(200, content=b"WAVDATA")
        return httpx.Response(404)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn", data={"speaker": "A", "src_lang": "auto", "tgt_lang": "en-IN"},
                    files={"audio_file": ("t.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    assert bodies[0]["reasoning_effort"] == "low"


def test_turn_under_two_words_is_unclear(appmod):
    calls = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        if str(request.url).endswith("/stt/v3"):
            return httpx.Response(200, json={"transcript": "नमस्ते।", "detected_language_code": "hi-IN"})
        return httpx.Response(500)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn", data={"speaker": "A", "src_lang": "auto", "tgt_lang": "en-IN"},
                    files={"audio_file": ("t.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "unclear"
    assert j.get("transcript") == "नमस्ते।"
    assert "translate_ms" not in j["timings"]
    assert not any(u.endswith("/chat/completions") for u in calls)
    assert not any(u.endswith("/tts/inference") for u in calls)


def test_turn_two_words_translates(appmod):
    async def handler(request: httpx.Request):
        url = str(request.url)
        if url.endswith("/stt/v3"):
            return httpx.Response(200, json={"transcript": "hello world", "detected_language_code": "en-IN"})
        if url.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "नमस्ते दुनिया"}}]})
        if url.endswith("/tts/inference"):
            return httpx.Response(200, content=b"WAVDATA")
        return httpx.Response(404)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn", data={"speaker": "A", "src_lang": "auto", "tgt_lang": "hi-IN"},
                    files={"audio_file": ("t.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert j["translation"] == "नमस्ते दुनिया"


def test_auto_detect_english_transcript_still_translates(appmod):
    calls = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        url = str(request.url)
        if url.endswith("/stt/v3"):
            return httpx.Response(200, json={"transcript": "Can you hear me?", "detected_language_code": "en-IN"})
        if url.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "నమస్కారం"}}]})
        if url.endswith("/tts/inference"):
            return httpx.Response(200, content=b"WAVDATA")
        return httpx.Response(500)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn", data={"speaker": "A", "src_lang": "auto", "tgt_lang": "te-IN"},
                    files={"audio_file": ("t.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert j["detected_lang"] == "en-IN"
    assert j["translation"] == "నమస్కారం"
    assert any(u.endswith("/chat/completions") for u in calls)


def test_identical_translation_retried_then_unclear(appmod):
    llm_bodies = []

    async def handler(request: httpx.Request):
        url = str(request.url)
        if url.endswith("/stt/v3"):
            return httpx.Response(200, json={"transcript": "hello world", "detected_language_code": "en-IN"})
        if url.endswith("/chat/completions"):
            llm_bodies.append(json.loads(request.content))
            result = "Hello, world!" if len(llm_bodies) == 1 else "HELLO WORLD."
            return httpx.Response(200, json={"choices": [{"message": {"content": result}}]})
        return httpx.Response(500)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn", data={"speaker": "A", "src_lang": "en-IN", "tgt_lang": "te-IN"},
                    files={"audio_file": ("t.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    assert r.json()["status"] == "unclear"
    assert len(llm_bodies) == 2
    assert "previous translation was unchanged" in llm_bodies[1]["messages"][0]["content"].lower()


def test_turn_text_success_skips_stt(appmod):
    calls = []

    async def handler(request: httpx.Request):
        url = str(request.url)
        calls.append(url)
        if url.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "నమస్కారం"}}]})
        if url.endswith("/tts/inference"):
            return httpx.Response(200, content=b"WAVDATA")
        return httpx.Response(500)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    with api_client(appmod) as ac:
        r = ac.post("/api/turn-text", data={
            "speaker": "A", "src_lang": "en-IN", "tgt_lang": "te-IN", "text": "hello there",
        })
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert j["transcript"] == "hello there"
    assert j["translation"] == "నమస్కారం"
    assert base64.b64decode(j["audio_b64"]) == b"WAVDATA"
    assert "stt_ms" in j["timings"]
    assert not any(url.endswith("/stt/v3") for url in calls)


def test_turn_text_empty_is_rejected(appmod):
    with api_client(appmod) as ac:
        r = ac.post("/api/turn-text", data={
            "speaker": "A", "src_lang": "en-IN", "tgt_lang": "te-IN", "text": "   ",
        })
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_duplicate_client_turn_id_runs_pipeline_once(appmod):
    calls = []

    async def handler(request: httpx.Request):
        url = str(request.url)
        calls.append(url)
        if url.endswith("/stt/v3"):
            return httpx.Response(200, json={"transcript": "hello world", "detected_language_code": "en-IN"})
        if url.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "నమస్కారం"}}]})
        if url.endswith("/tts/inference"):
            return httpx.Response(200, content=b"WAVDATA")
        return httpx.Response(500)

    appmod.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    data = {"speaker": "A", "src_lang": "auto", "tgt_lang": "te-IN", "client_turn_id": "test-client:1"}
    files = {"audio_file": ("t.wav", wav_bytes(), "audio/wav")}
    with api_client(appmod) as ac:
        first = ac.post("/api/turn", data=data, files=files)
        second = ac.post("/api/turn", data=data, files=files)
    assert first.status_code == second.status_code == 200
    assert first.json()["turn_id"] == second.json()["turn_id"]
    assert sum(url.endswith("/stt/v3") for url in calls) == 1
    assert sum(url.endswith("/chat/completions") for url in calls) == 1
    assert sum(url.endswith("/tts/inference") for url in calls) == 1
