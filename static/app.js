"use strict";

const $ = (s, el = document) => el.querySelector(s);
const panels = { A: $('[data-side="A"]'), B: $('[data-side="B"]') };
const other = { A: "B", B: "A" };
const MAX_TALK_MS = 25000; // Gnani REST STT limit is 30 s per request
const ENROLL_MAX_MS = 20000;
const ENROLL_MIN_MS = 10000;
let busy = false;
let requestInFlight = false;
let turnPending = false;
let active = null; // {side, rec}
let holdTimer = null;
let recordingClock = null;
const holding = { A: false, B: false };
const lastSample = { A: null, B: null }; // {blob, url, ms}
const clonedVoices = { A: false, B: false };
let tapMode = ls("bb_tap") !== "0"; // default ON: click to start, click to stop
let turnNo = 0;
let recordingToken = 0;
let audioPlaying = false;
let playbackCooldownUntil = 0;
let clientInstanceId = null;

// ---------------------------------------------------------------- helpers
function toast(msg, isErr = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "show" + (isErr ? " err" : "");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => (t.className = ""), isErr ? 6000 : 2800);
}
function setStatus(text, live = false) {
  const s = $("#status");
  s.textContent = text;
  s.classList.toggle("live", live);
}
function setState(side, st) {
  panels[side].dataset.state = st;
  $(".talk-label", panels[side]).textContent = st === "listening" ? "Recording" : st === "busy" ? "Working" : tapMode ? "Tap to talk" : "Hold to talk";
  $("[data-talk]", panels[side]).setAttribute("aria-label", `${tapMode ? "Tap" : "Hold"} to talk as speaker ${side}`);
}
function ls(key, val) {
  try {
    if (val === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, val);
  } catch (e) { return null; }
}
const hearLang = (side) => $("[data-hear]", panels[side]).value;
const srcLang = (side) => $("[data-src]", panels[side]).value;

function addBubble(side, cls, text, caption, heardAs) {
  const feed = $("[data-feed]", panels[side]);
  const empty = $(".empty", feed);
  if (empty) empty.remove();
  const b = document.createElement("div");
  b.className = "bubble " + cls;
  if (caption) {
    const s = document.createElement("small");
    s.textContent = caption;
    b.appendChild(s);
  }
  b.appendChild(document.createTextNode(text));
  if (heardAs) {
    const s2 = document.createElement("small");
    s2.className = "heard-as";
    s2.textContent = heardAs;
    b.appendChild(s2);
  }
  feed.appendChild(b);
  feed.scrollTop = feed.scrollHeight;
}

function stopTracks(stream) {
  if (stream) stream.getTracks().forEach((t) => t.stop());
}

// ---------------------------------------------------------------- audio: record + convert to 16 kHz mono WAV
class Recorder {
  static async start() {
    const raw = $("#raw").checked;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: !raw, autoGainControl: true },
    });
    const r = new Recorder(stream);
    r.rec.start(250);
    return r;
  }
  constructor(stream) {
    this.stream = stream;
    this.chunks = [];
    this.rec = new MediaRecorder(stream);
    this.rec.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.t0 = performance.now();
  }
  stop() {
    if (this._stopping) return this._stopping;
    this._stopping = new Promise((res) => {
      const done = () => {
        stopTracks(this.stream);
        const blob = this.chunks.length ? new Blob(this.chunks, { type: this.rec.mimeType || "audio/webm" }) : null;
        res(blob ? { blob, ms: performance.now() - this.t0 } : null);
      };
      if (this.rec.state === "inactive") return done();
      this.rec.onstop = done;
      this.rec.onerror = done;
      try { this.rec.stop(); } catch (e) { done(); }
    });
    return this._stopping;
  }
}

async function decodeToMono(blob) {
  if (!blob || blob.size < 1500) throw new Error("Recording was empty. Hold the button for at least 1 second while speaking.");
  const buf = await blob.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) throw new Error("Audio decoding is not supported in this browser.");
  const ctx = new Ctx();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(buf);
  } catch (e) {
    throw new Error("Could not read that recording. Hold the button a little longer and speak clearly, then try again.");
  } finally {
    try { await ctx.close(); } catch (_) {}
  }
  const rate = 16000;
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * rate)), rate);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return { samples: rendered.getChannelData(0), rate, duration: rendered.duration };
}

async function toWav16k(blob) {
  const { samples, rate } = await decodeToMono(blob);
  return encodeWav(samples, rate);
}

function analyzeSamples(samples, rate) {
  const n = samples.length || 1;
  let sumSq = 0, peak = 0, silent = 0;
  const silenceTh = 0.01;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i];
    const a = Math.abs(v);
    sumSq += v * v;
    if (a > peak) peak = a;
    if (a < silenceTh) silent++;
  }
  return {
    duration: n / rate,
    rms: Math.sqrt(sumSq / n),
    peak,
    clippedRatio: samples.length ? (Array.prototype.filter ? 0 : 0) : 0, // computed below
    silenceRatio: silent / n,
  };
}
function analyzeEnrollment(samples, rate) {
  const n = samples.length || 1;
  let sumSq = 0, peak = 0, silent = 0, clipped = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i];
    const a = Math.abs(v);
    sumSq += v * v;
    if (a > peak) peak = a;
    if (a < 0.01) silent++;
    if (a >= 0.98) clipped++;
  }
  return {
    duration: n / rate,
    rms: Math.sqrt(sumSq / n),
    peak,
    clippedRatio: clipped / n,
    silenceRatio: silent / n,
  };
}

function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

async function playAudio(a) {
  if ($("#mute").checked) {
    toast("Playback muted for testing.");
    return;
  }
  if (audioPlaying) {
    toast("Audio is already playing.", true);
    return;
  }
  audioPlaying = true;
  const finished = new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      audioPlaying = false;
      playbackCooldownUntil = performance.now() + 600;
      resolve();
    };
    a.onended = finish;
    a.onerror = finish;
  });
  try {
    await a.play();
  } catch (e) {
    audioPlaying = false;
    playbackCooldownUntil = 0;
    toast("Tap the page once, then try again (browser blocked audio).", true);
    return;
  }
  await finished;
}

async function playB64(b64) {
  await playAudio(new Audio("data:audio/wav;base64," + b64));
}

function getClientInstanceId() {
  if (!clientInstanceId) {
    clientInstanceId = window.crypto && crypto.randomUUID
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }
  return clientInstanceId;
}

// ---------------------------------------------------------------- turn flow
async function submitTurn(side, request, trigger, recordingMs = 0, audioPlayingAtStart = false) {
  if (audioPlaying) {
    toast("Wait for audio playback to finish before sending another turn.", true);
    return;
  }
  if (requestInFlight || busy) return;
  requestInFlight = true;
  busy = true;
  const turn = ++turnNo;
  let wav = null;
  setState(side, "busy");
  setStatus("Working", true);
  try {
    if (request.audioBlob) wav = await toWav16k(request.audioBlob);
    console.info(
      `[BolBridge] turn #${turn} trigger=${trigger} recording_ms=${Math.round(recordingMs)} ` +
      `wav_bytes=${wav ? wav.size : 0} speaker=${side} audio_playing_at_start=${audioPlayingAtStart}`
    );
    const fd = new FormData();
    fd.append("speaker", side);
    fd.append("src_lang", srcLang(side));
    fd.append("tgt_lang", hearLang(other[side]));
    fd.append("client_turn_id", `${getClientInstanceId()}:${turn}`);
    if (wav) fd.append("audio_file", wav, "turn.wav");
    else fd.append("text", request.text);

    const r = await fetch(request.audioBlob ? "/api/turn" : "/api/turn-text", { method: "POST", body: fd });
    let j;
    try { j = await r.json(); } catch (_) { throw new Error("Request failed."); }
    if (turn !== turnNo) return;
    if (!r.ok) throw new Error(j.detail || "Request failed");
    if (j.turn_id) console.info(`[BolBridge] turn #${turn} server turn_id=${j.turn_id} status=${j.status}`);
    if (j.status === "unclear") {
      addBubble(side, "warn", j.message);
      if (j.transcript) addBubble(side, "own", j.transcript, "You said");
      showTimings(j.timings);
      return;
    }
    addBubble(side, "own", j.transcript, j.detected_name || "You said");
    const voice = j.voice_used || j.voice;
    const caption = voice === "cloned" ? `Cloned voice (Speaker ${side})` : "Stock voice";
    addBubble(other[side], "heard", j.translation, caption, `heard as: ${j.transcript}`);
    showTimings(j.timings);
    if ($("#mute").checked) {
      toast("Playback muted for testing.");
    } else {
      setStatus("Speaking", true);
      await playB64(j.audio_b64);
    }
  } catch (e) {
    if (turn === turnNo) toast(e.message || String(e), true);
  } finally {
    if (turn === turnNo) {
      requestInFlight = false;
      busy = false;
      setState(side, "idle");
      setStatus("Ready");
    }
  }
}

async function runTurn(side, out, meta) {
  if (!out || out.ms < 1000) {
    setState(side, "idle");
    setStatus("Ready");
    return toast("Recording too short. Speak for at least 1 second.", true);
  }
  return submitTurn(side, { audioBlob: out.blob }, meta.trigger, out.ms, meta.audioPlayingAtStart);
}

function showTimings(t) {
  if (!t) return;
  const set = (id, v) => ($(id).textContent = v == null ? "-" : (v / 1000).toFixed(2) + " s");
  set("#t-stt", t.stt_ms); set("#t-llm", t.translate_ms); set("#t-tts", t.tts_ms); set("#t-total", t.total_ms);
  const segs = document.querySelectorAll("#latbar .seg");
  segs[0].style.flex = t.stt_ms || 1;
  segs[1].style.flex = t.translate_ms || 0.001;
  segs[2].style.flex = t.tts_ms || 0.001;
}

// Stop an in-progress recording without sending a turn (window blur, tab hide, pointer cancel).
function updateRecordingClock() {
  if (!active || !active.rec) return;
  const elapsed = Math.floor((performance.now() - active.rec.t0) / 1000);
  $("[data-rec-time]", panels[active.side]).textContent = `${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, "0")}`;
}

function clearRecordingTimers() {
  clearTimeout(holdTimer);
  clearInterval(recordingClock);
  holdTimer = null;
  recordingClock = null;
  for (const side of ["A", "B"]) $("[data-rec-time]", panels[side]).hidden = true;
}

function playbackBlocked() {
  if (audioPlaying) {
    toast("Wait for translated audio to finish before speaking.", true);
    return true;
  }
  const remaining = playbackCooldownUntil - performance.now();
  if (remaining > 0) {
    toast(`Wait ${Math.ceil(remaining / 100) / 10} s after playback before speaking.`, true);
    return true;
  }
  return false;
}

function cancelRecording() {
  if (!active) return;
  const side = active.side;
  holding[side] = false;
  clearRecordingTimers();
  const rec = active.rec;
  recordingToken++;
  active = null;
  if (rec) rec.stop().then(() => {}, () => {});
  setState(side, "idle");
  if (!busy) setStatus("Ready");
}

// ---------------------------------------------------------------- hold to talk
async function press(side, trigger = "unknown") {
  if (playbackBlocked()) return;
  if (busy || requestInFlight || turnPending) return;
  if (tapMode) {
    if (holding[side]) { release(side, trigger); return; }
    if (active) return;
  } else if (holding[side] || active) {
    return;
  }
  const token = ++recordingToken;
  holding[side] = true;
  active = { side, rec: null, token, trigger, audioPlayingAtStart: audioPlaying };
  setState(side, "listening");
  setStatus("Listening", true);
  try {
    const rec = await Recorder.start();
    if (document.hidden || !active || active.token !== token || !holding[side]) {
      await rec.stop();
      if (active && active.token === token) {
        active = null;
        holding[side] = false;
        setState(side, "idle");
        if (!busy) setStatus("Ready");
      }
      return;
    }
    active.rec = rec;
    $("[data-rec-time]", panels[side]).hidden = false;
    updateRecordingClock();
    recordingClock = setInterval(updateRecordingClock, 200);
    holdTimer = setTimeout(() => release(side, "max-duration"), MAX_TALK_MS);
  } catch (e) {
    if (active && active.token === token) {
      active = null;
      holding[side] = false;
      if (!busy) setState(side, "idle");
      if (!busy) setStatus("Ready");
      toast("Microphone blocked. Allow mic access in the browser address bar.", true);
    }
  }
}
async function release(side, trigger = "unknown") {
  if (!holding[side]) return;
  holding[side] = false;
  if (!active || active.side !== side) { setState(side, "idle"); if (!busy) setStatus("Ready"); return; }
  const current = active;
  clearRecordingTimers();
  active = null;
  if (!current.rec) {
    recordingToken++;
    setState(side, "idle");
    if (!busy) setStatus("Ready");
    return;
  }
  const { rec } = current;
  turnPending = true;
  setState(side, "busy");
  setStatus("Working", true);
  try {
    const out = await rec.stop();
    await runTurn(side, out, { trigger: current.trigger || trigger, audioPlayingAtStart: current.audioPlayingAtStart });
  } catch (e) {
    toast(e.message || String(e), true);
    setState(side, "idle");
    setStatus("Ready");
  } finally {
    turnPending = false;
  }
}

// ---------------------------------------------------------------- voice enrollment
const enrolling = { A: null, B: null };
const enrollTimers = { A: null, B: null };
async function toggleEnroll(side) {
  const btn = $("[data-enroll]", panels[side]);
  const playBtn = $("[data-play-sample]", panels[side]);
  if (enrolling[side]) {
    clearTimeout(enrollTimers[side]);
    enrollTimers[side] = null;
    const rec = enrolling[side];
    enrolling[side] = null;
    btn.classList.remove("recording");
    btn.textContent = "Checking sample...";
    btn.disabled = true;
    try {
      const out = await rec.stop();
      if (!out) throw new Error("No audio captured. Try again.");
      const mono = await decodeToMono(out.blob);
      const stats = analyzeEnrollment(mono.samples, mono.rate);
      if (stats.duration < 10) throw new Error("Recording too short. Speak clearly for at least 10 seconds.");
      if (stats.duration > 28) toast("Sample is long (>28s); shorter 10–25s samples usually work better.");
      if (stats.rms < 0.01) throw new Error("Recording is too quiet. Move closer to the mic and speak clearly.");
      if (stats.clippedRatio > 0.02 || stats.peak >= 0.999) throw new Error("Recording is heavily clipped. Speak a little softer or move farther from the mic.");
      if (stats.silenceRatio > 0.6) throw new Error("Recording is mostly silence. Speak continuously for 10–20 seconds.");

      const wav = encodeWav(mono.samples, mono.rate);
      if (lastSample[side] && lastSample[side].url) URL.revokeObjectURL(lastSample[side].url);
      lastSample[side] = { blob: wav, url: URL.createObjectURL(wav), ms: stats.duration * 1000 };
      if (playBtn) playBtn.disabled = false;

      btn.textContent = "Saving voice...";
      const fd = new FormData();
      fd.append("speaker", side);
      fd.append("consent", "true");
      fd.append("audio_file", wav, "voice.wav");
      const r = await fetch("/api/enroll", { method: "POST", body: fd });
      let j; try { j = await r.json(); } catch (_) { throw new Error("Could not clone voice."); }
      if (!r.ok) throw new Error(j.detail || "Could not clone voice");
      markCloned(side, true);
      toast(`Speaker ${side}'s voice is ready.`);
    } catch (e) {
      toast(e.message || String(e), true);
    } finally {
      btn.disabled = false;
      btn.textContent = "Record voice sample";
    }
    return;
  }
  if (!$("[data-consent]", panels[side]).checked) return toast("Tick the consent box first.", true);
  if (active || busy || turnPending || Object.values(enrolling).some(Boolean)) return toast("Please wait for the current recording to finish.", true);
  try {
    enrolling[side] = await Recorder.start();
    btn.classList.add("recording");
    btn.textContent = "Stop and save voice";
    enrollTimers[side] = setTimeout(() => { if (enrolling[side]) toggleEnroll(side); }, ENROLL_MAX_MS);
  } catch (e) {
    toast("Microphone blocked. Allow mic access in the browser address bar.", true);
  }
}
function markCloned(side, on) {
  clonedVoices[side] = !!on;
  const b = $("[data-badge]", panels[side]);
  b.textContent = on ? "Cloned voice" : "Stock voice";
  b.classList.toggle("cloned", on);
  $(".tag").textContent = Object.values(clonedVoices).some(Boolean)
    ? "Speak your language. They hear theirs, in your voice."
    : "Speak your language. They hear theirs.";
}

async function previewVoice(side) {
  const btn = $("[data-preview]", panels[side]);
  if (btn.disabled || busy || turnPending || active || Object.values(enrolling).some(Boolean)) return;
  btn.disabled = true;
  try {
    const fd = new FormData();
    fd.append("speaker", side);
    fd.append("tgt_lang", hearLang(side));
    const r = await fetch("/api/voice-preview", { method: "POST", body: fd });
    let j; try { j = await r.json(); } catch (_) { throw new Error("Preview failed."); }
    if (!r.ok) throw new Error(j.detail || "Preview failed");
    markCloned(side, !!j.embedding_present);
    await playB64(j.audio_b64);
  } catch (e) {
    toast(e.message || String(e), true);
  } finally {
    btn.disabled = false;
  }
}

async function removeVoice(side) {
  const btn = $("[data-remove-voice]", panels[side]);
  if (btn.disabled || busy || turnPending || active || enrolling[side]) return;
  btn.disabled = true;
  try {
    const fd = new FormData();
    fd.append("speaker", side);
    const r = await fetch("/api/remove-voice", { method: "POST", body: fd });
    let j; try { j = await r.json(); } catch (_) { throw new Error("Could not remove voice."); }
    if (!r.ok) throw new Error(j.detail || "Could not remove voice");
    markCloned(side, false);
    toast(`Speaker ${side}'s voice removed.`);
  } catch (e) {
    toast(e.message || String(e), true);
  } finally {
    btn.disabled = false;
  }
}

function playSample(side) {
  const s = lastSample[side];
  if (!s || !s.url) return toast("No sample recorded yet. Record a voice sample first.");
  if (active || busy || Object.values(enrolling).some(Boolean)) return toast("Stop the current recording before playback.", true);
  const a = new Audio(s.url);
  playAudio(a);
}

// ---------------------------------------------------------------- init
async function init() {
  let s;
  try {
    const r = await fetch("/api/state");
    s = await r.json();
  } catch (e) {
    toast("Could not load app state.", true);
    return;
  }
  const opts = (auto) =>
    (auto ? '<option value="auto">Auto-detect</option>' : "") +
    Object.entries(s.languages).map(([c, n]) => `<option value="${c}">${n}</option>`).join("");
  const defaults = { A: { src: "en-IN", hear: "te-IN" }, B: { src: "te-IN", hear: "en-IN" } };
  for (const side of ["A", "B"]) {
    const p = panels[side];
    const src = $("[data-src]", p), hear = $("[data-hear]", p);
    src.innerHTML = opts(true);
    hear.innerHTML = opts(false);
    src.value = ls(`bb_src_${side}`) || defaults[side].src;
    hear.value = ls(`bb_hear_${side}`) || defaults[side].hear;
    src.onchange = () => ls(`bb_src_${side}`, src.value);
    hear.onchange = () => ls(`bb_hear_${side}`, hear.value);
    const instructions = tapMode
      ? "Tap once to start speaking, then tap again to stop. The other side will hear it translated."
      : "Hold the button and speak. The other side will hear it translated.";
    $("[data-feed]", p).innerHTML = `<p class="empty">${instructions}</p>`;
    markCloned(side, s.enrolled && s.enrolled[side]);

    const hint = $("[data-src-hint]", p);
    const refreshHint = () => { if (hint) hint.style.display = src.value === "auto" ? "" : "none"; };
    src.onchange = () => { ls(`bb_src_${side}`, src.value); refreshHint(); };
    hear.onchange = () => ls(`bb_hear_${side}`, hear.value);
    refreshHint();

    const talk = $("[data-talk]", p);
    talk.addEventListener("pointerdown", (e) => {
      if (tapMode) return;
      e.preventDefault();
      if (talk.setPointerCapture) try { talk.setPointerCapture(e.pointerId); } catch (_) {}
      press(side, "pointer");
    });
    talk.addEventListener("pointerup", () => { if (!tapMode) release(side, "pointer"); });
    talk.addEventListener("pointercancel", cancelRecording);
    talk.addEventListener("pointerleave", (e) => {
      if (!tapMode && active && active.side === side && !talk.hasPointerCapture(e.pointerId)) cancelRecording();
    });
    talk.addEventListener("click", () => { if (tapMode) press(side, "click"); });
    talk.addEventListener("contextmenu", (e) => e.preventDefault());
    $("[data-text-turn]", p).addEventListener("submit", (e) => {
      e.preventDefault();
      const input = $("[data-text-input]", p);
      const text = input.value.trim();
      if (!text) return toast("Type a sentence before sending.", true);
      if (active || turnPending || Object.values(enrolling).some(Boolean)) return toast("Stop the current recording before sending text.", true);
      if (audioPlaying) return toast("Wait for audio playback to finish before sending text.", true);
      if (busy || requestInFlight) return toast("Wait for the current turn to finish before sending text.", true);
      const button = $("button", e.currentTarget);
      button.disabled = true;
      submitTurn(side, { text }, "click").finally(() => { button.disabled = false; });
      input.value = "";
    });
    $("[data-enroll]", p).onclick = () => toggleEnroll(side);
    $("[data-play-sample]", p).onclick = () => playSample(side);
    $("[data-preview]", p).onclick = () => previewVoice(side);
    $("[data-remove-voice]", p).onclick = () => removeVoice(side);
  }
  const tapEl = $("#tap");
  tapEl.checked = tapMode;
  tapEl.onchange = () => {
    if (active) cancelRecording();
    tapMode = tapEl.checked;
    ls("bb_tap", tapMode ? "1" : "0");
    const instructions = tapMode
      ? "Tap once to start speaking, then tap again to stop. The other side will hear it translated."
      : "Hold the button and speak. The other side will hear it translated.";
    for (const side of ["A", "B"]) {
      setState(side, panels[side].dataset.state || "idle");
      const empty = $(".empty", $("[data-feed]", panels[side]));
      if (empty) empty.textContent = instructions;
    }
  };
  addEventListener("blur", cancelRecording);
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") cancelRecording(); });
  const keyMap = { a: "A", l: "B" };
  addEventListener("keydown", (e) => {
    if (e.repeat || (e.target.tagName && /^(SELECT|INPUT|TEXTAREA)$/.test(e.target.tagName))) return;
    const side = keyMap[e.key.toLowerCase()];
    if (side) { e.preventDefault(); press(side, "key"); }
  });
  addEventListener("keyup", (e) => {
    if (tapMode) return;
    const side = keyMap[e.key.toLowerCase()];
    if (side) { e.preventDefault(); release(side, "key"); }
  });
  if (!s.gnani_key) toast("GNANI_API_KEY missing in .env", true);
  else if (!s.llm) toast("Translation LLM not set in .env", true);
}
init();
