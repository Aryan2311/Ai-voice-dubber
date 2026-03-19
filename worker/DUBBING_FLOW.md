# Dubbing pipeline – what we do and why it was misaligned

## Worker layout (new)

- **pipeline/** – ASR → translate → rewrite (syllable-aware) → TTS → align → RVC → merge.
- **models/** – Whisper, NLLB, Phi-3 (rewrite), StyleTTS2 (TTS), RVC (voice conversion).
- **prosody/** – Extract pitch/energy and pauses; segment audio used as **prosody** reference only.
- **translation/** – `syllable_counter.py` (count syllables for en/hi); rewrite uses **target_syllables** (±20%) for alignment.
- **scheduler/** – Job queue and `gpu_lock` for sequential GPU work.

**Dual reference:** StyleTTS2 gets **prosody** from the original segment audio only; **voice identity** is applied by RVC after TTS. So emotion/rhythm come from the original, voice from the trained RVC model.

**Syllable-aware:** After NLLB translate, we count syllables in the original segment and pass `target_syllables` to the LLM rewrite so the dubbed line stays rhythm-compatible (±20% tolerance).

**Reference audio preprocessing** (`worker/utils/reference_audio.py`): All voice references are preprocessed before use: trim to 3–8 s, loudness normalize, resample to 22050 Hz mono. Prosody segment refs are normalized and resampled the same way. Optional: set `isolate_voice=True` and install Demucs for noise removal.

**Env (optional):** `STYLETTS2_CHECKPOINT`, `STYLETTS2_CONFIG`; `RVC_MODEL_PATH`; `REWRITE_LLM_MODEL`; `WHISPER_MODEL_SIZE`. Optional: `demucs` for voice isolation on reference audio.

Backend APIs and S3 keys are unchanged.

## Current flow (high level)

1. **Transcript** – Whisper transcribes the source audio → segments with `start`, `end`, `text` (and optional `full_text`).
2. **Translate** – Each segment’s `text` is translated to the target language; `start`/`end` are kept.
3. **TTS** – For each segment we call XTTS to generate speech for the translated text. Each segment becomes one WAV file (or several if we split long text for the 400-token limit).
4. **Merge** – We merge the new audio with the video (or output audio-only).

## Why it wasn’t fitting timestamps (before the fix)

- We **concatenated** all TTS WAVs in segment order: `clip1 + clip2 + clip3 + ...` with no regard to time.
- So the **first** TTS clip started at 0:00, the second right after the first clip ends, etc. Original segment times (e.g. segment 2 at 0:45–1:02) were **ignored**.
- TTS speaks at its own speed, so:
  - If TTS is slower than the original, the dubbed track gets longer and drifts.
  - If TTS is faster, we get gaps.
- Result: dubbed speech does not line up with the original timestamps or with subtitles, and can sound like “gibberish in between” because phrases appear at the wrong times.

## Why most of the time is in voice generation (TTS)

- **Transcript / subtitles**: Whisper runs once on the whole file; translation is one pass over segments (fast CPU).
- **Dubbing**: For **each segment** we call XTTS (GPU) to generate speech. With hundreds of segments, that’s hundreds of TTS calls. The Coqui library logs “Text splitted to sentences” and “Processing time” per call; that **Processing time** (e.g. ~2–3 s) is the actual **voice synthesis** (GPU inference), not a separate “sentence splitting” step. We suppress those verbose logs and instead log `TTS segment i/N (generating voice)` so it’s clear the time is in generating voice.

## Fix: timestamp-aligned dubbing

- We use each segment’s **start** and **end** from the transcript.
- For each segment we:
  - Generate TTS for that segment’s text.
  - **Time-stretch** the TTS clip so its duration equals `end - start`.
  - **Place** that clip in the final track at time `start` (not just “after the previous clip”).
- The final dubbed track has the same length as the source; each phrase plays at the correct time and for the correct duration.
