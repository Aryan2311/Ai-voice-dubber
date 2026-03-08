# Dubbing pipeline – what we do and why it was misaligned

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
