# Dubbing pipeline (current)

## Worker layout

- **pipeline/** – ASR → translate → rewrite (syllable-aware) → TTS → align → merge
- **models/** – Whisper, NLLB, Phi-3 (rewrite), StyleTTS2
- **prosody/** – Extract pitch/energy/pauses from original segment audio
- **translation/** – Syllable counting utilities for rewrite constraints
- **scheduler/** – Sequential GPU lock and local queue

## Core behavior

- Prosody comes from original segment audio (`style_audio_path`).
- Optional voice reference can guide StyleTTS2 when no prosody segment is provided.
- Timestamp alignment is preserved by per-segment time-stretch and timeline placement.
- Backend API contract and S3 keys remain unchanged.

## Reference preprocessing

`worker/utils/reference_audio.py` preprocesses references before TTS use:

- optional Demucs isolation
- normalization
- resample to 22050 Hz mono
- trim to 3–8 seconds

## Env

- `STYLETTS2_CHECKPOINT`, `STYLETTS2_CONFIG`
- `REWRITE_LLM_MODEL`
- `WHISPER_MODEL_SIZE`
- Optional: Demucs installed for voice isolation
