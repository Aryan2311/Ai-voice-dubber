# Worker GPU Management

Single EC2 GPU worker: one heavy job at a time to avoid CUDA OOM and contention.

## Architecture

```
SQS Job
   │
   ▼
Worker Listener (thread)
   │
   ▼
Local Job Queue
   │
   ▼
Processor (thread) → GPU Scheduler
   │
   ├── GPU job? → gpu_session() → AI pipeline → empty_cache()
   └── CPU job?  → run without lock
   │
   ▼
Delete SQS message
```

## Modules

- **gpu/gpu_locks.py** — Global `GPU_LOCK` (threading.Lock), `MAX_GPU_JOBS = 1`.
- **gpu/gpu_manager.py** — `gpu_session()`: acquire lock, yield, clear CUDA cache, release lock.
- **gpu/vram_monitor.py** — `get_vram_usage()`, `get_vram_usage_mb()`, `log_vram()`, optional `has_enough_vram()`.
- **gpu/timeout.py** — Optional `job_timeout(seconds)` for Unix (signal.SIGALRM).
- **scheduler/job_scheduler.py** — Local `Queue`, `add_job(receipt_handle, job)`, `get_job()`.

## Job types and GPU

| Job type             | GPU lock | Notes                          |
|----------------------|----------|--------------------------------|
| TRANSCRIBE           | Yes      | Whisper                        |
| TRANSLATE_TRANSCRIPT | Yes      | NLLB                           |
| GENERATE_SUBTITLE    | No       | Format only (SRT/VTT from JSON)|
| TEXT_TO_SPEECH       | Yes      | StyleTTS2                      |
| DUB_MEDIA            | Yes      | Whisper, NLLB, StyleTTS2       |

## VRAM (example)

- Whisper large: ~10 GB  
- StyleTTS2: ~6 GB  
- Total worst case remains within 24 GB with sequential execution.

## Model loading and warmup

- By default, models are loaded **once** in `load_models_once()` at startup, **strictly in order**: Whisper → NLLB → StyleTTS2 → optional StyleTTS GPU warmup. No parallel model initialization in that sequence.
- Logs use `[startup] 1/4` … `4/4` with **before/after** RSS (Linux) and GPU memory, and **elapsed seconds** per phase.
- Before any Hugging Face use, the worker sets `HF_HUB_DOWNLOAD_MAX_WORKERS=1`, `HF_HUB_ENABLE_HF_TRANSFER=0`, and `TOKENIZERS_PARALLELISM=false` so hub/tokenizer work stays as sequential as the library allows (shard downloads may still show progress bars).
- Optional **warmup** (e.g. StyleTTS2 "hello") runs inside a `gpu_session()` as phase **4/4** after load to avoid first-inference delay.

### Reduce startup load (EC2 “hanging” during first boot)

Heavy startup = Whisper + NLLB + StyleTTS2 download/load + optional GPU warmup, all competing for CPU RAM, disk, and GPU. Tune with env vars (Docker `-e` or compose):

| Variable | Effect |
|----------|--------|
| `WORKER_LAZY_LOAD=1` | **Skip all** startup loads; each model loads on **first job** that needs it. Lightest boot; **first** transcript/TTS/dub is much slower. |
| `WORKER_SKIP_WARMUP=1` | Skip StyleTTS2 **GPU warmup** inference after load (saves one GPU spike at boot). |
| `MODEL_LOAD_STAGGER_SEC=15` | **Seconds** to sleep between each eager load step + `gc` / `empty_cache` (spread peak RAM/IO). Use `10`–`30` if the instance feels stuck. |
| `WHISPER_MODEL_SIZE` | Default `base`. Smaller = less VRAM/RAM at startup (e.g. `small`, `tiny` for lighter boxes — quality tradeoff). |

## Safe execution

- Every GPU job runs inside `with gpu_session(): ...`.
- On exit, `torch.cuda.empty_cache()` is called and VRAM is logged.
- Never run parallel GPU jobs unless VRAM is explicitly budgeted (e.g. multi-GPU later).

## Optional timeout

- For long-running or hung jobs, use `job_timeout(seconds)` (Unix only) or process-level limits.
- Not enabled by default to avoid signal side effects.

## Future

- Multi-GPU: assign GPU by job type (e.g. GPU0 ASR, GPU1 TTS).
- Ray Serve / Triton / K8s GPU scheduling for horizontal scaling.
