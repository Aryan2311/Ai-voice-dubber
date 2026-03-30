# Worker GPU Management

Single EC2 GPU worker: one heavy job at a time to avoid CUDA OOM and contention.
Experimental translation/TTS overlap, when enabled, happens only inside a single dub job.

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
| TRANSLATE_TRANSCRIPT | Yes      | Mistral 7B translation         |
| GENERATE_SUBTITLE    | Yes      | May trigger translation inline |
| TEXT_TO_SPEECH       | Yes      | XTTS                           |
| DUB_MEDIA            | Yes      | Whisper, Mistral, XTTS overlap |

## VRAM (example)

- Whisper large: ~10 GB  
- Mistral 7B 4-bit: several GB, workload-dependent  
- XTTS: ~6 GB  
- Overlap requires enough free VRAM after both XTTS and Mistral are resident.
- The worker checks headroom explicitly and fails the overlap path if free VRAM is too low.

## Model loading and warmup

- Models are loaded **once** in `load_models_once()` at startup.
- XTTS and Mistral are preloaded together so the dub job can overlap batch translation and TTS.
- Optional **warmup** (e.g. XTTS "hello") runs inside a `gpu_session()` after load to avoid first-inference delay.

## Safe execution

- Every GPU job runs inside `with gpu_session(): ...`.
- On exit, `torch.cuda.empty_cache()` is called and VRAM is logged.
- Never run parallel GPU jobs unless VRAM is explicitly budgeted (e.g. multi-GPU later).
- DUB overlap uses internal producer/consumer threads inside one GPU job; it does not allow two SQS jobs to run at once.

## Optional timeout

- For long-running or hung jobs, use `job_timeout(seconds)` (Unix only) or process-level limits.
- Not enabled by default to avoid signal side effects.

## Future

- Multi-GPU: assign GPU by job type (e.g. GPU0 ASR, GPU1 TTS).
- Ray Serve / Triton / K8s GPU scheduling for horizontal scaling.
