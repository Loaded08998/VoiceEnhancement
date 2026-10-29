# CONTEXT.md — speech-enh-bench

## RESUME PROMPT
Goal: Build a 9-pair × 6-model speech enhancement benchmark pipeline (make-pair mode B, SNR=5dB).
Read this file fully; run `scripts/status.sh` and `git log --oneline -10`; do not redo steps marked DONE; re-verify only items marked IN PROGRESS; continue from NEXT ACTION; obey the Rules (no fabrication, no sudo, ask only for user-only inputs).

## NEXT ACTION
LAST COMMAND: HF_HOME=./models/hf ASTEROID_CACHE=./models/asteroid envs/asteroid/bin/python adapters/dccrn.py --in /tmp/smoke_16k.wav --out /tmp/out_dccrn.wav --device auto
RESULT: PASSED (SR=16000, 160000 samples, RMS=-55.0 dBFS, CUDA runtime=0.290s, SUBSTITUTE_WEIGHTS)
NEXT: Phase 3f: CleanUNet adapter (envs/cleanunet, clone CleanUNet, pull weights, adapters/cleanunet.py, smoke test)

## Phase Checklist
- [x] Phase 0: Bootstrap
- [x] Phase 1: Core audio library
- [x] Phase 2: Evaluator + make-pair + tests
- [x] Phase 3a: MetricGAN+ adapter
- [x] Phase 3b: DeepFilterNet adapter
- [x] Phase 3c: Demucs adapter
- [x] Phase 3d: ConvTasNet adapter
- [x] Phase 3e: DCCRN adapter
- [ ] Phase 3f: CleanUNet adapter
- [ ] Phase 4: Orchestrator + CLI
- [ ] Phase 5: End-to-end run
- [ ] Phase 6: README + final docs

## Model Status Table
| Model | Env | Weights | Adapter | Last Result | Known Issue |
|-------|-----|---------|---------|-------------|-------------|
| metricgan_plus | envs/speechbrain | HF:speechbrain/metricgan-plus-voicebank | adapters/metricgan_plus.py | PASSED (smoke 10s: 0.32s CUDA) | - |
| deepfilternet | envs/dfn | DeepFilterNet3 (auto) | adapters/deepfilternet.py | PASSED (smoke 10s: 0.97s CUDA) | - |
| demucs | envs/denoiser | dns64 | adapters/demucs.py | PASSED (smoke 10s: 0.26s CUDA) | - |
| convtasnet | envs/asteroid | JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k ⚠SUBSTITUTE | adapters/convtasnet.py | PASSED (smoke 10s: 0.21s CUDA) | Unbounded scale clipped & normalized |
| dccrn | envs/asteroid | JorisCos/DCCRNet_Libri1Mix_enhsingle_16k ⚠SUBSTITUTE | adapters/dccrn.py | PASSED (smoke 10s: 0.29s CUDA) | - |
| cleanunet | envs/cleanunet | DNS-large-full (LFS) | none | TODO | - |

## SPEC DIGEST
- 6 models: MetricGAN+ (16kHz), DeepFilterNet3 (48kHz), Demucs dns64 (16kHz), ConvTasNet (16kHz), DCCRN (16kHz), CleanUNet (16kHz)
- 9 test pairs: 3 Clean (Speech1, Speech2, Speech3) x 3 Noise (market, noisy_classroom, traffic)
- SNR: 5 dB mix
- Metrics: PESQ-WB (16kHz), SI-SDR, SI-SDRi, Delta PESQ
- Output: WAV only, float32

## ENVIRONMENT FACTS
- OS: Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic
- CPU: AMD Ryzen 7 8700F, 16 cores, 30 GiB RAM
- GPU: NVIDIA RTX 5060, 8 GiB VRAM, Driver 595.99, CUDA 13.2 (sm_120 = Blackwell)
- Python 3.11: /usr/bin/python3.11 (Python 3.11.18)
- ffmpeg: 8.0.1
- git-lfs: 3.7.1
- uv: 0.13.6 at /home/krish/.local/bin/uv
- torch/torchaudio: 2.11.0+cu128 in envs/eval, envs/speechbrain, envs/dfn, envs/denoiser, envs/asteroid
- GPU self-test: PASSED (torch.cuda.is_available()=True, t=torch.randn(1).cuda() OK, archs includes sm_120)

## DECISIONS & DEVIATIONS
- SNR: 5 dB (user confirmed)
- CleanUNet LFS fallback: HF mirror → BLOCKED (user confirmed)
- MP3 export: disabled (WAV only)
- Primary benchmark: 9 make-pair combos (3 Clean × 3 Noise)
- PyTorch version: 2.11.0+cu128 (cu128 index installs 2.11.0 with native sm_120 support)
- DeepFilterNet shim: added `torchaudio.backend.common.AudioMetaData` shim and passed CPU audio tensor to `enhance()`
- ConvTasNet scaling: unconstrained model output amplitude scaled down when peak > 1.0

## WORKING COMMANDS
- Status: `scripts/status.sh`
- Test all: `envs/eval/bin/python -m pytest tests/ -v`
- Smoke MetricGAN+: `HF_HOME=./models/hf envs/speechbrain/bin/python adapters/metricgan_plus.py --in /tmp/smoke_16k.wav --out /tmp/out_metricgan.wav --device auto`
- Smoke DeepFilterNet: `XDG_CACHE_HOME=./models/cache envs/dfn/bin/python adapters/deepfilternet.py --in /tmp/smoke_48k.wav --out /tmp/out_dfn.wav --device auto`
- Smoke Demucs: `TORCH_HOME=./models/torch envs/denoiser/bin/python adapters/demucs.py --in /tmp/smoke_16k.wav --out /tmp/out_demucs.wav --device auto`
- Smoke ConvTasNet: `HF_HOME=./models/hf ASTEROID_CACHE=./models/asteroid envs/asteroid/bin/python adapters/convtasnet.py --in /tmp/smoke_16k.wav --out /tmp/out_convtasnet.wav --device auto`
- Smoke DCCRN: `HF_HOME=./models/hf ASTEROID_CACHE=./models/asteroid envs/asteroid/bin/python adapters/dccrn.py --in /tmp/smoke_16k.wav --out /tmp/out_dccrn.wav --device auto`
- Full run: `./run.sh --snr-db 5 --device auto`

## KNOWN ISSUES & FIXES
- DeepFilterNet with torchaudio 2.11: torchaudio refactored backend module; shimmed `torchaudio.backend.common.AudioMetaData` in `adapters/deepfilternet.py`.

## OPEN QUESTIONS / BLOCKERS
(none — all answered)

## Footer
Last updated: 2026-10-29T13:42:00Z
Git commit: 4e7b94a
