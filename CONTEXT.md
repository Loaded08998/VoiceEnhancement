# CONTEXT.md — speech-enh-bench

## RESUME PROMPT
Goal: Build a 9-pair × 6-model speech enhancement benchmark pipeline (make-pair mode B, SNR=5dB).
Read this file fully; run `scripts/status.sh` and `git log --oneline -10`; do not redo steps marked DONE; re-verify only items marked IN PROGRESS; continue from NEXT ACTION; obey the Rules (no fabrication, no sudo, ask only for user-only inputs).

## NEXT ACTION
LAST COMMAND: envs/eval/bin/python -m pytest tests/ -v
RESULT: 13 passed in 2.20s
NEXT: Phase 3a: MetricGAN+ adapter (envs/speechbrain, adapters/metricgan_plus.py, smoke test)

## Phase Checklist
- [x] Phase 0: Bootstrap
- [x] Phase 1: Core audio library
- [x] Phase 2: Evaluator + make-pair + tests
- [ ] Phase 3a: MetricGAN+ adapter
- [ ] Phase 3b: DeepFilterNet adapter
- [ ] Phase 3c: Demucs adapter
- [ ] Phase 3d: ConvTasNet adapter
- [ ] Phase 3e: DCCRN adapter
- [ ] Phase 3f: CleanUNet adapter
- [ ] Phase 4: Orchestrator + CLI
- [ ] Phase 5: End-to-end run
- [ ] Phase 6: README + final docs

## Model Status Table
| Model | Env | Weights | Adapter | Last Result | Known Issue |
|-------|-----|---------|---------|-------------|-------------|
| metricgan_plus | envs/speechbrain | HF:speechbrain/metricgan-plus-voicebank | none | TODO | - |
| deepfilternet | envs/dfn | DeepFilterNet3 (auto) | none | TODO | - |
| demucs | envs/denoiser | dns64 | none | TODO | - |
| convtasnet | envs/asteroid | JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k ⚠SUBSTITUTE | none | TODO | - |
| dccrn | envs/asteroid | JorisCos/DCCRNet_Libri1Mix_enhsingle_16k ⚠SUBSTITUTE | none | TODO | - |
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
- torch/torchaudio: 2.11.0+cu128 in envs/eval
- GPU self-test: PASSED (torch.cuda.is_available()=True, t=torch.randn(1).cuda() OK, archs includes sm_120)

## DECISIONS & DEVIATIONS
- SNR: 5 dB (user confirmed)
- CleanUNet LFS fallback: HF mirror → BLOCKED (user confirmed)
- MP3 export: disabled (WAV only)
- Primary benchmark: 9 make-pair combos (3 Clean × 3 Noise)
- PyTorch version: 2.11.0+cu128 (cu128 index installs 2.11.0 with native sm_120 support)

## WORKING COMMANDS
- Status: `scripts/status.sh`
- Test all: `envs/eval/bin/python -m pytest tests/ -v`
- Smoke test 10s: `ffmpeg -y -i inputs/Clean/Speech1.mp3 -t 10 -ac 1 -ar 16000 /tmp/smoke_16k.wav`
- Full run: `./run.sh --snr-db 5 --device auto`

## KNOWN ISSUES & FIXES
(none yet)

## OPEN QUESTIONS / BLOCKERS
(none — all answered)

## Footer
Last updated: 2026-10-29T13:21:00Z
Git commit: c8530fd
