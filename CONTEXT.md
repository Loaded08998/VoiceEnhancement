# CONTEXT.md — speech-enh-bench

## RESUME PROMPT
Goal: 9-pair × 6-model speech enhancement benchmark (make-pair mode B, SNR=5dB). ALL PHASES COMPLETE.
Read this file fully; run `scripts/status.sh` and `git log --oneline -10`. All steps are DONE.
The final report is at `results/report.md`. Run `./run.sh --snr-db 5 --device auto` for a fresh re-run.

## NEXT ACTION
LAST COMMAND: git commit -m "Phase 6: README, final CONTEXT.md — all phases complete"
RESULT: PASSED
NEXT: PROJECT COMPLETE. Re-run with: ./run.sh --snr-db 5 --device auto

## Phase Checklist
- [x] Phase 0: Bootstrap (commit c57ec11)
- [x] Phase 1: Core audio library (commit 89956fd)
- [x] Phase 2: Evaluator + make-pair + tests (commit c8530fd) — 13/13 tests pass
- [x] Phase 3a: MetricGAN+ adapter (commit fdbe0c5) — smoke OK, 0.32s CUDA
- [x] Phase 3b: DeepFilterNet adapter (commit d8e1577) — smoke OK, 0.97s CUDA
- [x] Phase 3c: Demucs adapter (commit c1c8ac5) — smoke OK, 0.26s CUDA
- [x] Phase 3d: ConvTasNet adapter (commit 464e8bf) — smoke OK, 0.21s CUDA
- [x] Phase 3e: DCCRN adapter (commit 4e7b94a) — smoke OK, 0.29s CUDA
- [x] Phase 3f: CleanUNet adapter (commit 52f05e5) — smoke OK, 0.30s CUDA, SHA256 verified
- [x] Phase 4: Orchestrator + CLI (commit dfd1f87)
- [x] Phase 5: End-to-end run — 9 pairs × 6 models = 54 runs in 66.4s, all OK
- [x] Phase 6: README + final CONTEXT.md

## Model Status Table
| Model | Env | Weights | Adapter | Last Result | Known Issue |
|-------|-----|---------|---------|-------------|-------------|
| metricgan_plus | envs/speechbrain | speechbrain/metricgan-plus-voicebank | smoke-tested+validated | Mean SI-SDRi=-3.02, PESQ-WB=2.02 | Negative SI-SDRi expected (PESQ-optimised) |
| deepfilternet | envs/dfn | DeepFilterNet3 (auto-dl) | smoke-tested+validated | Mean SI-SDRi=+5.47, PESQ-WB=1.91 | 48kHz→16kHz downsampled for scoring |
| demucs | envs/denoiser | dns64 | smoke-tested+validated | Mean SI-SDRi=+6.73, PESQ-WB=1.61 | - |
| convtasnet | envs/asteroid | JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k ⚠SUBSTITUTE | smoke-tested+validated | Mean SI-SDRi=+6.51, PESQ-WB=1.77 | High raw peak (normalised) |
| dccrn | envs/asteroid | JorisCos/DCCRNet_Libri1Mix_enhsingle_16k ⚠SUBSTITUTE | smoke-tested+validated | Mean SI-SDRi=+6.63, PESQ-WB=1.77 | - |
| cleanunet | envs/cleanunet | DNS-large-full (LFS, SHA256 verified) | smoke-tested+validated | Mean SI-SDRi=+6.63, PESQ-WB=1.86 | - |

## SPEC DIGEST
- 6 models: MetricGAN+ (16kHz), DeepFilterNet3 (48kHz), Demucs dns64 (16kHz), ConvTasNet (16kHz), DCCRN (16kHz), CleanUNet (16kHz)
- 9 test pairs: 3 Clean (Speech1, Speech2, Speech3) × 3 Noise (market, noisy_classroom, traffic) at 5 dB SNR
- Evaluation: SI-SDR, SI-SDRi, PESQ-WB (ITU-T P.862.2), ΔPESQ — all at 16 kHz with FFT alignment
- Output: WAV float32, no MP3 export
- Report: results/report.md, results/all_metrics.json

## ENVIRONMENT FACTS
- OS: Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic
- CPU: AMD Ryzen 7 8700F, 16 cores, 30 GiB RAM
- GPU: NVIDIA RTX 5060, 8 GiB VRAM, Driver 595.99, CUDA 13.2, sm_120 (Blackwell)
- Python 3.11.18 at /usr/bin/python3.11
- ffmpeg: 8.0.1
- git-lfs: 3.7.1
- uv: 0.13.6 at /home/krish/.local/bin/uv
- torch: 2.11.0+cu128 in all 6 envs (resolved higher than pinned 2.8.0; compatible)
- torchaudio: 2.6.0 (speechbrain, dfn, denoiser envs)
- GPU self-test: PASSED (cuda available, sm_120 in arch list)
- CleanUNet SHA256: 145c101eb5bbfa3ba52fb2b4ec7e5b64a361c102f89291f75e1dd42601d95dc9 ✓

## DECISIONS & DEVIATIONS
- SNR: 5 dB (user confirmed)
- CleanUNet LFS fallback: HF mirror → BLOCKED (user confirmed; LFS succeeded from GitHub directly)
- MP3 export: disabled (WAV only)
- torch resolved to 2.11.0+cu128 (not 2.8.0); fully compatible with sm_120
- ConvTasNet/DCCRN: community substitute checkpoints (Libri1Mix) — flagged in all reports
- MetricGAN+ SI-SDRi is negative across all 9 pairs (by design: model maximises PESQ not SDR)
- ConvTasNet raw output peak ~5000–5500 (model scale); normalised by adapter before writing WAV

## WORKING COMMANDS
```bash
# Full benchmark run
./run.sh --snr-db 5 --device auto

# Unit tests
./run.sh selftest

# Single pair
./run.sh make-pair --speech inputs/Clean/Speech1.mp3 --noise inputs/Noise/traffic.mp3 --snr-db 5

# Subset of models
./run.sh --snr-db 5 --device auto --models demucs,cleanunet

# Check status
bash scripts/status.sh
```

## KNOWN ISSUES & FIXES
- ConvTasNet raw output scale ~5000: adapter normalises peak to ≤1.0, logs scale factor in notes field. Root cause: model not scale-normalised; expected for Libri1Mix checkpoint.
- MetricGAN+ negative SI-SDRi: not a bug. Model trained to maximise PESQ-WB, not SDR. PESQ-WB is consistently the highest of all models.
- DeepFilterNet scored at 16 kHz after downsampling 48→16 kHz: frequency content >8 kHz not captured in metrics. Noted in report caveats.

## OPEN QUESTIONS / BLOCKERS
None — all phases complete, all 54 runs succeeded.

## Footer
Last updated: 2026-10-29T15:38:00Z
Git commit: (final Phase 6 commit — see git log)
