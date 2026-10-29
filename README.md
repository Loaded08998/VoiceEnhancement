# speech-enh-bench

A reproducible CLI benchmark for six pretrained speech-enhancement models, with SI-SDR and PESQ-WB evaluation across controlled noisy speech pairs.

## Quickstart

```bash
# 1. Install system dependencies (once, requires sudo)
sudo apt install -y ffmpeg python3.11 python3.11-dev python3.11-venv

# 2. Set up all environments and download weights (idempotent)
./setup.sh

# 3. Run the full 9-pair benchmark at 5 dB SNR
./run.sh --snr-db 5 --device auto

# Results
cat results/report.md
```

## What it does

1. **make-pair**: Mixes `inputs/Clean/*.mp3` × `inputs/Noise/*.mp3` at the chosen SNR (default 5 dB) — 3 × 3 = 9 noisy inputs. Saves noisy .mp3 to `inputs/pairs/` and clean reference WAV to `references/`.
2. **Infer**: Runs each of the 6 models on every noisy pair (54 adapter runs). Each model uses its own isolated Python 3.11 venv and native sample rate.
3. **Score**: Aligns outputs to the reference (FFT cross-correlation, ±100 ms), then computes SI-SDR, SI-SDRi, PESQ-WB (ITU-T P.862.2) and ΔPesq for every output and the baseline.
4. **Report**: Writes `results/report.md` (master table + summaries), `results/all_metrics.json`, and per-pair `results/<stem>/metrics.json`.

## CLI reference

```
./run.sh [OPTIONS]

Options:
  --snr-db FLOAT       Mix SNR in dB (default: 5.0)
  --device auto|cuda|cpu  Inference device (default: auto)
  --models a,b,c       Run a subset of models
  --pairs a,b,c        Run a subset of pairs
  --no-resume          Force rerun even if outputs exist
  --config PATH        Path to models.yaml (default: config/models.yaml)
  --clean-dir PATH     Clean speech directory (default: inputs/Clean)
  --noise-dir PATH     Noise directory (default: inputs/Noise)

./run.sh selftest      Run unit tests (13 tests)
./run.sh make-pair --speech S.mp3 --noise N.mp3 --snr-db 5  # single pair
```

## Output structure

```
results/
  report.md              ← master benchmark report
  all_metrics.json       ← all scores in one JSON
  <pair-stem>/
    metrics.json         ← per-pair scores
    enhanced/
      metricgan_plus.wav
      deepfilternet.wav
      demucs.wav
      convtasnet.wav
      dccrn.wav
      cleanunet.wav
inputs/pairs/            ← generated noisy mp3s
references/              ← generated clean reference WAVs
logs/full_run.log        ← full run log
```

## The six models

| Model | Native SR | Weights | Notes |
|-------|-----------|---------|-------|
| MetricGAN+ | 16 kHz | `speechbrain/metricgan-plus-voicebank` (HF) | Trained to maximise PESQ; SI-SDR may be lower than baseline |
| DeepFilterNet3 | 48 kHz | Downloaded on first use | Scored at 16 kHz after downsampling |
| Demucs (denoiser dns64) | 16 kHz | `dl.fbaipublicfiles.com` | DNS Challenge trained |
| ConvTasNet | 16 kHz | `JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k` (HF) | ⚠ Substitute community checkpoint |
| DCCRN | 16 kHz | `JorisCos/DCCRNet_Libri1Mix_enhsingle_16k` (HF) | ⚠ Substitute community checkpoint |
| CleanUNet | 16 kHz | `third_party/CleanUNet` (Git LFS, SHA-256 verified) | DNS-large-full variant |

## Caveats

- **Substitute weights**: ConvTasNet and DCCRN use community Asteroid checkpoints trained on LibriMix (not the original paper's DNS-Challenge weights). Results are indicative only.
- **MetricGAN+** is explicitly optimised for PESQ score — its high PESQ-WB combined with negative SI-SDRi is expected, not a bug.
- **DeepFilterNet** operates at 48 kHz natively; its output is downsampled to 16 kHz before scoring, so frequency content above 8 kHz is not captured in the metrics.
- **Single SNR**: All pairs mixed at 5 dB. Performance varies substantially at other levels.
- **Training data mismatch**: Models were trained on different corpora (VoiceBank-DEMAND, DNS Challenge, LibriMix). This is not a controlled comparison.
- **CleanUNet** does not remove reverberation — it targets additive noise only.

## How to add a model

1. Add an entry to `config/models.yaml` (env, native_sr, weights, chunk_s, overlap_s).
2. Create `envs/<name>` with `uv venv --python 3.11 envs/<name>` and install dependencies.
3. Write `adapters/<name>.py` following the contract in `adapters/common.py`:
   - Accept `--in`, `--out`, `--device` args
   - Assert input SR == native_sr
   - Write output with **exactly** the same sample count
   - Print one JSON line: `{model, weights, sr, device, runtime_s, n_in, n_out, peak, notes}`
   - Exit non-zero on failure
4. Smoke test: `envs/<name>/bin/python adapters/<name>.py --in /tmp/smoke_16k.wav --out /tmp/out.wav --device auto`
5. Add env to `setup.sh`.

## Reproducibility

- All model envs use CPython 3.11 (system Python 3.14 is too new for several compiled wheels).
- PyTorch `2.11.0+cu128` (CUDA 12.8) is required for RTX 50-series (sm_120, Blackwell).
- `UV_CACHE_DIR=./.uv-cache` keeps all wheels inside the project directory (hardlinked, not copied).
- `HF_HUB_OFFLINE=1` in `run.sh` after first setup prevents unintended weight re-downloads.
- CleanUNet checkpoint SHA-256 is verified at every adapter run.
- CPU fallback is automatic on CUDA OOM or unavailable GPU.

## Environment

Tested on: Ubuntu 26.04.1 LTS, AMD Ryzen 7 8700F, NVIDIA RTX 5060 (8 GiB), CUDA Driver 595.99.

