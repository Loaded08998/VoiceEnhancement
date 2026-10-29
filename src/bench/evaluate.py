import json
import os
import numpy as np
import soundfile as sf
import soxr

EVAL_SR = 16000
MAX_LAG_S = 0.1


def resample_to_eval_sr(data: np.ndarray, sr: int) -> np.ndarray:
    if sr == EVAL_SR:
        return data
    return soxr.resample(data, sr, EVAL_SR, quality='VHQ')


def align(
    ref: np.ndarray, deg: np.ndarray, sr: int = EVAL_SR, max_lag_s: float = MAX_LAG_S
) -> tuple[np.ndarray, np.ndarray]:
    """Align deg to ref using FFT cross-correlation. Returns (ref_trim, deg_trim)."""
    max_lag = int(max_lag_s * sr)
    n = max(len(ref), len(deg))
    n_fft = 1
    while n_fft < 2 * n:
        n_fft <<= 1
    R = np.fft.rfft(deg, n=n_fft) * np.conj(np.fft.rfft(ref, n=n_fft))
    cc = np.fft.irfft(R)
    search = np.concatenate([cc[: max_lag + 1], cc[n_fft - max_lag :]])
    best = np.argmax(np.abs(search))
    if best <= max_lag:
        lag = best
    else:
        lag = best - len(search)
    if lag > 0:
        deg = deg[lag:]
        ref = ref[: len(deg)]
    elif lag < 0:
        ref = ref[-lag:]
        deg = deg[: len(ref)]
    L = min(len(ref), len(deg))
    return ref[:L], deg[:L]


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant SDR (dB). Ref and est must be same length."""
    assert len(ref) == len(est), f'Length mismatch: {len(ref)} != {len(est)}'
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
    target = alpha * ref
    noise = est - target
    val = 10 * np.log10((np.dot(target, target) + 1e-8) / (np.dot(noise, noise) + 1e-8))
    return float(val)


def pesq_wb(ref: np.ndarray, deg: np.ndarray, sr: int = EVAL_SR) -> tuple[float, str]:
    """PESQ wide-band. Returns (score, reason)."""
    try:
        from pesq import PesqError, pesq

        assert sr in (8000, 16000), f'PESQ requires 8000 or 16000 Hz, got {sr}'
        if len(ref) < 0.25 * sr or len(deg) < 0.25 * sr:
            return float('nan'), 'too_short'
        if np.max(np.abs(deg)) < 1e-6:
            return float('nan'), 'silent'
        val = pesq(sr, ref, deg, 'wb', on_error=PesqError.RETURN_VALUES)
        return float(val), ''
    except Exception as e:
        return float('nan'), str(e)[:80]


def score_pair(ref_wav: str, noisy_wav: str, enhanced_wavs: dict, out_dir: str) -> dict:
    """Score all models for one pair. Returns nested dict."""

    def load_resample(path):
        data, sr = sf.read(path, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        return resample_to_eval_sr(data, sr)

    ref = load_resample(ref_wav)
    noisy = load_resample(noisy_wav)

    all_results = {}

    # Score noisy (baseline)
    ref_a, noisy_a = align(ref, noisy)
    noisy_sisdr = si_sdr(ref_a, noisy_a)
    noisy_pesq, noisy_pesq_reason = pesq_wb(ref_a, noisy_a)
    all_results['baseline'] = {
        'si_sdr': noisy_sisdr,
        'si_sdri': 0.0,
        'pesq_wb': noisy_pesq,
        'pesq_reason': noisy_pesq_reason,
        'delta_pesq': 0.0,
        'status': 'ok',
    }

    for model_name, enh_wav in enhanced_wavs.items():
        if enh_wav is None or not os.path.exists(enh_wav):
            all_results[model_name] = {
                'status': 'FAILED',
                'si_sdr': float('nan'),
                'pesq_wb': float('nan'),
            }
            continue
        try:
            enh = load_resample(enh_wav)
            ref_a, enh_a = align(ref, enh)
            model_sisdr = si_sdr(ref_a, enh_a)
            model_pesq, reason = pesq_wb(ref_a, enh_a)
            all_results[model_name] = {
                'si_sdr': model_sisdr,
                'si_sdri': model_sisdr - noisy_sisdr,
                'pesq_wb': model_pesq,
                'pesq_reason': reason,
                'delta_pesq': (
                    (model_pesq - noisy_pesq)
                    if (not np.isnan(model_pesq) and not np.isnan(noisy_pesq))
                    else float('nan')
                ),
                'status': 'ok',
            }
        except Exception as e:
            all_results[model_name] = {
                'status': f'ERROR:{e}',
                'si_sdr': float('nan'),
                'pesq_wb': float('nan'),
            }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results

