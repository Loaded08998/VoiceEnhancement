import os
import numpy as np
import pytest
import soundfile as sf
import soxr
from src.bench.audio_io import OverlapAddChunker
from src.bench.evaluate import EVAL_SR, align, pesq_wb, score_pair, si_sdr


def _get_speech_16k():
    data, sr = sf.read('inputs/Clean/Speech1.mp3', dtype='float32')
    data_16k = soxr.resample(data, sr, EVAL_SR, quality='VHQ')
    # Use 3 seconds of real speech
    return data_16k[: EVAL_SR * 3]


def test_pesq_identical():
    # 1. PESQ(x, x, "wb") -> ≈ 4.64
    x = _get_speech_16k()
    score, reason = pesq_wb(x, x, sr=16000)
    assert reason == ''
    assert abs(score - 4.64) < 0.05, f'Expected ~4.64, got {score}'


def test_sisdr_at_5db():
    # 2. SI-SDR(x + noise @ 5 dB) -> ≈ 5.0 ±0.2 dB
    np.random.seed(42)
    ref = _get_speech_16k()
    ref = ref - np.mean(ref)

    # Generate noise orthogonal to reference
    noise = np.random.randn(len(ref)).astype(np.float32)
    noise = noise - (np.dot(noise, ref) / np.dot(ref, ref)) * ref
    target_noise_std = np.std(ref) / (10 ** (5.0 / 20.0))
    noise = noise * (target_noise_std / np.std(noise))

    noisy = ref + noise
    val = si_sdr(ref, noisy)
    assert abs(val - 5.0) < 0.2, f'Expected ~5.0 dB, got {val}'


def test_scale_invariance():
    # 3. Scale invariance: SI-SDR(ref, 0.1 * est) == SI-SDR(ref, est)
    np.random.seed(42)
    ref = _get_speech_16k()
    noise = np.random.randn(len(ref)).astype(np.float32) * 0.05
    est = ref + noise

    score1 = si_sdr(ref, est)
    score2 = si_sdr(ref, 0.1 * est)
    assert (
        abs(score1 - score2) < 1e-4
    ), f'Scale invariance failed: {score1} vs {score2}'


def test_align_20ms_lag():
    # 4. 20 ms lag recovery by align()
    sr = 16000
    lag_samples = int(0.020 * sr)  # 20 ms = 320 samples
    ref = _get_speech_16k()
    # Deg is delayed by 20 ms (padded with zeros at beginning)
    deg = np.pad(ref[:-lag_samples], (lag_samples, 0))

    ref_a, deg_a = align(ref, deg, sr=sr, max_lag_s=0.1)
    # After alignment, the overlapping parts should match
    max_diff = np.max(np.abs(ref_a - deg_a))
    assert max_diff < 1e-4, f'Max diff after 20ms lag recovery: {max_diff}'


def test_pesq_silence_nan():
    # 5. Silence -> NaN with reason, no crash
    ref = _get_speech_16k()
    silent = np.zeros(len(ref), dtype=np.float32)
    score, reason = pesq_wb(ref, silent, sr=16000)
    assert np.isnan(score)
    assert reason == 'silent'


def test_pesq_48k_error():
    # 6. 48 kHz input -> NaN/error until resampled (pesq_wb asserts 8000/16000)
    ref = np.random.randn(48000 * 2).astype(np.float32)
    score, reason = pesq_wb(ref, ref, sr=48000)
    assert np.isnan(score)
    assert 'PESQ requires 8000 or 16000 Hz' in reason


def test_chunker_identity_noop():
    # 7. OverlapAddChunker identity (no-op model returns same samples)
    sr = 16000
    ref = _get_speech_16k()
    chunker = OverlapAddChunker(ref, sr, chunk_s=1.0, overlap_s=0.1)
    chunks_out = [(chunk.copy(), start) for chunk, start in chunker.chunks()]
    rec = OverlapAddChunker.reconstruct(
        chunks_out, len(ref), sr, chunk_s=1.0, overlap_s=0.1
    )
    max_err = np.max(np.abs(rec - ref))
    assert max_err < 1e-4, f'Identity reconstruction error too high: {max_err}'


def test_make_pair_and_score_pair(tmp_path):
    # End-to-end integration test of make_pair, evaluate, and score_pair
    from src.bench.make_pair import make_pair
    clean = 'inputs/Clean/Speech1.mp3'
    noise = 'inputs/Noise/traffic.mp3'
    out_mp3 = str(tmp_path / 'pair.mp3')
    out_ref = str(tmp_path / 'pair_ref.wav')

    info = make_pair(clean, noise, 5.0, out_mp3, out_ref)
    assert os.path.exists(out_mp3)
    assert os.path.exists(out_ref)
    assert info['snr_db'] == 5.0

    # Fake enhanced: same as ref
    enh_wav = str(tmp_path / 'fake_enh.wav')
    data, sr = sf.read(out_ref)
    sf.write(enh_wav, data, sr, subtype='FLOAT')

    scores = score_pair(out_ref, out_mp3, {'fake_model': enh_wav}, str(tmp_path / 'eval_out'))
    assert 'baseline' in scores
    assert 'fake_model' in scores
    assert scores['fake_model']['status'] == 'ok'
    assert scores['fake_model']['si_sdri'] > 0
