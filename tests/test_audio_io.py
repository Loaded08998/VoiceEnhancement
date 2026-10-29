import os
import tempfile
import numpy as np
import pytest
import soundfile as sf
from src.bench.audio_io import OverlapAddChunker, decode_to_wav, resample_wav, sha256_file


def test_sha256_file(tmp_path):
    test_file = tmp_path / "test.bin"
    content = b"hello audio benchmark"
    test_file.write_bytes(content)
    import hashlib

    expected = hashlib.sha256(content).hexdigest()
    assert sha256_file(str(test_file)) == expected


def test_overlap_add_chunker_identity():
    sr = 16000
    duration_s = 5.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    # Synthetic signal: mixture of sines
    data = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)).astype(
        np.float32
    )

    chunk_s = 1.0
    overlap_s = 0.1
    chunker = OverlapAddChunker(data, sr, chunk_s=chunk_s, overlap_s=overlap_s)

    chunks_out = []
    for chunk, start in chunker.chunks():
        # Identity model
        chunks_out.append((chunk.copy(), start))

    reconstructed = OverlapAddChunker.reconstruct(
        chunks_out, len(data), sr, chunk_s=chunk_s, overlap_s=overlap_s
    )

    assert len(reconstructed) == len(data)
    max_err = np.max(np.abs(reconstructed - data))
    assert max_err < 1e-4, f"Reconstruction error too high: {max_err}"


def test_resample_wav_math(tmp_path):
    src_sr = 48000
    dst_sr = 16000
    duration_s = 2.0
    n_samples = int(src_sr * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)
    sig = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    src_wav = str(tmp_path / "src.wav")
    dst_wav = str(tmp_path / "dst.wav")
    sf.write(src_wav, sig, src_sr, subtype="FLOAT")

    out_samples = resample_wav(src_wav, dst_wav, dst_sr)
    assert out_samples == int(dst_sr * duration_s)

    data_res, sr_res = sf.read(dst_wav, dtype="float32")
    assert sr_res == dst_sr
    assert len(data_res) == out_samples


def test_resample_wav_identity(tmp_path):
    sr = 16000
    sig = np.random.randn(sr).astype(np.float32)
    src_wav = str(tmp_path / "src_16k.wav")
    dst_wav = str(tmp_path / "dst_16k.wav")
    sf.write(src_wav, sig, sr, subtype="FLOAT")

    out_samples = resample_wav(src_wav, dst_wav, sr)
    assert out_samples == len(sig)
    data_out, sr_out = sf.read(dst_wav, dtype="float32")
    assert sr_out == sr
    np.testing.assert_array_equal(data_out, sig)


def test_decode_to_wav(tmp_path):
    input_mp3 = "inputs/Clean/Speech1.mp3"
    assert os.path.exists(input_mp3)

    out_wav = str(tmp_path / "speech1.wav")
    sr, n_samples = decode_to_wav(input_mp3, out_wav, log_sha=True)

    assert sr == 48000
    assert n_samples > 0
    assert os.path.exists(out_wav)

    data, r_sr = sf.read(out_wav, dtype="float32")
    assert r_sr == sr
    assert len(data) == n_samples
    assert data.ndim == 1
    assert np.all(np.isfinite(data))

