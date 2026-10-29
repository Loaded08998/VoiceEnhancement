import hashlib
import json
import os
import subprocess
import sys
import numpy as np
import soundfile as sf
import soxr

SAMPLE_WIDTH = 4  # float32


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def decode_to_wav(src_path: str, out_wav: str, log_sha: bool = True) -> tuple[int, int]:
    """Decode any audio to mono float32 WAV at native sample rate."""
    # Probe native sample rate
    probe = subprocess.run(
        [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'a:0',
            '-show_entries',
            'stream=sample_rate',
            '-of',
            'csv=p=0',
            src_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    native_sr = int(probe.stdout.strip())
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    # Decode to mono float32 WAV
    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-i',
            src_path,
            '-ac',
            '1',
            '-ar',
            str(native_sr),
            '-sample_fmt',
            's16',
            out_wav,
        ],
        check=True,
        capture_output=True,
    )
    # Read back and convert to float32 WAV (soundfile)
    data, sr = sf.read(out_wav, dtype='float32')
    sf.write(out_wav, data, sr, subtype='FLOAT')
    n_samples = len(data)
    if log_sha:
        src_sha = sha256_file(src_path)
        out_sha = sha256_file(out_wav)
        print(
            json.dumps({
                'src': src_path,
                'src_sha256': src_sha,
                'out': out_wav,
                'out_sha256': out_sha,
                'sr': sr,
                'n_samples': n_samples,
            })
        )
    return sr, n_samples


def resample_wav(src_wav: str, dst_wav: str, target_sr: int, quality: str = 'VHQ') -> int:
    """Resample a mono float32 WAV to target_sr using soxr. Single-step, never chained."""
    data, sr = sf.read(src_wav, dtype='float32')
    os.makedirs(os.path.dirname(os.path.abspath(dst_wav)), exist_ok=True)
    if sr == target_sr:
        import shutil

        shutil.copy2(src_wav, dst_wav)
        return len(data)
    resampled = soxr.resample(data, sr, target_sr, quality=quality)
    sf.write(dst_wav, resampled, target_sr, subtype='FLOAT')
    return len(resampled)


class OverlapAddChunker:
    """Yield overlapping chunks and reconstruct with Hann crossfade."""

    def __init__(self, data: np.ndarray, sr: int, chunk_s: float, overlap_s: float):
        self.data = data
        self.sr = sr
        self.chunk_samples = int(chunk_s * sr)
        self.overlap_samples = int(overlap_s * sr)
        self.hop = self.chunk_samples - self.overlap_samples

    def chunks(self):
        n = len(self.data)
        start = 0
        while start < n:
            end = min(start + self.chunk_samples, n)
            yield self.data[start:end], start
            if end == n:
                break
            start += self.hop

    @staticmethod
    def reconstruct(
        chunks_and_starts, total_len: int, sr: int, chunk_s: float, overlap_s: float
    ) -> np.ndarray:
        chunk_samples = int(chunk_s * sr)
        overlap_samples = int(overlap_s * sr)
        out = np.zeros(total_len, dtype=np.float32)
        weight = np.zeros(total_len, dtype=np.float32)
        win = np.hanning(2 * overlap_samples).astype(np.float32) if overlap_samples > 0 else None
        fade_in = win[:overlap_samples] if win is not None else None
        fade_out = win[overlap_samples:] if win is not None else None
        for i, (chunk, start) in enumerate(chunks_and_starts):
            end = start + len(chunk)
            w = np.ones(len(chunk), dtype=np.float32)
            if i > 0 and overlap_samples > 0:
                fade_len = min(overlap_samples, len(chunk))
                w[:fade_len] = fade_in[:fade_len]
            if end < total_len and overlap_samples > 0:
                fade_len = min(overlap_samples, len(chunk))
                w[-fade_len:] = fade_out[-fade_len:]
            out[start:end] += chunk * w
            weight[start:end] += w
        np.divide(out, np.maximum(weight, 1e-8), out=out)
        return out

