import glob
import hashlib
import json
import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 16000
TARGET_PEAK = 0.95


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-10))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def make_pair(
    clean_mp3: str, noise_mp3: str, snr_db: float, out_mp3: str, out_ref_wav: str
) -> dict:
    """Mix clean+noise at snr_db. Returns info dict."""

    def decode(path):
        fd, tmp = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(TARGET_SR), tmp],
                capture_output=True,
                check=True,
            )
            data, sr = sf.read(tmp, dtype='float32')
            return data
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    clean = decode(clean_mp3)
    noise = decode(noise_mp3)

    # Tile noise to match clean length
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[: len(clean)]

    # Scale noise to target SNR
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / noise_rms)

    # Mix
    mixture = clean + noise

    # Normalize mixture to a standard peak level while preserving SNR. Clean
    # speech sources in this benchmark are quiet (~-40 dBFS RMS), which puts
    # level-sensitive models (e.g. MetricGAN+) far outside their trained range.
    peak = float(np.max(np.abs(mixture)))
    scale = TARGET_PEAK / peak if peak > 0 else 1.0
    mixture = mixture * scale
    clean_ref = clean * scale

    # Save reference WAV
    os.makedirs(os.path.dirname(os.path.abspath(out_ref_wav)), exist_ok=True)
    sf.write(out_ref_wav, clean_ref, TARGET_SR, subtype='FLOAT')

    # Encode noisy mix as mp3 via ffmpeg LAME ~192 kbps
    os.makedirs(os.path.dirname(os.path.abspath(out_mp3)), exist_ok=True)
    fd, tmp_wav = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        sf.write(tmp_wav, mixture, TARGET_SR, subtype='FLOAT')
        subprocess.run(
            ['ffmpeg', '-y', '-i', tmp_wav, '-codec:a', 'libmp3lame', '-b:a', '192k', out_mp3],
            capture_output=True,
            check=True,
        )
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    return {
        'clean': clean_mp3,
        'noise': noise_mp3,
        'snr_db': snr_db,
        'out_mp3': out_mp3,
        'out_ref_wav': out_ref_wav,
        'n_samples': len(clean_ref),
        'ref_sha256': sha256_file(out_ref_wav),
        'mp3_sha256': sha256_file(out_mp3),
    }


def make_all_pairs(
    clean_dir: str, noise_dir: str, snr_db: float, pairs_dir: str, refs_dir: str
) -> list[dict]:
    cleans = sorted(glob.glob(os.path.join(clean_dir, '*.mp3')))
    noises = sorted(glob.glob(os.path.join(noise_dir, '*.mp3')))
    results = []
    for c in cleans:
        for n in noises:
            cstem = os.path.splitext(os.path.basename(c))[0]
            nstem = os.path.splitext(os.path.basename(n))[0]
            stem = f"{cstem}+{nstem}"
            out_mp3 = os.path.join(pairs_dir, f"{stem}.mp3")
            out_ref = os.path.join(refs_dir, f"{stem}_ref.wav")
            if os.path.exists(out_mp3) and os.path.exists(out_ref):
                print(f"[make-pair] skipping {stem} (exists)")
                results.append(
                    {'stem': stem, 'out_mp3': out_mp3, 'out_ref_wav': out_ref, 'skipped': True}
                )
                continue
            print(f"[make-pair] creating {stem} @ {snr_db} dB SNR")
            info = make_pair(c, n, snr_db, out_mp3, out_ref)
            info['stem'] = stem
            results.append(info)
    return results

