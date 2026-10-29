import argparse
import json
import os
import sys
import time
import numpy as np
import soundfile as sf


def parse_adapter_args(model_name: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f'Adapter for {model_name}')
    p.add_argument('--in', dest='input', required=True, help='Path to input WAV')
    p.add_argument('--out', dest='output', required=True, help='Path to output WAV')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    return p.parse_args()


def select_device(preference: str) -> str:
    try:
        import torch

        if preference in ('cuda', 'auto'):
            if torch.cuda.is_available():
                return 'cuda'
        return 'cpu'
    except Exception:
        return 'cpu'


def print_result(model, weights, sr, device, runtime_s, n_in, n_out, peak, notes=''):
    print(
        json.dumps({
            'model': model,
            'weights': weights,
            'sr': sr,
            'device': device,
            'runtime_s': round(runtime_s, 3),
            'n_in': n_in,
            'n_out': n_out,
            'peak': round(float(peak), 6),
            'notes': notes,
        }),
        flush=True,
    )


def load_wav(path: str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype='float32')
    assert data.ndim == 1, f"Expected mono, got shape {data.shape}"
    return data, sr


def save_wav(path: str, data: np.ndarray, sr: int, model_name: str):
    peak = np.max(np.abs(data)) if len(data) > 0 else 0.0
    notes = ''
    if peak > 1.0:
        scale = 1.0 / peak
        data = data * scale
        notes = f'scaled_by_{scale:.6f}'
        print(f'[{model_name}] WARNING: clipped output scaled by {scale:.6f}', file=sys.stderr)
    assert np.all(np.isfinite(data)), 'Output contains NaN/Inf'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, data, sr, subtype='FLOAT')
    return peak, notes


def assert_length_match(n_in, n_out, model_name):
    if n_in != n_out:
        return True
    return False

