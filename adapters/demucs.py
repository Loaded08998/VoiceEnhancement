#!/usr/bin/env python3
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from adapters.common import load_wav, parse_adapter_args, print_result, save_wav, select_device


def main():
    args = parse_adapter_args('demucs')
    dev = select_device(args.device)
    import numpy as np
    import torch

    os.environ['TORCH_HOME'] = os.path.abspath('models/torch')

    data, sr = load_wav(args.input)
    assert sr == 16000, f'Expected 16000 Hz, got {sr}'
    n_in = len(data)

    from denoiser import pretrained

    m = pretrained.dns64().to(dev).eval()
    assert m.sample_rate == 16000

    # normalise by std
    std = float(data.std())
    if std < 1e-6:
        std = 1.0
    data_norm = data / std

    t0 = time.time()
    with torch.inference_mode():
        x = torch.from_numpy(data_norm).float().to(dev).unsqueeze(0).unsqueeze(0)
        y = m(x).squeeze().cpu().numpy()
    runtime = time.time() - t0

    y = y * std  # denormalize

    if len(y) > n_in:
        y = y[:n_in]
    elif len(y) < n_in:
        y = np.pad(y, (0, n_in - len(y)))

    peak, notes = save_wav(args.output, y, sr, 'demucs')
    print_result('demucs', 'dns64', sr, dev, runtime, n_in, len(y), peak, notes)


if __name__ == '__main__':
    main()
