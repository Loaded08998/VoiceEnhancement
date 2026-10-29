#!/usr/bin/env python3
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from adapters.common import load_wav, parse_adapter_args, print_result, save_wav, select_device


def main():
    args = parse_adapter_args('metricgan_plus')
    dev = select_device(args.device)

    import numpy as np
    import soundfile as sf
    import torch

    data, sr = load_wav(args.input)
    assert sr == 16000, f'Expected 16000 Hz, got {sr}'
    n_in = len(data)

    os.environ['HF_HOME'] = os.path.abspath('models/hf')
    from speechbrain.inference.enhancement import SpectralMaskEnhancement

    enh = SpectralMaskEnhancement.from_hparams(
        source='speechbrain/metricgan-plus-voicebank',
        savedir='models/metricgan-plus-voicebank',
        run_opts={'device': dev},
    )

    t0 = time.time()
    with torch.inference_mode():
        x = torch.from_numpy(data).to(dev)
        lengths = torch.tensor([1.0], device=dev)
        y = enh.enhance_batch(x[None], lengths=lengths).squeeze().cpu().numpy()
    runtime = time.time() - t0

    # match length
    if len(y) > n_in:
        y = y[:n_in]
    elif len(y) < n_in:
        y = np.pad(y, (0, n_in - len(y)))

    peak, notes = save_wav(args.output, y, sr, 'metricgan_plus')
    print_result(
        'metricgan_plus',
        'speechbrain/metricgan-plus-voicebank',
        sr,
        dev,
        runtime,
        n_in,
        len(y),
        peak,
        notes,
    )


if __name__ == '__main__':
    main()

