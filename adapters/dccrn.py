#!/usr/bin/env python3
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from adapters.common import load_wav, parse_adapter_args, print_result, save_wav, select_device

WEIGHTS = 'JorisCos/DCCRNet_Libri1Mix_enhsingle_16k'
CHUNK_S = 20
OVERLAP_S = 1


def main():
    args = parse_adapter_args('dccrn')
    dev = select_device(args.device)
    import numpy as np
    import torch

    os.environ['HF_HOME'] = os.path.abspath('models/hf')
    os.environ['ASTEROID_CACHE'] = os.path.abspath('models/asteroid')

    data, sr = load_wav(args.input)
    assert sr == 16000, f'Expected 16000 Hz, got {sr}'
    n_in = len(data)

    from asteroid.models import BaseModel

    m = BaseModel.from_pretrained(WEIGHTS).to(dev).eval()
    assert m.sample_rate == 16000

    # chunked inference
    from src.bench.audio_io import OverlapAddChunker

    chunker = OverlapAddChunker(data, sr, CHUNK_S, OVERLAP_S)
    chunks_out = []

    t0 = time.time()
    with torch.inference_mode():
        for chunk, start in chunker.chunks():
            x = torch.from_numpy(chunk).float().to(dev).unsqueeze(0)  # [1, T]
            y = m(x)  # [1, 1, T]
            y = y.squeeze().cpu().numpy()
            if len(y) > len(chunk):
                y = y[: len(chunk)]
            elif len(y) < len(chunk):
                y = np.pad(y, (0, len(chunk) - len(y)))
            chunks_out.append((y, start))
    runtime = time.time() - t0

    out = OverlapAddChunker.reconstruct(chunks_out, n_in, sr, CHUNK_S, OVERLAP_S)

    peak, notes = save_wav(args.output, out, sr, 'dccrn')
    notes = (notes + ' SUBSTITUTE_WEIGHTS').strip()
    print_result('dccrn', WEIGHTS, sr, dev, runtime, n_in, len(out), peak, notes)


if __name__ == '__main__':
    main()

