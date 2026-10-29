#!/usr/bin/env python3
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath('third_party/CleanUNet'))
from adapters.common import load_wav, parse_adapter_args, print_result, save_wav, select_device

WEIGHTS = 'third_party/CleanUNet/exp/DNS-large-full/checkpoint/pretrained.pkl'
WEIGHTS_SHA = '145c101eb5bbfa3ba52fb2b4ec7e5b64a361c102f89291f75e1dd42601d95dc9'
CHUNK_S = 10
OVERLAP_S = 1


def verify_sha(path, expected):
    import hashlib

    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    actual = h.hexdigest()
    assert actual == expected, f'SHA256 mismatch: {actual} != {expected}'


def main():
    args = parse_adapter_args('cleanunet')
    dev = select_device(args.device)
    import numpy as np
    import torch

    verify_sha(WEIGHTS, WEIGHTS_SHA)

    cfg_path = 'third_party/CleanUNet/configs/DNS-large-full.json'
    with open(cfg_path) as f:
        cfg = json.load(f)['network_config']

    from network import CleanUNet

    net = CleanUNet(**cfg)
    ckpt = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt['model_state_dict'])
    net = net.to(dev).eval()

    data, sr = load_wav(args.input)
    assert sr == 16000, f'Expected 16000 Hz, got {sr}'
    n_in = len(data)

    from src.bench.audio_io import OverlapAddChunker

    chunker = OverlapAddChunker(data, sr, CHUNK_S, OVERLAP_S)
    chunks_out = []

    t0 = time.time()
    with torch.inference_mode():
        for chunk, start in chunker.chunks():
            x = torch.from_numpy(chunk.copy()).float().to(dev)
            x = x.unsqueeze(0).unsqueeze(0)  # [1,1,T]
            y = net(x.clone()).squeeze().cpu().numpy()  # clone: forward normalises in-place
            if len(y) > len(chunk):
                y = y[: len(chunk)]
            elif len(y) < len(chunk):
                y = np.pad(y, (0, len(chunk) - len(y)))
            chunks_out.append((y, start))
    runtime = time.time() - t0

    out = OverlapAddChunker.reconstruct(chunks_out, n_in, sr, CHUNK_S, OVERLAP_S)

    peak, notes = save_wav(args.output, out, sr, 'cleanunet')
    print_result('cleanunet', WEIGHTS, sr, dev, runtime, n_in, len(out), peak, notes)


if __name__ == '__main__':
    main()

