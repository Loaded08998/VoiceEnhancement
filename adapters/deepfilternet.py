#!/usr/bin/env python3
import os
import sys
import time
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from adapters.common import load_wav, parse_adapter_args, print_result, save_wav, select_device


def main():
    args = parse_adapter_args('deepfilternet')
    dev = select_device(args.device)
    import numpy as np
    import torch
    import torchaudio

    # Shim missing torchaudio.backend.common.AudioMetaData in newer torchaudio
    backend = types.ModuleType('torchaudio.backend')
    common = types.ModuleType('torchaudio.backend.common')

    class AudioMetaData:
        pass

    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules['torchaudio.backend'] = backend
    sys.modules['torchaudio.backend.common'] = common
    torchaudio.backend = backend

    os.environ['XDG_CACHE_HOME'] = os.path.abspath('models/cache')

    data, sr = load_wav(args.input)
    assert sr == 48000, f'Expected 48000 Hz, got {sr}'
    n_in = len(data)

    from df.enhance import enhance, init_df

    model_res = init_df(post_filter=False, log_file=None)
    model, df_state = model_res[0], model_res[1]
    model = model.to(dev)

    t0 = time.time()
    with torch.inference_mode():
        # enhance expects CPU audio tensor [C, T] and internally transfers features to device
        x = torch.from_numpy(data)
        y = enhance(model, df_state, x[None]).squeeze().cpu().numpy()
    runtime = time.time() - t0

    if len(y) > n_in:
        y = y[:n_in]
    elif len(y) < n_in:
        y = np.pad(y, (0, n_in - len(y)))

    peak, notes = save_wav(args.output, y, sr, 'deepfilternet')
    print_result('deepfilternet', 'DeepFilterNet3', sr, dev, runtime, n_in, len(y), peak, notes)


if __name__ == '__main__':
    main()
