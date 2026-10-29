import json
import os
from collections import defaultdict
import numpy as np


def build_report(all_pair_metrics: dict, out_md_path: str) -> str:
    """Build a comprehensive markdown report from all pair metrics."""
    lines = []
    lines.append('# Speech Enhancement Benchmark Report')
    lines.append('')
    lines.append('Benchmark evaluation across 9 speech-noise pairs at 5 dB SNR.')
    lines.append('')

    # Master table
    lines.append('## 1. Master Results Table')
    lines.append('')
    lines.append('| Pair | Model | SI-SDR (dB) | SI-SDRi (dB) | PESQ-WB | ΔPESQ | Status |')
    lines.append('|------|-------|-------------|--------------|---------|-------|--------|')

    # Gather rows
    # Each entry in all_pair_metrics: { pair_stem: { model_name: { 'si_sdr': ..., 'si_sdri': ..., 'pesq_wb': ..., 'delta_pesq': ..., 'status': ... } } }
    model_stats = defaultdict(lambda: {'sisdri': [], 'pesq': []})
    noise_model_sisdri = defaultdict(lambda: defaultdict(list))
    speech_model_pesq = defaultdict(lambda: defaultdict(list))

    sorted_pairs = sorted(all_pair_metrics.keys())
    for pair in sorted_pairs:
        pair_dict = all_pair_metrics[pair]
        # Parse clean and noise stems: e.g. Speech1+market -> speech=Speech1, noise=market
        parts = pair.split('+', 1)
        speech_stem = parts[0]
        noise_stem = parts[1] if len(parts) > 1 else 'unknown'

        # Sort models in this pair by SI-SDR descending (baseline first or included in sort)
        items = []
        for model_name, m in pair_dict.items():
            sisdr = m.get('si_sdr', float('nan'))
            items.append((model_name, m, -9999.0 if np.isnan(sisdr) else sisdr))
        items.sort(key=lambda x: x[2], reverse=True)

        for model_name, m, _ in items:
            s_sdr = (
                f"{m.get('si_sdr', float('nan')):.2f}"
                if not np.isnan(m.get('si_sdr', float('nan')))
                else 'N/A'
            )
            s_sdri = (
                f"{m.get('si_sdri', float('nan')):.2f}"
                if not np.isnan(m.get('si_sdri', float('nan')))
                else 'N/A'
            )
            s_pesq = (
                f"{m.get('pesq_wb', float('nan')):.2f}"
                if not np.isnan(m.get('pesq_wb', float('nan')))
                else 'N/A'
            )
            s_dpesq = (
                f"{m.get('delta_pesq', float('nan')):.2f}"
                if not np.isnan(m.get('delta_pesq', float('nan')))
                else 'N/A'
            )
            status = m.get('status', 'ok')

            lines.append(
                f'| {pair} | {model_name} | {s_sdr} | {s_sdri} | {s_pesq} | {s_dpesq} | {status} |'
            )

            if model_name != 'baseline' and status == 'ok':
                val_sdri = m.get('si_sdri', float('nan'))
                val_pesq = m.get('pesq_wb', float('nan'))
                if not np.isnan(val_sdri):
                    model_stats[model_name]['sisdri'].append(val_sdri)
                    noise_model_sisdri[noise_stem][model_name].append(val_sdri)
                if not np.isnan(val_pesq):
                    model_stats[model_name]['pesq'].append(val_pesq)
                    speech_model_pesq[speech_stem][model_name].append(val_pesq)

    lines.append('')

    # Overall Summary Ranking
    lines.append('## 2. Overall Model Rankings')
    lines.append('')
    lines.append('| Model | Mean SI-SDRi (dB) | Mean PESQ-WB | Evaluated Pairs |')
    lines.append('|-------|-------------------|--------------|-----------------|')

    overall_rankings = []
    for model_name, data in model_stats.items():
        m_sdri = np.mean(data['sisdri']) if data['sisdri'] else float('nan')
        m_pesq = np.mean(data['pesq']) if data['pesq'] else float('nan')
        n_pairs = len(data['sisdri'])
        overall_rankings.append((model_name, m_sdri, m_pesq, n_pairs))

    overall_rankings.sort(key=lambda x: -9999.0 if np.isnan(x[1]) else x[1], reverse=True)
    for model_name, m_sdri, m_pesq, n_pairs in overall_rankings:
        sdri_str = f'{m_sdri:.2f}' if not np.isnan(m_sdri) else 'N/A'
        pesq_str = f'{m_pesq:.2f}' if not np.isnan(m_pesq) else 'N/A'
        lines.append(f'| {model_name} | {sdri_str} | {pesq_str} | {n_pairs}/9 |')

    lines.append('')

    # Best model per noise type
    lines.append('## 3. Best Model per Noise Type (by Mean SI-SDRi)')
    lines.append('')
    lines.append('| Noise Type | Best Model | Mean SI-SDRi (dB) |')
    lines.append('|------------|------------|-------------------|')
    for noise_stem in sorted(noise_model_sisdri.keys()):
        best_model = None
        best_val = -float('inf')
        for model_name, vals in noise_model_sisdri[noise_stem].items():
            if vals and np.mean(vals) > best_val:
                best_val = np.mean(vals)
                best_model = model_name
        if best_model:
            lines.append(f'| {noise_stem} | {best_model} | {best_val:.2f} |')
        else:
            lines.append(f'| {noise_stem} | N/A | N/A |')

    lines.append('')

    # Best model per speech file
    lines.append('## 4. Best Model per Speech File (by Mean PESQ-WB)')
    lines.append('')
    lines.append('| Speech File | Best Model | Mean PESQ-WB |')
    lines.append('|-------------|------------|--------------|')
    for speech_stem in sorted(speech_model_pesq.keys()):
        best_model = None
        best_val = -float('inf')
        for model_name, vals in speech_model_pesq[speech_stem].items():
            if vals and np.mean(vals) > best_val:
                best_val = np.mean(vals)
                best_model = model_name
        if best_model:
            lines.append(f'| {speech_stem} | {best_model} | {best_val:.2f} |')
        else:
            lines.append(f'| {speech_stem} | N/A | N/A |')

    lines.append('')

    # Caveats Section
    lines.append('## 5. Methodology & Caveats')
    lines.append('')
    lines.append(
        '1. **Substitute Weights**: Asteroid models (`convtasnet` and `dccrn`) use'
        ' `JorisCos/*_Libri1Mix_enhsingle_16k` checkpoints trained on LibriMix speech separation'
        ' data rather than general DNS challenge noise.'
    )
    lines.append(
        '2. **Native Sampling Rates**: DeepFilterNet3 processes audio natively at 48 kHz;'
        ' audio was resampled using soxr VHQ to 48 kHz for inference and back to 16 kHz for'
        ' standardized PESQ-WB and SI-SDR evaluation.'
    )
    lines.append(
        '3. **Single SNR Level**: Mixtures were generated at a single controlled SNR of 5 dB.'
        ' Performance may vary at lower (-5 dB, 0 dB) or higher (>10 dB) SNR levels.'
    )
    lines.append(
        '4. **Chunking Artifacts**: Long clips were processed with OverlapAddChunker (1-second'
        ' Hann crossfade) where specified to avoid GPU OOM.'
    )
    lines.append(
        '5. **Evaluation Metric Alignment**: All outputs were aligned to reference speech using'
        ' FFT cross-correlation before computing SI-SDR and PESQ-WB.'
    )
    lines.append('')

    report_content = '\n'.join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(out_md_path)), exist_ok=True)
    with open(out_md_path, 'w') as f:
        f.write(report_content)

    return report_content

