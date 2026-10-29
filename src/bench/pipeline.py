import argparse
import glob
import json
import os
import subprocess
import sys
import time
import yaml
import numpy as np
import soundfile as sf

from src.bench.audio_io import decode_to_wav, resample_wav
from src.bench.make_pair import make_all_pairs
from src.bench.evaluate import score_pair
from src.bench.report import build_report


def parse_args():
    parser = argparse.ArgumentParser(description="Speech Enhancement Benchmark Pipeline")
    parser.add_argument("--snr-db", type=float, default=5.0, help="Mixing SNR in dB (default: 5.0)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Inference device")
    parser.add_argument("--no-resume", action="store_true", help="Force rerun even if results exist")
    parser.add_argument("--models", type=str, default="", help="Comma-separated subset of models to run")
    parser.add_argument("--pairs", type=str, default="", help="Comma-separated subset of pairs to run")
    parser.add_argument("--config", type=str, default="config/models.yaml", help="Path to models.yaml")
    parser.add_argument("--clean-dir", type=str, default="inputs/Clean", help="Clean audio directory")
    parser.add_argument("--noise-dir", type=str, default="inputs/Noise", help="Noise audio directory")
    parser.add_argument("--pairs-dir", type=str, default="inputs/pairs", help="Output directory for pairs")
    parser.add_argument("--refs-dir", type=str, default="references", help="Output directory for clean references")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for benchmark results")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory for execution logs")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline():
    args = parse_args()
    start_time = time.time()

    print("=" * 70)
    print("Speech Enhancement Benchmark Pipeline")
    print(f"  SNR: {args.snr_db} dB | Device: {args.device} | Resume: {not args.no_resume}")
    print("=" * 70)

    # 1. Load config
    cfg = load_config(args.config)
    all_models = cfg.get("models", {})
    if args.models:
        selected_model_names = [m.strip() for m in args.models.split(",") if m.strip()]
        for m in selected_model_names:
            if m not in all_models:
                raise ValueError(f"Unknown model: '{m}'. Available: {list(all_models.keys())}")
    else:
        selected_model_names = list(all_models.keys())

    print(f"Selected models ({len(selected_model_names)}): {', '.join(selected_model_names)}")

    # 2. Generate or verify pairs
    print("\n--- [Step 1/3] Generating / Verifying Audio Pairs ---")
    os.makedirs(args.pairs_dir, exist_ok=True)
    os.makedirs(args.refs_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    pairs_info = make_all_pairs(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        snr_db=args.snr_db,
        pairs_dir=args.pairs_dir,
        refs_dir=args.refs_dir,
    )

    if args.pairs:
        filter_stems = set(p.strip() for p in args.pairs.split(",") if p.strip())
        pairs_info = [p for p in pairs_info if p["stem"] in filter_stems]
        if not pairs_info:
            raise ValueError(f"No pairs matched filter: {args.pairs}")

    print(f"Total pairs to benchmark: {len(pairs_info)}")

    # 3. Process each pair
    print("\n--- [Step 2/3] Running Model Inferences & Scoring ---")
    all_pair_metrics = {}

    for p_idx, pair in enumerate(pairs_info, 1):
        stem = pair["stem"]
        pair_mp3 = pair["out_mp3"]
        ref_wav = pair["out_ref_wav"]
        pair_res_dir = os.path.join(args.results_dir, stem)
        pair_log_dir = os.path.join(args.logs_dir, stem)
        os.makedirs(pair_res_dir, exist_ok=True)
        os.makedirs(pair_log_dir, exist_ok=True)

        print(f"\n[{p_idx}/{len(pairs_info)}] Processing pair: {stem}")

        # Decode noisy mixture to native WAV
        noisy_native_wav = os.path.join(pair_res_dir, "noisy_native.wav")
        if not os.path.exists(noisy_native_wav) or args.no_resume:
            decode_to_wav(pair_mp3, noisy_native_wav, log_sha=False)

        # Resample to required sample rates
        required_srs = {16000} | {all_models[m]["native_sr"] for m in selected_model_names}
        for sr in required_srs:
            noisy_sr_wav = os.path.join(pair_res_dir, f"noisy_{sr}.wav")
            if not os.path.exists(noisy_sr_wav) or args.no_resume:
                resample_wav(noisy_native_wav, noisy_sr_wav, sr)

        # Run each model adapter
        enhanced_wavs = {}
        for m_name in selected_model_names:
            m_cfg = all_models[m_name]
            m_sr = m_cfg["native_sr"]
            m_env = m_cfg["env"]
            in_wav = os.path.join(pair_res_dir, f"noisy_{m_sr}.wav")
            out_wav = os.path.join(pair_res_dir, f"{m_name}.wav")
            log_path = os.path.join(pair_log_dir, f"{m_name}.log")

            if not args.no_resume and os.path.exists(out_wav) and os.path.getsize(out_wav) > 1000:
                print(f"  [{m_name}] Cached output exists (resuming)")
                enhanced_wavs[m_name] = out_wav
                continue

            python_bin = os.path.join(m_env, "bin", "python")
            cmd = [
                python_bin,
                f"adapters/{m_name}.py",
                "--in", in_wav,
                "--out", out_wav,
                "--device", args.device,
            ]

            sub_env = os.environ.copy()
            sub_env["HF_HOME"] = os.path.abspath("models/hf")
            sub_env["TORCH_HOME"] = os.path.abspath("models/torch")
            sub_env["XDG_CACHE_HOME"] = os.path.abspath("models/cache")
            sub_env["ASTEROID_CACHE"] = os.path.abspath("models/asteroid")
            sub_env["PYTHONPATH"] = os.path.abspath(".") + ":" + sub_env.get("PYTHONPATH", "")

            t0 = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=sub_env,
                )
                duration = time.time() - t0
                with open(log_path, "w") as lf:
                    lf.write(f"=== COMMAND: {' '.join(cmd)} ===\n")
                    lf.write(f"=== STDOUT ===\n{proc.stdout}\n")
                    lf.write(f"=== STDERR ===\n{proc.stderr}\n")
                    lf.write(f"=== EXIT: {proc.returncode} ===\n")

                if proc.returncode == 0 and os.path.exists(out_wav):
                    meta = {}
                    for line in reversed(proc.stdout.strip().splitlines()):
                        line = line.strip()
                        if line.startswith("{") and line.endswith("}"):
                            try:
                                meta = json.loads(line)
                                break
                            except Exception:
                                pass
                    peak_val = meta.get("peak", float("nan"))
                    print(f"  [{m_name}] OK ({duration:.2f}s, peak={peak_val:.3f})")
                    enhanced_wavs[m_name] = out_wav
                else:
                    print(f"  [{m_name}] FAILED (exit code {proc.returncode}, see {log_path})")
                    enhanced_wavs[m_name] = None
            except subprocess.TimeoutExpired:
                print(f"  [{m_name}] TIMEOUT after 600s")
                with open(log_path, "w") as lf:
                    lf.write(f"=== COMMAND: {' '.join(cmd)} ===\n")
                    lf.write("=== TIMEOUT after 600s ===\n")
                enhanced_wavs[m_name] = None
            except Exception as ex:
                print(f"  [{m_name}] ERROR: {ex}")
                with open(log_path, "w") as lf:
                    lf.write(f"=== COMMAND: {' '.join(cmd)} ===\n")
                    lf.write(f"=== EXCEPTION: {ex} ===\n")
                enhanced_wavs[m_name] = None

        # Score pair
        noisy_16k_wav = os.path.join(pair_res_dir, "noisy_16000.wav")
        pair_metrics = score_pair(
            ref_wav=ref_wav,
            noisy_wav=noisy_16k_wav,
            enhanced_wavs=enhanced_wavs,
            out_dir=pair_res_dir,
        )
        all_pair_metrics[stem] = pair_metrics

        base_sisdr = pair_metrics.get("baseline", {}).get("si_sdr", float("nan"))
        base_pesq = pair_metrics.get("baseline", {}).get("pesq_wb", float("nan"))
        print(f"  Score: Baseline SI-SDR={base_sisdr:.2f} dB, PESQ={base_pesq:.2f}")
        for m_name in selected_model_names:
            m_res = pair_metrics.get(m_name, {})
            if m_res.get("status") == "ok":
                s_sdr = m_res.get("si_sdr", float("nan"))
                s_sdri = m_res.get("si_sdri", float("nan"))
                s_pesq = m_res.get("pesq_wb", float("nan"))
                s_dpesq = m_res.get("delta_pesq", float("nan"))
                print(f"    {m_name:15s} SI-SDR={s_sdr:6.2f} (Δ={s_sdri:+5.2f}) | PESQ={s_pesq:4.2f} (Δ={s_dpesq:+5.2f})")
            else:
                print(f"    {m_name:15s} status={m_res.get('status')}")

    # 4. Generate Master Report
    print("\n--- [Step 3/3] Generating Master Benchmark Report ---")
    report_path = os.path.join(args.results_dir, "report.md")
    report_content = build_report(all_pair_metrics, report_path)
    total_time = time.time() - start_time
    print(f"Benchmark completed in {total_time:.1f}s.")
    print(f"Report written to: {report_path}")

    # Also save aggregated metrics.json
    agg_metrics_path = os.path.join(args.results_dir, "all_metrics.json")
    with open(agg_metrics_path, "w") as f:
        json.dump(all_pair_metrics, f, indent=2, default=str)
    print(f"Consolidated metrics saved to: {agg_metrics_path}")


if __name__ == "__main__":
    run_pipeline()
