#!/usr/bin/env bash
WS=/home/krish/Desktop/VoiceEnhancement
cd "$WS"
echo "=== git log (last 5) ==="
git log --oneline -5 2>/dev/null || echo "no commits"
echo ""
echo "=== envs ==="
for e in envs/eval envs/asteroid envs/speechbrain envs/dfn envs/denoiser envs/cleanunet; do
  [ -d "$e" ] && echo "  ✓ $e" || echo "  ✗ $e"
done
echo ""
echo "=== models ==="
[ -d "models/metricgan-plus-voicebank" -o -d "models/hf/hub/models--speechbrain--metricgan-plus-voicebank" ] && echo "  ✓ metricgan-plus" || echo "  ✗ metricgan-plus"
[ -d "third_party/CleanUNet" ] && echo "  ✓ CleanUNet cloned" || echo "  ✗ CleanUNet"
[ -f "third_party/CleanUNet/exp/DNS-large-full/checkpoint/pretrained.pkl" ] && echo "  ✓ CleanUNet weights" || echo "  ✗ CleanUNet weights"
echo ""
echo "=== outputs ==="
for pair in results/*/; do
  [ -d "$pair" ] || continue
  n=$(find "$pair" -maxdepth 1 -name '*.wav' ! -name 'noisy_*' | wc -l)
  echo "  $pair: $n enhanced files"
done
echo ""
echo "=== recent logs ==="
for log in logs/*.log; do
  [ -f "$log" ] && { echo "--- $log (last 5 lines) ---"; tail -5 "$log"; }
done
echo ""
echo "=== NEXT ACTION ==="
grep "^## NEXT ACTION" -A 2 CONTEXT.md 2>/dev/null | tail -2
