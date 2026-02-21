#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LRS="5e-5 1e-4 3e-4 1e-3 3e-3"

echo "=== Building ==="
make diffusion data
make flow_matching

echo ""
echo "=== Cleaning old output ==="
for lr in $LRS; do
  rm -rf "output/ddpm_lr${lr}" "output/flow_lr${lr}"
  rm -f "diffusion_${lr}.bin" "flow_matching_${lr}.bin"
done

# ── Flow Matching (fast: 100 Euler steps per sample) ─────────────
echo ""
echo "=== Flow Matching: 5 learning rates (parallel) ==="
FLOW_PIDS=""
for lr in $LRS; do
  tag="flow_lr${lr}"
  echo "  Starting $tag ..."
  stdbuf -oL ./flow_matching --lr "$lr" \
    --outdir "output/$tag" \
    --weights "flow_matching_${lr}.bin" \
    > "/tmp/${tag}.log" 2>&1 &
  FLOW_PIDS="$FLOW_PIDS $!"
done

echo "  Waiting for Flow Matching runs..."
for pid in $FLOW_PIDS; do wait "$pid"; done
echo "  All Flow Matching runs done."

# ── DDPM (slow: 1000 reverse steps per sample) ───────────────────
echo ""
echo "=== DDPM: 5 learning rates (2 at a time) ==="
LR_ARR=($LRS)

run_ddpm() {
  local lr=$1
  local tag="ddpm_lr${lr}"
  echo "  Starting $tag ..."
  stdbuf -oL ./diffusion --lr "$lr" \
    --outdir "output/$tag" \
    --weights "diffusion_${lr}.bin" \
    > "/tmp/${tag}.log" 2>&1
  echo "  $tag done."
}

# batch 1: first two
run_ddpm "${LR_ARR[0]}" &
run_ddpm "${LR_ARR[1]}" &
wait
# batch 2: next two
run_ddpm "${LR_ARR[2]}" &
run_ddpm "${LR_ARR[3]}" &
wait
# batch 3: last one
run_ddpm "${LR_ARR[4]}"

# ── Post-processing ──────────────────────────────────────────────
echo ""
echo "=== Generating GIFs ==="
for lr in $LRS; do
  python3 make_gif.py --dir "output/ddpm_lr${lr}" --duration 100
  python3 make_gif.py --dir "output/flow_lr${lr}" --duration 100
done

echo ""
echo "=== Plotting loss curves ==="
python3 plot_loss.py

echo ""
echo "=== Done ==="
ls -lh output/*_training.gif 2>/dev/null | awk '{print "  "$NF, $5}'
echo "  output/loss.png  output/loss.svg"
