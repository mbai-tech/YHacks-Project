#!/bin/bash
set -e

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_BASE="$HOME/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit/snapshots"
SNAPSHOT=$(ls "$MODEL_BASE")
MODEL="$MODEL_BASE/$SNAPSHOT"
DATA="$SCRIPT_DIR/data/mlx_train"
ADAPTERS="$SCRIPT_DIR/adapters/ct_water_risk"
PORT=11434

cd "$SCRIPT_DIR"
source .venv/bin/activate

# ── Commands ─────────────────────────────────────────────────────────────────
usage() {
  echo "Usage: ./train.sh [train|generate|serve]"
  echo ""
  echo "  train           Fine-tune with LoRA on data/mlx_train"
  echo "  prepare-future  Generate 2026–2050 prediction training data"
  echo "  retrain         Re-train with updated dataset (500 iters)"
  echo "  generate        Run a test prompt against the fine-tuned model"
  echo "  serve           Start local server on port $PORT for the Streamlit app"
  exit 1
}

case "${1:-}" in

  train)
    echo "▸ Starting LoRA fine-tune on $MODEL"
    echo "▸ Adapter output → $ADAPTERS"
    echo ""
    mkdir -p "$ADAPTERS"
    python -m mlx_lm lora \
      --model "$MODEL" \
      --data "$DATA" \
      --train \
      --iters 100 \
      --batch-size 2 \
      --num-layers 8 \
      --steps-per-report 1 \
      --steps-per-eval 100 \
      --val-batches 5 \
      --max-seq-length 512 \
      --save-every 100 \
      --adapter-path "$ADAPTERS"
    echo ""
    echo "✓ Training complete. Adapters saved to $ADAPTERS"
    ;;

  prepare-future)
    echo "▸ Generating future prediction training data (2026–2050)..."
    python "$SCRIPT_DIR/scripts/generate_future_predictions.py"
    echo ""
    echo "✓ Training data updated. Run './train.sh retrain' to fine-tune."
    ;;

  retrain)
    echo "▸ Re-training LoRA with updated dataset (500 iters)..."
    echo "▸ Adapter output → $ADAPTERS"
    echo ""
    mkdir -p "$ADAPTERS"
    python -m mlx_lm lora \
      --model "$MODEL" \
      --data "$DATA" \
      --train \
      --iters 100 \
      --batch-size 2 \
      --num-layers 8 \
      --steps-per-report 10 \
      --steps-per-eval 100 \
      --val-batches 10 \
      --max-seq-length 512 \
      --save-every 100 \
      --adapter-path "$ADAPTERS"
    echo ""
    echo "✓ Retraining complete. Adapters saved to $ADAPTERS"
    ;;

  generate)
    echo "▸ Running test prompt against fine-tuned model..."
    echo ""
    python -m mlx_lm generate \
      --model "$MODEL" \
      --adapter-path "$ADAPTERS" \
      --max-tokens 300 \
      --temp 0.1 \
      --prompt "<|user|>
What is the highest-risk facility in Hartford County, CT, and what makes it dangerous?
<|assistant|>
"
    ;;

  serve)
    echo "▸ Starting MLX server on http://127.0.0.1:$PORT"
    echo "▸ Base model: $MODEL"
    if [ -d "$ADAPTERS" ]; then
      echo "▸ Adapters: $ADAPTERS"
      ADAPTER_FLAG="--adapter-path $ADAPTERS"
    else
      echo "⚠ No adapters found at $ADAPTERS — serving base model"
      ADAPTER_FLAG=""
    fi
    echo ""
    python -m mlx_lm server \
      --model "$MODEL" \
      $ADAPTER_FLAG \
      --port $PORT
    ;;

  *)
    usage
    ;;
esac
