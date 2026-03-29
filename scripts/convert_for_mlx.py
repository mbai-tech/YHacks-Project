"""Convert finetune.jsonl to MLX-LM format and split train/valid."""

import json
import random
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent
SRC   = ROOT / "data" / "finetune.jsonl"
OUTDIR = ROOT / "data" / "mlx_train"
OUTDIR.mkdir(exist_ok=True)

random.seed(42)

records = []
with open(SRC) as f:
    for line in f:
        p = json.loads(line)
        # MLX-LM chat format
        text = f"<|user|>\n{p['prompt']}\n<|assistant|>\n{p['completion']}"
        records.append({"text": text})

random.shuffle(records)

n_val   = max(50, int(len(records) * 0.1))
n_test  = max(20, int(len(records) * 0.05))
val     = records[:n_val]
test    = records[n_val:n_val + n_test]
train   = records[n_val + n_test:]

for split, data in [("train", train), ("valid", val), ("test", test)]:
    out = OUTDIR / f"{split}.jsonl"
    with open(out, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")
    print(f"{split}: {len(data)} records → {out}")
