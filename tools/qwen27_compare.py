"""Compare native FP32 traces to independently decoded NumPy HGN execution."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference, actual):
    expected = np.load(reference / "embedding.npy")
    errors = []
    worst = 0.0
    for token in range(len(expected)):
        for layer in range(-1, 64):
            name = "embedding" if layer < 0 else f"layer{layer}"
            refname = "embedding.npy" if layer < 0 else f"layer{layer:02d}.npy"
            ref = np.load(reference / refname)[token].astype(np.float64)
            got = np.fromfile(actual / f"t{token}-{name}.f32", "<f4").astype(np.float64)
            if got.shape != ref.shape or not np.isfinite(got).all():
                raise AssertionError(f"invalid trace {token}/{name}")
            rel = float(np.linalg.norm(got-ref)/max(np.linalg.norm(ref), 1e-20))
            worst = max(worst, rel)
            if rel > 1e-4:
                errors.append({"token": token, "layer": layer, "relative_l2": rel})
    ref = np.load(reference / "logits.npy").astype(np.float64)
    got = np.stack([np.fromfile(actual/f"t{i}-logits.f32", "<f4") for i in range(len(ref))]).astype(np.float64)
    def logsoftmax(x):
        x = x-x.max(-1, keepdims=True)
        return x-np.log(np.exp(x).sum(-1, keepdims=True))
    p, q = logsoftmax(ref), logsoftmax(got)
    kl = (np.exp(p)*(p-q)).sum(-1)
    result = {"pass": not errors and bool((kl < 1e-6).all()) and bool((ref.argmax(-1)==got.argmax(-1)).all()),
              "worst_layer_relative_l2": worst, "logit_max_abs": float(np.abs(got-ref).max()),
              "kl": kl.tolist(), "expected_tokens": ref.argmax(-1).tolist(),
              "actual_tokens": got.argmax(-1).tolist(), "failures": errors[:12]}
    print(json.dumps(result, indent=2))
    return result["pass"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("actual", type=Path)
    args = ap.parse_args()
    raise SystemExit(0 if compare(args.reference, args.actual) else 1)
