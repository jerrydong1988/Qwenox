"""Independent NumPy FP32 reference for the Qwen3.8-27B HGN text trunk.

No GPU, model download, requantization, or full-model float allocation. Matrices
are decoded in bounded row blocks. Equations follow the Qwen3_5 reference:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
"""
import argparse
import json
import mmap
import os
import struct
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np


def bf16(x):
    return (x.astype(np.uint32) << 16).view(np.float32)


def e4m3(x):
    e = ((x >> 3) & 15).astype(np.int32)
    m = (x & 7).astype(np.float32)
    y = np.where(e == 0, m * 2.0**-9, (1 + m / 8) * np.exp2(e - 7))
    y = (y * np.where(x & 128, -1, 1)).astype(np.float32)
    return np.where((e == 15) & (m == 7), np.nan, y)


class HGN:
    def __init__(self, path):
        self.file = open(path, "rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        h = struct.unpack_from("<4sIIIQQQ64s", self.mm)
        if h[0] != b"HGN1" or h[1] != 2 or h[6] != len(self.mm):
            raise ValueError("invalid HGN header")
        if not h[7].startswith(b"Qwen3.8-27B"):
            raise ValueError("not a Qwen3.8-27B checkpoint")
        self.records = {}
        for i in range(h[2]):
            r = struct.unpack_from("<96sII4QQQQ", self.mm, h[4] + i * 160)
            name = r[0].split(b"\0")[0].decode()
            self.records[name] = (r[1], r[3:3 + r[2]], r[7], r[8], r[9] >> 32)

    def rows(self, name, start=0, stop=None):
        dt, shape, offset, size, qp = self.records[name]
        count = int(np.prod(shape))
        if dt == 0:
            if size != count * 2:
                raise ValueError(name)
            arr = np.frombuffer(self.mm, "<u2", count=count, offset=offset).reshape(shape)
            return bf16(arr[start:stop])
        rows, cols = shape
        stop = rows if stop is None else stop
        if qp != 0 and not (dt == 5 and qp == 1):
            raise ValueError(f"unsupported qparam for {name}: {qp}")
        if dt == 6:
            if size != count + rows * 2:
                raise ValueError(name)
            codes = np.frombuffer(self.mm, "u1", count=count, offset=offset).reshape(shape)
            scales = bf16(np.frombuffer(self.mm, "<u2", count=rows, offset=offset+count)[start:stop])
            return e4m3(codes[start:stop]) * scales[:, None]
        if dt == 5:
            stride = cols // 2 + cols // 16
            if cols % 256 or size != 64 + rows * stride:
                raise ValueError(name)
            cb = np.frombuffer(self.mm, "<f4", count=16, offset=offset)
            packed = np.frombuffer(self.mm, "u1", count=rows*stride, offset=offset+64).reshape(rows, stride)[start:stop]
            codes = packed[:, :cols//2]
            nib = np.empty((stop-start, cols), "u1")
            nib[:, 0::2], nib[:, 1::2] = codes & 15, codes >> 4
            scales = np.repeat(packed[:, cols//2:].copy().view('<f2').astype(np.float32), 32, axis=1) if qp == 1 else np.repeat(e4m3(packed[:, cols//2:]), 16, axis=1)
            return cb[nib] * scales
        raise ValueError(f"unsupported dtype {dt}: {name}")

    def linear(self, name, x, block=512):
        rows = self.records[name][1][0]
        out = np.empty((*x.shape[:-1], rows), np.float32)
        for lo in range(0, rows, block):
            hi = min(rows, lo+block)
            out[..., lo:hi] = x @ self.rows(name, lo, hi).T
        return out


def sigmoid(x):
    return np.exp(-np.logaddexp(np.float32(0), -x))


def silu(x):
    return x * sigmoid(x)


def norm(x, w, centered=True):
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + np.float32(1e-6)) * (w + int(centered))


def rope(x, positions):
    y = x.copy()
    angles = positions[:, None] * np.power(1e7, -np.arange(32, dtype=np.float64)/32)
    co, si = np.cos(angles).astype(np.float32)[:, None, :], np.sin(angles).astype(np.float32)[:, None, :]
    y[..., :32] = x[..., :32] * co - x[..., 32:64] * si
    y[..., 32:64] = x[..., 32:64] * co + x[..., :32] * si
    return y


def forward(ck, ids, output, limit=64):
    output.mkdir(parents=True, exist_ok=True)
    x = np.stack([ck.rows("embed_tokens.weight", t, t+1)[0] for t in ids])
    np.save(output / "embedding.npy", x)
    p = len(ids)
    for layer in range(limit):
        started = time.monotonic()
        prefix = f"layers.{layer}."
        w = lambda key: ck.rows(prefix + key)
        lin = lambda key, val: ck.linear(prefix + key + ".weight", val)
        z = norm(x, w("input_layernorm.weight"))
        if layer % 4 != 3:
            key = "linear_attn."
            raw = lin(key+"in_proj_qkv", z)
            gate = lin(key+"in_proj_z", z).reshape(p, 48, 128)
            a, b = lin(key+"in_proj_a", z), lin(key+"in_proj_b", z)
            conv_w = w(key+"conv1d.weight")[:, 0, :]
            padded = np.pad(raw, ((3, 0), (0, 0)))
            conv = silu(sum(padded[i:i+p] * conv_w[:, i] for i in range(4)))
            q = conv[:, :2048].reshape(p, 16, 128)
            k = conv[:, 2048:4096].reshape(p, 16, 128)
            v = conv[:, 4096:].reshape(p, 48, 128)
            q = q / np.sqrt(np.sum(q*q, -1, keepdims=True)+np.float32(1e-6)) / np.sqrt(np.float32(128))
            k = k / np.sqrt(np.sum(k*k, -1, keepdims=True)+np.float32(1e-6))
            q, k = np.repeat(q, 3, 1), np.repeat(k, 3, 1)
            decay = np.exp(-np.exp(w(key+"A_log")) * np.logaddexp(np.float32(0), a+w(key+"dt_bias")))
            beta = sigmoid(b)
            state = np.zeros((48, 128, 128), np.float32)
            y = np.empty((p, 48, 128), np.float32)
            for pos in range(p):
                state *= decay[pos, :, None, None]
                delta = (v[pos] - np.einsum("hkv,hk->hv", state, k[pos])) * beta[pos, :, None]
                state += k[pos, :, :, None] * delta[:, None, :]
                y[pos] = np.einsum("hkv,hk->hv", state, q[pos])
            y = norm(y, w(key+"norm.weight"), False) * silu(gate)
            y = lin(key+"out_proj", y.reshape(p, 6144))
        else:
            key = "self_attn."
            qg = lin(key+"q_proj", z).reshape(p, 24, 512)
            q, gate = qg[..., :256], qg[..., 256:]
            k = lin(key+"k_proj", z).reshape(p, 4, 256)
            v = lin(key+"v_proj", z).reshape(p, 4, 256)
            q = rope(norm(q, w(key+"q_norm.weight")), np.arange(p))
            k = rope(norm(k, w(key+"k_norm.weight")), np.arange(p))
            k, v = np.repeat(k, 6, 1), np.repeat(v, 6, 1)
            score = np.einsum("qhd,khd->hqk", q, k) / np.float32(16)
            score[:, np.triu_indices(p, 1)[0], np.triu_indices(p, 1)[1]] = -np.inf
            prob = np.exp(score-score.max(-1, keepdims=True))
            prob /= prob.sum(-1, keepdims=True)
            y = np.einsum("hqk,khd->qhd", prob, v) * sigmoid(gate)
            y = lin(key+"o_proj", y.reshape(p, 6144))
        x = x+y
        z = norm(x, w("post_attention_layernorm.weight"))
        x = x + lin("mlp.down_proj", silu(lin("mlp.gate_proj", z))*lin("mlp.up_proj", z))
        if not np.isfinite(x).all():
            raise ValueError(f"nonfinite layer {layer}")
        np.save(output / f"layer{layer:02d}.npy", x)
        print(json.dumps({"layer": layer, "seconds": time.monotonic()-started, "rms": float(np.sqrt(np.mean(x*x)))}), flush=True)
    if limit == 64:
        z = norm(x, ck.rows("norm.weight"))
        logits = ck.linear("lm_head.weight", z)
        np.save(output / "logits.npy", logits)
        print(json.dumps({"argmax": logits.argmax(-1).tolist()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", default="248045,872,198,9707,248046,198,248045,77091,198")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=64)
    args = ap.parse_args()
    forward(HGN(args.model), [int(t) for t in args.tokens.split(",")], args.output, args.layers)
