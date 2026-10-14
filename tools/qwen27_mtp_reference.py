"""Independent NumPy MTP reference using the same HGN and saved target states.

Equations follow vLLM qwen3_5_mtp.py: normalized final target state at t,
embedding t+1, concat [embedding, hidden], one dense full-attention layer.
"""
import argparse
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np
from qwen27_reference import HGN, norm, rope, sigmoid, silu


def forward(ck, target, next_ids, output):
    hidden = norm(np.load(target / 'layer63.npy'), ck.rows('norm.weight'))
    emb = np.stack([ck.rows('embed_tokens.weight', t, t+1)[0] for t in next_ids])
    joined = np.concatenate([norm(emb, ck.rows('mtp.pre_fc_norm_embedding.weight')),
                             norm(hidden, ck.rows('mtp.pre_fc_norm_hidden.weight'))], -1)
    x = ck.linear('mtp.fc.weight', joined)
    p = len(next_ids)
    prefix = 'mtp.layers.0.'
    w = lambda key: ck.rows(prefix+key)
    lin = lambda key, val: ck.linear(prefix+key+'.weight', val)
    z = norm(x, w('input_layernorm.weight'))
    qg = lin('self_attn.q_proj', z).reshape(p, 24, 512)
    q, gate = qg[..., :256], qg[..., 256:]
    k = lin('self_attn.k_proj', z).reshape(p, 4, 256)
    v = lin('self_attn.v_proj', z).reshape(p, 4, 256)
    q = rope(norm(q, w('self_attn.q_norm.weight')), np.arange(p))
    k = rope(norm(k, w('self_attn.k_norm.weight')), np.arange(p))
    k, v = np.repeat(k, 6, 1), np.repeat(v, 6, 1)
    scores = np.einsum('qhd,khd->hqk', q, k)/np.float32(16)
    scores[:, np.triu_indices(p, 1)[0], np.triu_indices(p, 1)[1]] = -np.inf
    probs = np.exp(scores-scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    attn = np.einsum('hqk,khd->qhd', probs, v)*sigmoid(gate)
    x += lin('self_attn.o_proj', attn.reshape(p, 6144))
    z = norm(x, w('post_attention_layernorm.weight'))
    x += lin('mlp.down_proj', silu(lin('mlp.gate_proj', z))*lin('mlp.up_proj', z))
    logits = ck.linear('lm_head.weight', norm(x, ck.rows('mtp.norm.weight')))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output/'logits.npy', logits)
    print({'argmax': logits.argmax(-1).tolist()}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--target', type=Path, required=True)
    ap.add_argument('--next-tokens', default='2,3,4,5')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    forward(HGN(args.model), args.target, [int(x) for x in args.next_tokens.split(',')], args.output)
