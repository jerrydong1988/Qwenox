"""Independent NumPy DFlash2 equations on the SAME decoded HGN values.

Reference: z-lab/dflash 07ebd93db9f472af339b644bb70221ad8428328a,
dflash/model_mlx.py (dynamic convolution, noncausal SWA, greedy selector).
This validates arithmetic, not the quality of the original unquantized model.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np
from qwen27_reference import HGN, norm, silu


TAPS = [5, 19, 33, 47, 61]


def rope(x, positions):
    angles = np.asarray(positions)[:, None] * np.power(1e7, -np.arange(64, dtype=np.float64)/64)
    co, si = np.cos(angles).astype('f4')[:, None], np.sin(angles).astype('f4')[:, None]
    return np.concatenate([x[..., :64]*co-x[..., 64:]*si,
                           x[..., 64:]*co+x[..., :64]*si], axis=-1)


def convolve(x, dynamic, base, side):
    d = dynamic.reshape(len(x), 2, 2, 320)[:, side]
    out = np.zeros_like(x)
    for tap in range(2):
        if tap >= len(x):
            continue
        values = x if tap == 0 else np.concatenate([np.zeros_like(x[:1]), x[:-1]])
        out += values * (base[side, tap] + np.repeat(d[:, tap], 16, axis=-1))
    return out


def forward(ck, target, anchor, output, block=8, offset=0, full_vocab=False):
    output.mkdir(parents=True, exist_ok=True)
    lin = lambda n, x: ck.linear(n, x)
    w = lambda n: ck.rows('drafter.'+n)
    ctx = norm(lin('drafter.fc.weight', target), w('hidden_norm.weight'), False)
    np.save(output/'context.npy', ctx)
    x = ck.rows('embed_tokens.weight', anchor, anchor+1)
    x = np.concatenate([x, np.repeat(ck.rows('embed_tokens.weight', 248070, 248071), block-1, axis=0)])
    pos = len(ctx)+offset
    for layer in range(5):
        p = f'drafter.layers.{layer}.'
        ww = lambda n: ck.rows(p+n)
        mm = lambda n, a: lin(p+n+'.weight', a)
        n = norm(x, ww('input_layernorm.weight'), False)
        dynamic = mm('attention_conv.kernel_projection', n)
        n = convolve(n, dynamic, ww('attention_conv.base_kernel'), 0)
        q = rope(norm(mm('self_attn.q_proj', n).reshape(block,32,128), ww('self_attn.q_norm.weight'), False), np.arange(pos,pos+block))
        keys = np.concatenate([mm('self_attn.k_proj', ctx), mm('self_attn.k_proj', n)])
        keys = rope(norm(keys.reshape(-1,8,128), ww('self_attn.k_norm.weight'), False), np.arange(offset,pos+block))
        values = np.concatenate([mm('self_attn.v_proj', ctx), mm('self_attn.v_proj', n)]).reshape(-1,8,128)
        keys, values = np.repeat(keys,4,axis=1), np.repeat(values,4,axis=1)
        scores = np.einsum('qhd,khd->hqk',q,keys)/np.float32(np.sqrt(128))
        for t in range(block):
            scores[:,t,:max(0,len(ctx)+t-2047)] = -np.inf
        prob = np.exp(scores-scores.max(axis=-1,keepdims=True))
        prob /= prob.sum(axis=-1,keepdims=True)
        a = np.einsum('hqk,khd->qhd',prob,values).reshape(block,4096)
        x += convolve(mm('self_attn.o_proj',a),dynamic,ww('attention_conv.base_kernel'),1)
        n = norm(x,ww('post_attention_layernorm.weight'),False)
        dynamic = mm('mlp_conv.kernel_projection',n)
        n = convolve(n,dynamic,ww('mlp_conv.base_kernel'),0)
        f = mm('mlp.down_proj',silu(mm('mlp.gate_proj',n))*mm('mlp.up_proj',n))
        x += convolve(f,dynamic,ww('mlp_conv.base_kernel'),1)
        np.save(output/f'layer{layer}.npy',x)
    x = norm(x,w('norm.weight'),False)
    logits = lin('lm_head.weight' if full_vocab else 'draft_lm_head.weight',x)
    gate = lin('drafter.candidate_selector.hidden_projection.weight',x)
    _, _, off, size, _ = ck.records['draft_vocab_map']
    mapping = np.frombuffer(ck.mm,'<i4',count=size//4,offset=off)
    if full_vocab: mapping = np.arange(248320)
    proposals = []
    pred = anchor
    for t in range(1,block):
        top = np.argsort(-logits[t],kind='stable')[:16]
        ids = mapping[top]
        a = ck.rows('drafter.candidate_selector.predecessor_codebook',pred,pred+1)[0]*gate[t]
        b = np.stack([ck.rows('drafter.candidate_selector.successor_codebook',i,i+1)[0] for i in ids])
        score = logits[t,top] + b@a
        pred = int(ids[score.argmax()]); proposals.append(pred)
    np.save(output/'normalized.npy',x)
    np.save(output/'logits.npy',logits)
    np.save(output/'gate.npy',gate)
    (output/'result.json').write_text(json.dumps({'anchor':anchor,'proposals':proposals,'block':block,'offset':offset}),encoding='utf-8')
    return proposals


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('model');ap.add_argument('--target',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--anchor',type=int,default=5)
    ap.add_argument('--block',type=int,default=8)
    ap.add_argument('--full-vocab',action='store_true')
    args=ap.parse_args()
    target=np.concatenate([np.load(args.target/f'layer{i:02d}.npy') for i in TAPS],axis=-1)
    print(forward(HGN(args.model),target,args.anchor,args.output,args.block,full_vocab=args.full_vocab),flush=True)
