"""Validate all DFlash2 layers, logits, selector projection and final proposals."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('reference',type=Path);ap.add_argument('actual',type=Path)
    ap.add_argument('--position',type=int,default=4);ap.add_argument('--log',type=Path,required=True)
    args=ap.parse_args();errors=[]
    for name in ['layer0','layer1','layer2','layer3','layer4','normalized','logits','gate']:
        a=np.load(args.reference/(name+'.npy')).astype(np.float64)
        b=np.fromfile(args.actual/f't{args.position}-dflash-{name}.f32','<f4').reshape(a.shape).astype(np.float64)
        assert np.isfinite(a).all() and np.isfinite(b).all(),name
        item={'name':name,'relative_l2':float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30)),
              'max_abs':float(np.max(np.abs(a-b)))}
        assert item['relative_l2']<1e-4,item
        if name=='logits':
            def logsoftmax(v):
                v=v-v.max(-1,keepdims=True)
                return v-np.log(np.exp(v).sum(-1,keepdims=True))
            p,q=logsoftmax(a),logsoftmax(b)
            item['kl_max']=float((np.exp(p)*(p-q)).sum(-1).max())
            item['top1_match']=bool((a.argmax(-1)==b.argmax(-1)).all())
            assert item['kl_max']<1e-6 and item['top1_match'],item
        errors.append(item)
    expected=json.loads((args.reference/'result.json').read_text())['proposals']
    lines=args.log.read_text(encoding='utf-8-sig').splitlines()
    actual=[int(t) for t in next(line for line in lines if line.startswith('dflash proposals')).split()[2:]]
    assert expected==actual,(expected,actual)
    print(json.dumps({'pass':True,'errors':errors,'proposals':actual},indent=2))


if __name__=='__main__':main()
