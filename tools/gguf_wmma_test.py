"""Prepare the actual native-GGUF host dispatch for the standalone GPU test.
Build tools/gguf_wmma_test.cu with HIP and -Ibuild after running this script.
"""
from pathlib import Path
root=Path(__file__).resolve().parent.parent
source=(root/'src/gpu/parts/30_host_util.inc').read_text(encoding='utf8')
a=source.index('// --- GGUF native dense (dtype')
b=source.index('// ht / q6g64 -> bf16',a)
(root/'build').mkdir(exist_ok=True)
(root/'build/gguf_dense_host_under_test.inc').write_text(source[a:b],encoding='utf8')
