# GGUF kernels and Windows web launcher — 2026-11-15

Subsequent UI redesign: [LAUNCHER_REDESIGN_20261115.md](LAUNCHER_REDESIGN_20261115.md)
removes the decorative media dependency and connects maintained live/history
metrics. The appearance and media instructions below describe the initial port.

The maintained branch now supports the locally available Qwen3.8-Flash-Next
`qwen4exp` GGUF weights on Windows and an optional web launcher using the existing
native supervisor. This is a source integration with isolated runtime validation.
The two daily installations still use their original HGN weights and executables.

## Provenance and scope

Base: `8e76e5427177d39c11f5a40322edc60b50924db6`, `codex/qwen38-27b`.
Official source: <https://github.com/IIIIIllllIIIIIlllll/Qwenox>, reviewed tip
`4cb03125fda8e601dba4030cdb95310f9cc63da6`.

| Official change | Adaptation |
| --- | --- |
| `07bf813` | Windows pure-GGUF loading, IQ3_S/IQ4_XS/IQ4_NL/Q6_K support, expert kernels |
| `03a1401`, `1d7f882`, `3814895` | Native dense formats, Q6_K tuning and row-invariant WMMA verifier |
| `d7d0159` | Embedded GGUF tokenizer and web launcher assets/backend foundation |
| `87f71d9` | Draft/save/profile interaction |

This is not a wholesale merge. QDV1 draft-vocabulary selection, DSpine, the upstream
API-owned engine lifecycle, upstream performance overlay and unrelated console
changes are not included. This does not add Qwen27 GGUF support or general support
for arbitrary GGUF architectures.

The actual Flash UD-IQ4_XS model has IQ3_S/IQ4_NL and mixed IQ4_XS/Q8_0 experts.
Its dense map contains 512 Q8g32 tensors and one native Q6_K output matrix. The
native-format optimization avoids converting supported dense matrices to Q8g32;
the WMMA switch accelerates multi-row speculative verification when it helps.

## Maintained behavior

- HGN LUT/dequantization kernels retain their original implementation. GGUF LUT
  kernels live in a separate `gguf_moe` namespace, preventing an upstream helper
  replacement from changing HGN numerics.
- `GDEC_V2_HT_DEQ_EXACT=1` still uses the safe block-synchronized `hgnv2gpu::ht_dense`.
  The reverted streamed HT optimization is not restored. The Flash three-on
  settings remain enabled in the existing profiles.
- `QWENOX_GG_WMMA=1` selects native dense WMMA for multi-row verification up to
  48 rows; `0` retains the chunked GEMV path. Native dense layout and WMMA policy
  enter the GGUF snapshot fingerprint so different math policies do not reuse
  each other's persistent KV state. Persistent-cache resume was not exercised.
- Embedded GGUF tokenization shares the maintained tokenizer implementation.
  HGN tokenizer-directory loading remains available.

## Web launcher

`launcher_settings.exe` manages profiles, saved configuration, draft edits,
start/stop/restart, status and logs. Windows starts `start_win.exe --managed`
inside an owned kill-on-close job; the native supervisor retains ownership of
its engine/API children. An externally started service can be observed but is
not terminated by the web launcher.

Profiles preserve maintained private `GDEC_*`, `SPEC_*`, `V2_*`, `ROPE_*` policies
and API model names. Switching profiles removes stale per-model flags from the
previous profile. Config writes are validated and atomic, preserving a backup;
cross-origin management requests are rejected. Running-service links and health
checks keep the endpoint selected at start. Cancelled workers cannot revive
service state after a late health response, and stale monitors cannot stop a
newer service generation. Stop waits for the owned process group and its
original listening endpoints to close before reporting idle, preventing an
immediate restart from racing nested-job socket teardown.

The existing upstream visual theme is retained. Fields unsupported by the
maintained launcher are hidden; Flash/27B-specific controls follow the selected
model. Text fields have accessible labels, and profile/tab/format choices have
keyboard access. The inherited selected/dirty indicators and running-status glow
are intentional exceptions to the UI detector's three style warnings.

Build in a Windows Git Bash environment with the configured TheRock toolchain:

```sh
bash build_win.sh api
bash build_win.sh launcher
bash build_win.sh web-launcher
```

The web executable needs this branch's `start_win.exe`, engine and API, plus the
usual runtime DLLs/databases under `build/`. Keep `media/panel-bg.png` and
`media/qwen-charactor-alpha.png` next to the runtime. HTML is embedded at build
time. If an SDK resource compiler is available, the file icon is embedded too.

Start a prepared runtime with:

```powershell
.\launcher_settings.exe --root C:\path\to\prepared-runtime
```

Do not point it at an old installed `start_win.exe`: that executable lacks the
new managed/GGUF launch contract. The isolated validation runtime is
`build/ui-runtime`; its three profiles use test ports and 8K contexts and are
not replacements for the daily Flash 256K profile.

For GGUF, select the first Flash shard, its matching Q8_0 MTP sidecar, and an
appropriate API model name. The format marker is `# start_win: weights=gguf`;
the model keys are `GGUF_FILE`, `GGUF_MTP_FILE`, and optional `GGUF_VISION_FILE`.
An empty MTP field disables the sidecar. Image inference was not tested.

## Correctness and lifecycle evidence

All tests use the deployed-compatible adjacent HIP runtime, not the different
system HIP DLL. Model tests inspected the loaded HIP/BLAS module paths.

| Check | Result |
| --- | --- |
| Native dense WMMA: five formats, eight production shapes, F32/BF16 inputs | 80/80; row-count invariance and output guards pass; worst relative error 0.001120642 against independent CPU dequantization + FP64 dot products |
| Independent IQ3_S and Q6_K CPU dequantization | 1,048,576 floats per type; zero mismatches |
| Real GGUF expert layer, P=64 | No nonfinite values; 3 experts / 8 slots checked; up/down maximum relative L2 5.008e-4 / 2.148e-4 |
| Real Q6_K output matrix | 81,920 BF16 values match CPU dequantization; GEMV relative L2 8.273e-8; F32/BF16 row invariance passes |
| Embedded tokenizer | 131 encode/offset and 131 decode comparisons match HGN tokenizer; special IDs match; two malformed files rejected |
| GGUF full-model on/off comparison | 10 short and 6 long response messages identical with WMMA disabled/enabled |
| HGN installed baseline versus final engine | Flash and 27B each 12/12 token and logprob sequences identical; eight fixed-seed repeats per model identical |
| Maintained API regression | Full regression passes; bounded tool-resume suite passes 395 assertions |
| Web management | 50 assertions: private profile policies, stale-key removal, invalid-input preservation, cross-origin rejection, missing supervisor and cancelled startup probes |
| Browser workflows | Flash HGN, 27B HGN and Flash GGUF each started, returned the correct `42` API answer, then stopped through the UI |
| Responsive/interaction | Desktop, 800px and 390px inspected; no document horizontal overflow; no browser console errors; keyboard tab activation and draft/revert verified |

Validation was bounded to 8K model contexts with caches off. It is not a full
long-context, vision, concurrency-load, output-quality benchmark or Linux
validation. Windows autostart registry changes were not exercised.

## Measured benefit

These comparisons isolate **WMMA off versus on within the new GGUF implementation**.
They do not compare GGUF against HGN, or quantify the complete native-format
port against the former implementation. Both settings use the same weights,
runtime and prompts; only the verifier switch changes within each pair.

For chain MTP with gamma 8 and 512-token outputs, excluding the first run of each
prompt, the means of two remaining samples were:

| Prompt | WMMA off | WMMA on | Difference |
| --- | ---: | ---: | ---: |
| Python LRU cache | 28.40 tok/s | 29.05 tok/s | +2.3% |
| Chinese hash-collision explanation | 18.25 tok/s | 18.85 tok/s | +3.3% |

The short gamma-4 cases showed no speedup; on was generally slightly slower.
The long-output samples were run sequentially off then on, not randomized, and
the sample is small. Treat the gains as bounded observations, not a universal
performance guarantee. The switch is exposed for workload-specific comparison.

## Artifacts and installation boundary

Machine-readable identities, timings and checks:
[`benchmarks/gguf-launcher-20261115.json`](benchmarks/gguf-launcher-20261115.json).
Full local logs and screenshots remain under the isolated worktree's ignored
`build/` directory. Reusable tests are in `tools/gguf_*`,
`tools/ggd_dense_test.cu`, `tools/moe_gguf_test.cu` and
`tools/launcher_web_test.py`.

The arithmetic/regression candidate was built before the final GGUF snapshot
policy fingerprint bit. The final engine has identical arithmetic sources and
was rechecked against both HGN baselines and through real GGUF web inference.
Performance numbers retain the original measured executable identity.

No installed executable, launcher, configuration or model weight was replaced.
Both daily engine/config hashes match the previous deployment record; API,
launcher and runtime hashes were recorded again. Test processes were stopped
and both daily deployments remain stopped, matching their initial state.
