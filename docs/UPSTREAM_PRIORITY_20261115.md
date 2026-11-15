# Qwenox upstream selection — 2026-11-15

Follow-up: the subsequently authorized Flash GGUF kernel and Windows web-launcher
adaptation is recorded in [GGUF_LAUNCHER_20261115.md](GGUF_LAUNCHER_20261115.md).
The deferred decisions below describe the earlier selection pass.

## Reviewed range and decision

Official repository: <https://github.com/IIIIIllllIIIIIlllll/Qwenox>.
Reviewed `106f2ae6bb95a21c8c466e1a2ccc5e8532d55e7a..4cb03125fda8e601dba4030cdb95310f9cc63da6`
(five new commits). The remote main tip was rechecked after implementation.
Maintained starting point: `d2d0987b0be58a6e8801bfc062edcaee911678db` on
`codex/qwen38-27b`. Integration was developed in the isolated
`codex/upstream-priority-20261115` worktree.

| Official commit | Decision | Reason |
| --- | --- | --- |
| `3814895` GGUF dense WMMA and draft-vocabulary gathering | Deferred | This fork does not contain upstream's native GGUF dense kernel/dispatch implementation. Both current deployments use HGN. This commit depends on the earlier deferred dense-WMMA/QDV1 architecture; copying the host dispatch would not be a standalone HGN optimization. No speed benefit is claimed for the deployed models. |
| `d7d0159` web launcher and embedded GGUF tokenizer | Deferred | Replaces the native launch flow and adds a new configuration service. Embedded GGUF tokenization can be considered separately when adopting that model format; current HGN profiles already use explicit, validated tokenizer directories. |
| `9597795` unfinished tool-call recovery | Adapted selectively | Backport bounded same-slot continuation. Preserve the maintained strict schema validation and structured error contract instead of passing through unknown functions or missing required arguments. |
| `87f71d9` launcher draft/save interaction | Deferred | Depends on the new web launcher, which has not been adopted. |
| `4cb0312` console navigation and responsive sampling profiles | Deferred | Targets upstream static console assets. This branch uses its own embedded dashboard and native launcher. |

## Resulting behavior

Chat Completions, Responses and Messages, both JSON and SSE, can make one continuation
when an ordinary text/tool turn naturally ends inside a syntactically resumable tool call.
The continuation:

- Retains the same API slot/engine connection and appends the exact generated token IDs.
- Uses the smaller of the original effective token budget, remaining context and
  `QWENOX_TOOL_CALL_RESUME_MAX` (default 8192). It does not restart a request after a real
  length limit, caller stop or disconnect.
- Suppresses the maintained engine's actual EOS token set plus `endoftext` for sampling,
  while preserving any client-provided biases. Greedy mode gets one attempt without bias.
- Stops at the tool-call closer and restores that closer through the streaming parser.
  The internal stop does not become an Anthropic caller `stop_sequence`.
- Counts repair output tokens and timing once in the HTTP result. Prompt usage remains
  the original prompt; existing per-GEN operational telemetry still records both segments.
- Leaves unknown tools, broken framing and invalid arguments under the existing strict
  diagnostic contract. An unsuccessful repair remains an explicit tool-call error;
  its private repair ceiling is not reported as client token exhaustion.

Unlike the full upstream guard, this deliberately does not introduce think-block repair,
reasoning-only tool extraction, seed retries, JSON tool syntax or raw output-tail logging.
Vision requests and forced tool choices retain the previous path. A completed call is
still checked against the original schema before being accepted.

GPU kernels, the safe reference HT dequantization path, `service.conf`, launchers and
model/tokenizer files are unchanged. Flash retains `GDEC_GR_COMBINE_SIG=1`,
`GDEC_AXPY_SG_OUT=1`, `GDEC_LT_BUCKET=1`, and `GDEC_V2_HT_DEQ_EXACT=1`.

## Validation

- Windows native API compiled with AMD Clang 24, existing pinned libwebp 1.6.0 static
  dependency, and no GPU compilation or SDK/runtime installation.
- Final API SHA-256: `396aaad2400b8f2460ad26573141911585648227648a6278c8fa7c1af6e712d6`.
- New `tools/api_tool_resume_test.py`: 395 assertions with one slot and another 395 with
  two slots. Covers all three API contracts, SSE argument reconstruction, same connection,
  exact prompt-ID append, usage, EOS/client bias, greedy mode, real context clamp, budget
  exhaustion, private ceiling, stop strings, broken framing and cancellation during repair.
- The deployed baseline API fails the new bare-opener recovery case with its previous
  `invalid_tool_call` error; the candidate completes the call. This is a demonstrated
  behavior change, not just a parser-unit expectation.
- Existing parser tests (91 checks), full API regression, strict tool failure/diagnostic
  suite, Messages (26 checks), disconnect/drain, live metrics, and literal-token/cache
  protection suites passed. Final EOS-set adaptation was followed by rerunning the new
  single/double-slot suite, full API regression and strict tool failure suite.
- Final-build real-model comparison: Flash and Qwen27 each passed 12 baseline and 12
  candidate requests (48 total): arithmetic and a schema-checked file-read tool call,
  across all three APIs and both stream modes. Normalized content, tool arguments and
  terminal status matched baseline in every case. The fault itself is injected in the
  deterministic suite; these live checks establish normal-model compatibility.
- Both live profiles used an isolated 8192-token context; Flash used a 16384-token KV
  pool. Loaded HIP DLL paths were verified against each deployment. Existing engine,
  API, launcher, configuration and HIP DLL hashes were unchanged. Both services were
  stopped again after testing, matching their initial state. No long-context, GPU
  numerical or throughput claim is made from these bounded API checks.
- Machine-readable evidence summary: `docs/benchmarks/upstream-api-20261115.json`.

## Artifacts and deployment boundary

Local build, commands, source/file hashes, logs and model response records are retained
under `C:/Users/Jerry/Projects/qwenox-upstream-20261115/build/` (Git ignored).
The candidate executable is `build/qwenox-api.exe`; `source-identity.json` binds it to
the source files and compiler command. These are validation artifacts, not a production
release bundle.

This request updates the maintained source branch. Neither
`C:/Users/Jerry/llama.cpp/Qwenox` nor `C:/Users/Jerry/llama.cpp/Qwenox-qwen27` has its
deployed API/engine replaced. Real inference checks use separate loopback ports and
scratch working directories, with persistent checkpoint/token-cache writes disabled.
