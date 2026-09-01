# Every setting, and why it has that value

The full command the numbers in [RESULTS.md](RESULTS.md) were produced with:

```bash
docker run -d --name dsv4 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v /path/to/DeepSeek-V4-Flash-0731:/model \
  -v $R/config/speculative.py:/vllm/vllm/config/speculative.py:ro \
  -v $R/v1/worker/gpu/pp_utils.py:/vllm/vllm/v1/worker/gpu/pp_utils.py:ro \
  -v $R/v1/worker/gpu/model_runner.py:/vllm/vllm/v1/worker/gpu/model_runner.py:ro \
  -v $R/v1/worker/gpu/spec_decode/dspark/utils.py:/vllm/vllm/v1/worker/gpu/spec_decode/dspark/utils.py:ro \
  -v $R/model_executor/layers/sparse_attn_indexer.py:/vllm/vllm/model_executor/layers/sparse_attn_indexer.py:ro \
  --shm-size=16g -p 8098:8000 \
  dsv4-a100:devel vllm serve /model --served-model-name dsv4s \
  --pipeline-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 32768 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 8 \
  --trust-remote-code \
  --no-enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

`$R` is the `vllm/` directory of your patched checkout. The image installs vLLM with
`pip install -e .`, so `/vllm/vllm/...` is live code — bind-mounting the five patched
files applies them with **no rebuild**.

---

## The settings that matter most

### `--pipeline-parallel-size 4`  ← not tensor parallel
The single highest-impact choice. On these cards PP beats TP by **6.6× on prefill**
(5,321 vs 801 tok/s at 77k context) and roughly 2× on aggregate decode.

Why: tensor parallel performs **2 all-reduces per layer × 43 layers = 86 collectives**
per forward, each moving `tokens × hidden` bytes. Double the tokens and you double both
compute and communication, so a communication-bound setup stays communication-bound at
*every* sequence length — which is exactly why TP prefill measures flat at ~800 tok/s
from 1.5k to 77k tokens. Pipeline parallel moves the same payload only **3 times** (one
hand-off per stage boundary) and overlaps it with compute: **~28× less data on the wire**.

The CMP 170HX is PCIe **Gen2 x4 with no P2P** (~1.0 GB/s measured bus bandwidth), which
is close to the worst case for TP. On a machine with NVLink the trade-off would look
completely different.

### `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`
- **`dspark`, not `mtp`.** DeepSeek-V4-Flash-0731 replaced the single MTP layer with a
  3-layer DSpark stack (`mtp.{0,1,2}` + `markov_head` + `confidence_head`).
  `method:"mtp"` KeyErrors on `mtp_block.main_norm`.
- **5, and 5 exactly.** vLLM enforces `num_speculative_tokens >= dspark_block_size`
  (5 for this checkpoint) — below that the block/Markov machinery gets an unsupported
  layout and produces *garbled output*, not merely lower acceptance. Above is worse in
  practice: **7 measured 60.3 tok/s vs 5's 98.1**, because acceptance never extends past
  ~3 tokens so the extra drafts are pure waste.
- Worth **+1.93×** decode over the same config without it, and unlike on TP it keeps
  winning under load (see RESULTS).

### `--gpu-memory-utilization 0.85`
Not higher. Raising it takes headroom away from activations and CUDA-graph capture; at
**0.90 with the DSpark draft resident, capture OOMs**. The KV pool you would gain is
memory you cannot spend — at the ~128k usable ceiling a single context costs only about
6 GiB, and the pool at 0.85 already reports 1.8M tokens at `max-model-len 131072`.

### `--max-num-batched-tokens 2048`
Do **not** lower it: without it, activation memory during profiling eats the KV pool and
the engine dies with "No available memory for the cache blocks".

Raising it is pointless — measured twice, on two different stacks. 2048 → 8192 moved
prefill by ~4% (890 → 924 tok/s) and **halved the KV pool** (58,538 → 29,867 tokens).
Prefill is not chunk-size bound.

### `--kv-cache-dtype fp8`
Effectively required. DeepSeek-V4 asserts `fp8_ds_mla layout only supports fp8 kv-cache`.
FP8 KV does work correctly on sm_80 — the conversion is software; only FP8 *math* needs
sm_89+.

### `--no-enable-flashinfer-autotune`
FlashInfer's JIT runs at engine init and is a **hard requirement** (removing the package
gives `ModuleNotFoundError`), so the container must be able to compile it. See
[docker/Dockerfile.devel](docker/Dockerfile.devel) for why the base image needs a real
CUDA toolkit rather than pip CUDA wheels.

### `--tokenizer-mode deepseek_v4`
From the working launch command in
[vllm#50576](https://github.com/vllm-project/vllm/issues/50576).

### `--max-model-len` — ★ set it to what you need, up to the model maximum
**Previous versions of this file said a larger value gives you LESS usable context. That was a
symptom of the [logits-buffer bug](RESULTS.md#-context-ceiling--solved) and is withdrawn.**
With `DSV4_LOGITS_ROW_CHUNK` set, the full 1,048,576 works:

| `--max-model-len` | highest verified prompt | `DSV4_LOGITS_ROW_CHUNK` |
|---|---|---|
| 393,216 | 388,505 (one-shot) | 256 |
| **1,048,576** (model max) | **1,047,736** (one-shot) | **128** |
| **1,048,576** | **1,002,852** (**405-turn conversation**) | **64** |

⚠️ **Those first two rows are one-shot prefills.** A long *conversation* on `128` dies at
~718–733k with a CUDA illegal memory access — see
[accumulated vs one-shot](RESULTS.md#accumulated-conversation--one-shot-prefill). Chunk size costs
almost nothing (TTFT 7.48 s at `64` vs 7.08 s at `128`, measured at 750k), so **if in doubt use
64**.

The only reason not to set the maximum is **time**, not capacity: TTFT at 1M is **9.2 minutes**
and decode drops to ~40 tok/s. Pick the profile that matches your workload —

- everyday, ≤388k, one-shot documents → `--max-model-len 393216`, `DSV4_LOGITS_ROW_CHUNK=256`
- full 1M, one-shot documents → `--max-model-len 1048576`, `DSV4_LOGITS_ROW_CHUNK=128`
- ★ **long conversations / agents (any max-model-len) → `DSV4_LOGITS_ROW_CHUNK=64`**

⚠️ **None of these settings change retrieval accuracy**, which degrades with depth regardless
(~100% at 150k → ~30% at 900k). That is architectural, not a tuning problem. See
[RESULTS](RESULTS.md#retrieval-accuracy-vs-depth--the-window-is-reachable-not-uniformly-usable).

### `--reasoning-parser deepseek_v4` — ★ set it even if you never enable thinking

`thinking=False` is the default on 0731, so most people never see this. But the moment thinking is
on, the `<think>` delimiters are **special tokens** and get stripped on decode — so **without this
flag the reasoning text arrives inside `content` with nothing marking it as reasoning**, and replies
literally begin *"We need answer classic. Need be careful. User asks…"*. With the flag it lands in
its own field and `content` stays a clean answer. The flag is inert when thinking is off, so there
is no reason not to set it.

⚠️ On this build the returned field is **`reasoning`**, not `reasoning_content`. A client reading
only the latter sees zero thinking and cannot distinguish a thinking run from a non-thinking one.

**Enabling thinking** (worth it — [+19 points of long-context recall](RESULTS.md#thinking-recovers-a-lot-of-the-lost-long-context-recall--the-effort-level-does-not)):
per request via `"chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}`, or
server-side with `--default-chat-template-kwargs`. **Use `high`, not `max`** — `max` thinks 2.7×
harder for identical recall. A top-level `reasoning_effort` body param is the ambiguous path; it
changes behaviour but `/tokenize` cannot see it.

★ **Set `reasoning_effort` explicitly on every request, not just the first one — agentic clients
especially.** `0008` fixes the *default* (omitted now means thinking=True instead of silently
falling back to chat mode), but that's a one-time fix for a single request's prompt rendering, not
a guarantee across a long session. A session that never sends an explicit `reasoning_effort` can
still drift over many turns into the self-reinforcing "Let me..." phrase-lock loop `0008`/`0016`-
`0020` guard against — live testing on this repo's own served image found that pinning
`chat_template_kwargs: {"reasoning_effort": "high"}` on **every** request measurably reduces how
often loops occur in the first place, not just how quickly the repetition tripwires (`0017`,
`0020`) catch one after the fact. This matches a finding independently reported by
[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport): *"sessions that run without the
effort prefix gradually stop thinking and can enter self-reinforcing reasoning loops."* Treat the
tripwires as the safety net, not the fix — the actual fix is the client sending this on every
turn. (In opencode: set it under the model's `options.chat_template_kwargs` in `opencode.json`,
not just once in a system prompt.)

### `--override-generation-config '{"top_p":0.95}'`
Official 0731 guidance for agentic/tool-calling use is `temperature=1.0, top_p=0.95` (vs
`top_p=1.0` for general chat). Our launch scripts never set `top_p` at all before
2026-09-01, so the server silently fell back to the checkpoint's `generation_config.json`
default of `1.0` for every request — including agentic opencode sessions. Fixed via
`--override-generation-config`, which applies server-side regardless of what the client
sends.

⚠️ **Tested 2026-09-01, alongside `--reasoning-parser deepseek_v4` (below): did NOT fix
the >100k-context repetition-loop degeneration.** Loops still occurred. Root cause is
believed to sit deeper than sampling — see
[ggml-org/llama.cpp#26694](https://github.com/ggml-org/llama.cpp/issues/26694), an
unresolved upstream report of the identical shape (long agentic chat, local serving only,
temperature ruled out, suspected chat-template/stop-token boundary loss). Keep this flag
set regardless — it matches official guidance and is very unlikely to hurt — but do not
expect it alone to solve the loop problem. See `DSV4_RECOVERY_MODE` below for the
scheduler-level mitigation this project actually relies on.

### `--reasoning-parser deepseek_v4`
See the ★ entry lower in this file for what this flag does and why it matters whenever
thinking is enabled. **Gap found 2026-09-01:** this had been documented here as
load-bearing since before the DSML/repetition patch series, but was missing from both
`launch/run-pp-dspark.sh` and the (until then untracked) podman launcher actually used in
production. Fixed in `launch/run-pp-dspark-podman.sh`. If you're running from an older
checkout or a hand-rolled launch command, check this flag is actually present — docs
saying to set something is not the same as a script setting it.

### `--max-num-seqs 8`
Raise for serving. 128 was used for the concurrency sweeps and behaves well; DSpark keeps
winning all the way to 64 concurrent requests on PP.

### `--block-size 256`, `--trust-remote-code`
Standard for this model.

---

## Environment variables

| var | value | why |
|---|---|---|
| `NVIDIA_VISIBLE_DEVICES` | `0,1,2,3` | Four cards. **Three also works** (e.g. `1,2,3`) with `VLLM_PP_LAYER_PARTITION=15,15,13` — see [RESULTS](RESULTS.md#three-or-four-cards). An earlier version of this table said three does not work; that was wrong. |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Required; fork deadlocks with CUDA already initialised. |
| `HF_HUB_OFFLINE` | `1` | Local weights only; avoids a hub call on every start. |
| `CUDA_HOME` | set in image | The devel image sets `/usr/local/cuda`. |
| **`DSV4_LOGITS_ROW_CHUNK`** | **`64`** for conversations; `256`/`128` for one-shot | ★ **The context-ceiling fix** ([patch 0006](patches/0006-logits-row-chunk.patch)). Row-chunks the sparse indexer's `[M, N]` float32 logits transient. `0` = original upstream path, which dies at ~134k. **One-shot prefill:** `256` reaches ~957,600, `128` reaches 1,047,736. **Accumulating conversation:** `128` dies at ~718–733k (reproduced twice); **`64` reached 1,002,852 over 405 turns.** Costs almost nothing (TTFT 7.48 s vs 7.08 s at 750k). Affects only whether it crashes — **not** retrieval accuracy. |
| **`VLLM_PP_LAYER_PARTITION`** | `12,12,12,7` (4 cards) · `15,15,13` (3 cards) | Rebalances the 43 layers off pipeline rank 3, which uniquely carries `lm_head` **and** the DSpark drafter. **Does not affect the context ceiling** — but it removes an 8.7 GiB rank imbalance and grows the KV pool **~85%** (798,660 → 1,476,563 at `max-model-len 163840`). Must have one entry per pipeline rank, summing to 43. **On 3 cards this is required, not optional** — the default `[15,14,14]` fails during the Marlin FP4 expert repack because the last rank also carries `lm_head` and the DSpark drafter. |
| **`DSV4_RECOVERY_MODE`** | `nudge` (default) or `reopen` | Repetition-loop recovery strategy ([patch 0021](patches/0021-repetition-recovery-nudge-splice.patch) / [0022](patches/0022-repetition-recovery-reopen-turn-mode.patch)). `nudge` splices freeform steering text mid-turn; `reopen` splices the real `<｜end▁of▁sentence｜><｜User｜>go<｜Assistant｜>` turn-boundary sequence instead. Live-tested past ~100k context: `nudge` fires (confirmed in logs) but the model can keep mutating into new short repeats and exhaust the retry budget without recovering. `reopen` is the current hypothesis for why — untested pending a real run. `DSV4_NUDGE=0` disables recovery entirely regardless of mode. |

⚠️ **Do not bother with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — it is a hard
failure at model load on these cards: `expandable_segments: memory mapping failed with OOM on
device 3 while trying to map 20971520 bytes (free: 28626452480)`, i.e. it cannot map 20 MB
with 28.6 GiB free. CUDA VMM appears broken on GA100 CMP parts.

`--shm-size=16g` on the container — the default 64 MB is not enough for multiprocess workers.

---

## Benchmarking-only flags

Not for production, but required to reproduce the numbers:

- **`--no-enable-prefix-caching`** — without it, repeated benchmark prompts hit the prefix
  cache and prefill numbers become fiction (a 100k-token prompt appearing to prefill in
  0.5 s was how this was caught).
- **`ignore_eos: true`** in the request — otherwise the speculative and non-speculative
  configs generate *different numbers of tokens* and the comparison is meaningless.
- **`stream_options: {include_usage: true}`** — under speculative decoding a single SSE
  chunk carries several tokens (roughly the acceptance length), so a harness that counts
  chunks under-reports by that factor. Rate against the server's own `completion_tokens`.
- **Discard the first request after boot** — it carries Triton JIT compilation and reads
  roughly 4× low.

---

## Things deliberately NOT set

| flag | why not |
|---|---|
| `--enforce-eager` | **Never.** 8–10 tok/s, i.e. worse than no speculative decoding at all. CUDA graphs are worth ~12×. |
| `--tensor-parallel-size` | PP wins on this hardware; see above. TP3 is also arithmetically impossible (64 heads, 256 experts don't divide by 3). |
| `--quantization` | The checkpoint's format is auto-detected. Forcing it breaks MoE scale loading. |
| `VLLM_SM86_SPLITK=0` | Helped ~8% on the *older* overlay stack at low TP. Untested on this branch; the kernel it targets may not be in this path. |
