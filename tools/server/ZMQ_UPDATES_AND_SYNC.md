# `tools/server`: ZMQ Transport + Upstream Sync Notes

Status as of commit `1015ca54c` (2026‑05‑18).

This document explains:

1. **What changed in `tools/server`** to host the new ZMQ transport alongside the existing cpp‑httplib HTTP listener.
2. **The new `llama-server` CLI flags** you'll use to invoke and consume it.
3. **How to consume the ZMQ transport from PHP** — specifically, how to reimplement `\Llama\Request` (which the `swoole_llama` extension provided in-process) as a drop‑in ZMQ client class.
4. **Which fork‑specific features the upstream sync affected**, and what got re‑ported afterward. Includes the multi‑label BERT classification path that upstream still doesn't ship.

The wrapper introduced here is `\Llama\Zmq\Request` (plus `\Llama\Zmq\Endpoint` and `\Llama\Zmq\Server`), in [`tools/server/php/`](php/). The ported acceptance test is [`tools/server/test_multi_model.php`](test_multi_model.php). Both have been run end‑to‑end on the same llama-server binary that this document describes — non‑streaming chat completion, single‑input embedding, multi‑input embedding, and error paths all pass; streaming has been independently verified against the same model.

---

## 1. What changed in `tools/server`

### Transport infrastructure

Until the ZMQ work, `tools/server` had a single transport (cpp‑httplib) baked into `server.cpp` plus a `server_http_context` that owned the listener and its route table. Phase 1–3 of the ZMQ work introduced a second transport without forking the route logic:

| File | Role |
|---|---|
| `server-transport.h` | **New.** Transport‑agnostic `server_http_req`, `server_http_res`, `server_http_res_ptr`, `server_transport_handler_t`. Both transports build the same `req`/`res` and accept the same handler signature. The `server_http_res` struct now also carries an `on_end` callback (transport-agnostic; cpp-httplib fires it from `on_complete`, ZMQ fires it after the empty-terminator frame). |
| `server-http.{h,cpp}` | cpp‑httplib transport — slimmed to consume the types from `server-transport.h`. `server_http_context::handler_t` is now an alias for `server_transport_handler_t`. |
| `server-zmq.{h,cpp}` | **New.** libzmq‑based transport. ROUTER frontend + DEALER backend + worker pool via `inproc://llama_backend`. Multipart streaming with `ZMQ_SNDMORE`, in‑flight `rid → atomic<bool>` registry for per‑request cancellation, strict-padding pre‑extract of `rid` from raw payload so parse errors still echo the correlation id, multi-bind (`ipc://` + `tcp://` simultaneously), HWM/LINGER tuning. |
| `server-context.{h,cpp}` | `server_routes::routes()` — **new.** Returns the canonical list of `{method, path, handler}` triples. `server.cpp` iterates this list once and registers every entry on every *enabled* transport, so HTTP and ZMQ share the same router table. |
| `server.cpp` | Holds `ctx_http` and `ctx_zmq` unconditionally; only calls `init()`/`start()` on the enabled ones. Router‑mode `/models/load` and `/models/unload` are registered on both transports too. HTTP‑only post-route hooks (GCP Vertex compat, CORS proxy, built-in tools) are registered after the data‑driven loop because they're HTTP‑shaped features that don't map onto the ZMQ envelope wire format. |

### Foundation audit fixes (Phase 1)

These ride along with the transport work because the new transport multiplies the cost of every existing landmine:

- **`std::terminate()` in `server_response::recv`** — replaced with `return nullptr` on shutdown and matching null checks in callers. Without this, any clean SIGTERM with a request in flight aborted the process.
- **Async‑signal‑safe signal handler** — POSIX self‑pipe + listener thread. The OS handler only writes one byte (`write()` is async‑signal‑safe); the real shutdown logic runs in normal context. Second signal force‑exits via `_exit()`.
- **Task ID width** — `server_task::id`, `id_target`, `id_parent` and the relevant `id_tasks` set widened to `uint64_t`. Old code used `int` and reused after wraparound; the "drop result on no waiter" invariant introduced for shutdown only holds with monotonic non‑reused ids.
- **SES1 / base64 dedup** — slot save/restore and session save/restore each had their own copy of the SES1 encode/decode and a hand‑rolled base64. Centralized in `server-common.{h,cpp}` with overflow guards (`SES1_MAX_TOKENS = 1 << 20`, INT64‑cast on `n_keep + n_discard` so the comparison can't wrap).
- **Subprocess control plane** — the router→child "ready" marker was substring matched, so any log line that contained `cmd_child_to_router:ready` could fake a ready signal. Now nonce‑tagged via the `LLAMA_SERVER_ROUTER_NONCE` env var; strict full‑line match.
- **`session_state::data`** is now `std::shared_ptr<const std::vector<uint8_t>>`. Restore takes an O(1) refcount bump under the lock, then dereferences outside. Sessions accounting is an `std::atomic<size_t>`.

### Session API

The C++ surface is unchanged from before the merge; what's new is the in‑process session map's behavior:

- **String‑keyed** — `std::map<std::string, session_state>`. Keys are validated (`128‑char cap, no '/', NUL, or control chars`).
- **RAM‑budget LRU eviction with pin** — `--sessions-max-bytes N`. `pinned` sessions are skipped by eviction; a one-shot warning fires if the budget is over and every entry is pinned.
- **Pin/unpin endpoints** — `POST /sessions/:id/pin` and `/unpin`. `GET /sessions` reports the `pinned` flag.
- **One‑shot `session` body block** — request bodies on the five completion handlers (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`, `/v1/audio/transcriptions`) accept:

  ```json
  "session": {
      "restore_key":     "user-xyz",     // optional: load this blob into the slot before inference
      "save_key_after":  "user-xyz",     // optional: capture the slot back to this key after inference
      "evict_after":     false           // optional: when save_key_after lands, also drop the warm entry
  }
  ```

  The `on_end` hook is what fires `save_key_after` for streaming responses (after the final wire byte).

---

## 2. CLI flags

Default transport behavior is unchanged from upstream: **HTTP enabled, ZMQ disabled**. You opt into ZMQ.

| Flag | Default | What it does |
|---|---|---|
| `--enable-http` / `--no-enable-http` | enabled | Bind the cpp‑httplib listener. Disable only if you want a ZMQ‑only server. |
| `--enable-zmq` / `--no-enable-zmq` | disabled | Bind a libzmq ROUTER. Required for any other `--zmq-*` flag. |
| `--zmq-bind ENDPOINT` | (none) | ZMQ bind endpoint; repeatable. Accepts any libzmq URL — `ipc:///tmp/foo.sock`, `tcp://127.0.0.1:5555`, `tcp://*:5555`. If `--enable-zmq` is set with no `--zmq-bind`, defaults to `ipc:///tmp/llama-server-<PID>.sock`. |
| `--zmq-workers N` | `0` (auto) | Worker thread count. `0` derives from `n_parallel + 2`. |
| `--zmq-hwm N` | `64` | `ZMQ_SNDHWM` / `ZMQ_RCVHWM` for the frontend/backend/worker sockets. |
| `--sessions-max-bytes N` | `0` (unbounded) | Hard cap on the bytes held in the warm `/sessions` map. LRU eviction skips pinned entries. |

`--enable-http=false` together with `--enable-zmq=false` is rejected at startup with a clear error.

### Starting llama-server

Single LLM, HTTP only (upstream default — unchanged):

```bash
build/bin/llama-server \
    -m model.gguf \
    --port 8080 -c 4096 -ngl 999
```

Single LLM, ZMQ only (drops the TCP listener, useful in trusted hosts):

```bash
build/bin/llama-server \
    --no-enable-http --enable-zmq \
    --zmq-bind ipc:///tmp/llama-llm.sock \
    -m model.gguf -c 4096 -ngl 999
```

Single LLM, both transports — the route table is registered on both, so HTTP clients and ZMQ clients see the same endpoints:

```bash
build/bin/llama-server \
    --enable-http --enable-zmq \
    --zmq-bind ipc:///tmp/llama-llm.sock --zmq-bind tcp://127.0.0.1:5555 \
    -m model.gguf --port 8080 -c 4096 -ngl 999
```

Embedding server on its own ZMQ endpoint (the common multi‑model pattern; you spawn one of these per model):

```bash
build/bin/llama-server \
    --no-enable-http --enable-zmq \
    --zmq-bind ipc:///tmp/llama-emb.sock \
    -m embed.gguf --embeddings --pooling mean -c 512 -ngl 0
```

### Stopping it

`SIGTERM` / `SIGINT` (Ctrl-C) initiate a clean shutdown:

1. The async‑signal‑safe signal handler writes one byte to the self-pipe.
2. The shutdown listener thread reads it and invokes the real shutdown handler in normal context.
3. Both transports' `stop()` are called (idempotent).
4. In‑flight requests see `should_stop` flip; their handlers unwind. Clients get either a structured error envelope or a completed response that landed before the cutoff.
5. `ctx_server.terminate()` drains both queues, `start_loop()` returns, joins.

A second signal force-exits via `_exit(1)`. No `std::terminate()` aborts on the way out.

---

## 3. Reimplementing `\Llama\Request` on top of server-zmq

The `tools/server-coro` extension (`swoole_llama.so`) embedded `llama-server`'s core directly into the calling PHP process and exposed `\Llama\Request` as a userspace class that dispatched into in‑process route handlers. That model is being retired in favor of an out‑of‑process llama-server reached over ZMQ.

This repo now ships three small pure‑PHP classes under `tools/server/php/Llama/Zmq/`:

| Class | Replaces | Responsibility |
|---|---|---|
| `Llama\Zmq\Request` | `\Llama\Request` | One request → one DEALER socket. Same constructor surface, same `isStream()` / `getStatusCode()` / `getData()` / `next()` / `cancel()` methods. |
| `Llama\Zmq\Endpoint` | (n/a) | Static `model alias → ZMQ endpoint` registry. `Request` resolves an endpoint by reading the `model` field from the request body and calling `Endpoint::lookup()`, matching how `\Llama\Request` resolved by `model` field via the in‑process registry. |
| `Llama\Zmq\Server` | `swoole_llama_load_model` / `swoole_llama_model_ready` / `swoole_llama_unload_model` | Fork+exec a `llama-server` child via `Swoole\Process`, wait for it to clear the readiness gate, supervise, and shut it down cleanly. Registers the model alias in `Endpoint` automatically. |

The autoloader is `tools/server/php/autoload.php` (PSR-4 for `Llama\Zmq\`). Three PHP extensions must be loaded: `swoole`, `swoole_zmq` (from `rapier_babylon/ext-zmq/`), and `zmq` (only used for its `ZMQ::` constants).

### Surface parity with `\Llama\Request`

The `\Llama\Request` class registered by `tools/server-coro/coro-extension.cpp` exposes:

```
new \Llama\Request($params)
$req->isStream(): bool
$req->getStatusCode(): int
$req->getData(): ?array     // non-stream: full decoded JSON; stream: first decoded chunk
$req->next(): ?array        // stream only; null when stream ends
$req->cancel(): void
```

`\Llama\Zmq\Request` mirrors every one of these with the same semantics. You can swap the FQN and the rest of your code is unchanged — including the `ServerCoroClient.php` in Odin (which constructs `new \Llama\Request([...])` and reads `getData()` / `getStatusCode()` / iterates via `LlamaRequestIterator`).

### Constructor parameters

```php
new \Llama\Zmq\Request([
    'method'   => 'POST',                      // required
    'path'     => '/v1/chat/completions',      // required
    'body'     => json_encode([                // body is a JSON string, as in \Llama\Request
        'model'    => 'qwen-base',
        'messages' => [...],
        'stream'   => true,
    ]),
    'headers'  => ['Content-Type' => ['application/json']],  // optional; array-valued
                                                             // entries take their first element
    // The endpoint can be passed explicitly, OR omitted — in which case the body's
    // 'model' field is parsed and resolved via Llama\Zmq\Endpoint::lookup($model).
    // The latter is the pattern Odin/ServerCoroClient.php already uses.
    'endpoint' => 'ipc:///tmp/llama-llm.sock', // optional
    'id'       => 'optional-correlation-id',   // optional; defaults to 16 random hex chars
]);
```

### Spawning model children

`Llama\Zmq\Server::spawn()` forks a `Swoole\Process`, execs `llama-server` with `--no-enable-http --enable-zmq --zmq-bind <endpoint>` prepended to the caller's argv, and registers `model alias → endpoint` in the `Endpoint` registry. `waitReady()` polls a readiness probe until the inference subsystem reports a loaded model. `shutdown()` sends `SIGTERM`, polls for exit, escalates to `SIGKILL` if needed, reaps the child, and cleans up the `ipc://` socket file.

> **Important**: `Swoole\Process::start()` calls `fork(2)` and must run **outside** any coroutine context. Spawn all model servers in main script scope first, then enter `Co\run()` for `waitReady()` / request / `shutdown()` logic. This is different from the `swoole_llama` extension's `swoole_llama_load_model()` which loaded models *into* the calling process and could therefore be invoked from inside a coroutine.

### Minimal usage

```php
require_once __DIR__ . '/php/autoload.php';
use Llama\Zmq\{Request, Server, Endpoint};

// 1. Spawn outside any coroutine
$llm = Server::spawn([
    'endpoint' => 'ipc:///tmp/llama-llm.sock',
    'model'    => 'qwen-base',
    'args'     => [
        '-m', '/path/to/qwen.gguf',
        '--ctx-size', '4096', '--n-gpu-layers', '-1',
    ],
]);

// 2. Inside Co\run, wait then drive
\Co\run(function () use ($llm) {
    $llm->waitReady();

    $req = new Request([
        'method' => 'POST',
        'path'   => '/v1/chat/completions',
        'body'   => json_encode([
            'model'    => 'qwen-base',
            'messages' => [['role' => 'user', 'content' => 'Hi']],
            'stream'   => false,
        ]),
    ]);

    $data = $req->getData();
    echo $data['choices'][0]['message']['content'], "\n";

    $llm->shutdown();
});
```

For streaming:

```php
$req = new Request([..., 'body' => json_encode([..., 'stream' => true])]);
$content = '';
$first = $req->getData();   // first decoded chunk (typically {delta: {role: assistant}})
if ($first && isset($first['choices'][0]['delta']['content'])) {
    $content .= $first['choices'][0]['delta']['content'];
}
while (($c = $req->next()) !== null) {
    if (isset($c['choices'][0]['delta']['content'])) {
        $content .= $c['choices'][0]['delta']['content'];
    }
}
```

`cancel()` sends a one‑way `{"method": "CANCEL", "id": $rid}` envelope on the same DEALER socket; the worker on the C++ side looks the rid up in its in‑flight registry and flips the `should_stop` atomic, and the in‑flight handler unwinds.

### Acceptance test

[`tools/server/test_multi_model.php`](test_multi_model.php) is the port of `tools/server-coro/test_multi_model.php`. Same 17‑step shape; spawns two children (LLM + embedding), exercises chat completions, error paths, single and multi‑input embeddings, and clean shutdown. It uses what's on disk for models:

- LLM: `/home/jason-dev/rapier_babylon/Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf`
- Embedding: `/home/jason-dev/rapier_babylon/embeddinggemma-300m-qat-Q4_0.gguf`

Run with `php tools/server/test_multi_model.php`. Total runtime ~40 seconds on this machine (4B LLM on CUDA, 300M embedding on CPU). All 17 steps pass.

### Wire protocol (for reference — the wrapper handles this for you)

Request envelope (single DEALER frame, JSON):

```json
{ "method": "POST", "path": "/v1/chat/completions",
  "body":   "<JSON-encoded request body>",
  "headers": { "Content-Type": "application/json" },
  "id":     "<correlation-id>" }
```

Non‑stream reply (single frame after ROUTER strips client id):

```json
{ "status": 200, "content_type": "application/json; charset=utf-8",
  "headers": {}, "rid": "<correlation-id>",
  "stream": false, "data": "<raw response body as a string>" }
```

Stream reply (single multipart message):

```
frame 1 (SNDMORE):  { "status": 200, ..., "stream": true }
frame 2 (SNDMORE):  raw chunk bytes (typically "data: {...}\n\n")
...
frame N-1 (SNDMORE): final chunk
frame N (last):     empty terminator
```

Receiver reads frame‑by‑frame, checks `ZMQ_RCVMORE` between recvs; the empty terminator is the unambiguous end signal.

Cancel envelope (one-way, same DEALER socket):

```json
{ "method": "CANCEL", "id": "<correlation-id>" }
```

The server's worker looks the rid up in `in_flight`, flips the atomic, returns no reply.

---

## 4. Upstream sync impact on fork features

The merge in commit `e197823ee` brought in **1133 upstream commits** (~one month of upstream activity). Most of it landed cleanly; the parts that conflicted with fork-specific code are documented here.

### Architecture file restructure (the biggest one)

Upstream moved `load_hparams` and `load_tensors` out of `src/llama-model.cpp`'s monolithic switch statement into per‑arch files under `src/models/<arch>.cpp` (PR #22004). The fork's inline DistilBERT/ModernBERT classifier work, which lived as added cases in that switch, was overwritten by the merge.

Re‑ported in commit `a8e021ef0` ("re‑port DistilBERT/ModernBERT classifier support"):

- `src/llama-arch.cpp`: `LLM_ARCH_DISTILBERT` re‑registered in `LLM_ARCH_NAMES` so `convert_hf_to_gguf` model files with `general.architecture = "distilbert"` resolve to a real arch.
- `src/models/distilbert.cpp`: **new.** `llama_model_distilbert` is a small subclass of `llama_model_bert` that overrides only the layer-count → model-type label (LLM_TYPE_70M for 6-layer base). Tensor loading is inherited from BERT; `bert.cpp`'s classifier-tensor branch was extended to fire for `LLM_ARCH_DISTILBERT` too.
- `src/llama-graph.cpp` (`build_pooling`): the activation switch was preserved through the merge but had drifted — replaced `ggml_gelu` (tanh approximation) with `ggml_gelu_erf` (exact, matches `nn.GELU`) for ModernBERT, so classifier output now matches HuggingFace bit‑for‑bit.
- `src/models/modern-bert.cpp` already loads `cls_norm` and calls `set_swa_pattern(swa_period, true)` (dense-first); these survived the merge cleanly.

Also re-ported in `1015ca54c` ("drop ModernBERT decoder.* MLM-head tensors"): `conversion/bert.py`'s `ModernBertModel.filter_tensors` lost its `decoder.*` filter during the upstream restructure; without it, MLM‑head tensors from a fine-tuned-for-MLM ModernBERT slip through to `map_tensor_name` and fail the conversion. Re-added; mirrors `NeoBert.filter_tensors`.

### Conversion script split

Upstream split the monolithic `convert_hf_to_gguf.py` into per‑arch modules under `conversion/<family>.py`. Two structural changes worth knowing about:

- `convert_hf_to_gguf.py` (top level) is now a 282-line dispatcher; the BERT family classes live in `conversion/bert.py`.
- `modify_tensors` was split into two methods on each class:
  - `filter_tensors` — classmethod, handles prefix stripping and dropping unwanted tensors (returning `None` is the new "drop" signal, instead of returning `[]`).
  - `modify_tensors` — instance method, just handles renames now.

The old monolithic file is still in-tree at `convert_hf_to_gguf_bert_mod.py` for reference, but no longer wired into anything.

Drift checked and confirmed clean for all BERT-family classes (`BertModel`, `DistilBertModel`, `RobertaModel`, `NomicBertModel`, `NeoBert`, `XLMRobertaModel`, `JinaBertV2Model`, `ModernBertModel`). The only behavioral gap was the ModernBERT `decoder.*` filter, now fixed.

### Multi-label classification — **fork-only, still**

This is the biggest functional difference between this fork and upstream and warrants explicit documentation.

The fork ships a `/classify` and `/v1/classify` endpoint that returns **every label with its raw logit score**, sorted by score descending — i.e., true multi‑label output:

```bash
curl -X POST http://localhost:8080/v1/classify \
    -H 'Content-Type: application/json' \
    -d '{"inputs": "I am happy"}'
```

```json
{
  "model":  "distilbert-zeroshot-f32",
  "object": "list",
  "usage":  {"prompt_tokens": 4, "total_tokens": 4},
  "predictions": [
    {"label": "joy",     "score":  3.42},
    {"label": "sadness", "score": -1.18},
    {"label": "anger",   "score": -2.07},
    ...
  ]
}
```

The handler is `post_classify` in `server-context.cpp:4642`; the response shape comes from `server_task_result_classify::to_json` (`server-task.cpp:1685`) which iterates over **all** `predictions` and emits the full sorted array. `format_response_classify` (`server-common.cpp:1248`) wraps it as an OpenAI-style list response.

Upstream landed a reranker that only ever returns a single score per input (the rerank-relevance score, not a class label) and has no notion of per-label outputs from a `BertForSequenceClassification` head. To get a similar shape on upstream you would have to bolt your own postprocessing on top of `/v1/embeddings` and reimplement the softmax/argmax loop in client code — there is no upstream endpoint that emits `{label, score}` pairs.

This functionality requires:

1. **Arch with classifier outputs** — a `BertForSequenceClassification` or `DistilBertForSequenceClassification` (or the multi-label ModernBERT/RoBERTa variants) converted with the fork's `conversion/bert.py`. The converter's `cls_out_labels` plumbing reads `id2label` from the HF config and writes them into the GGUF; without it the model has classifier weights but no labels to attach to scores.
2. **`--reranking` at server start** — this is what flips `pooling_type = LLAMA_POOLING_TYPE_RANK`, which is what `post_classify`'s guard checks. (Yes the flag is named for rerankers; classification reuses the rerank pooling path because both need per-token logits routed through a classifier head.)
3. **DistilBERT ReLU activation** — fork-only. `build_pooling` in `src/llama-graph.cpp` dispatches the classifier-head activation on `model.arch`: DistilBERT uses `ggml_relu`, ModernBERT uses `ggml_gelu_erf`, everything else uses `ggml_tanh`. Upstream applies `tanh` unconditionally, which silently degrades DistilBERT accuracy (~85% top-label agreement with HF reference vs. ~100% with the fork's ReLU).

The `predictions` array is what makes this multi-label: a downstream PHP/Odin consumer can take the full array, apply sigmoid + threshold for multi-label classification, or take argmax for single-label. Either way the model's raw signal is preserved end-to-end.

### Other fork-specific code that survived the merge

The merge resolution preserved these (they were touched by upstream but not in conflicting ways):

- **The five Caelus task result types** in `server-task.h` / `server-task.cpp`: `server_task_result_classify`, `server_task_result_seq_state_get`, `server_task_result_seq_state_set`, `server_task_result_slot_tokens`, `server_task_result_context_shift`. All required by the session-aware request lifecycle, none of which upstream has.
- **`LLM_TENSOR_CLS_NORM`** in `src/llama-arch.{h,cpp}` and the matching `cls_norm` member on `llama_model`. The classifier-head LayerNorm for ModernBERT.
- **`server_routes::routes()`** as the canonical handler table — survived because it's the bridge between the existing imperative HTTP registration (router-mode handler swaps still happen on `server_routes` before `routes()` is materialized) and the new transport-agnostic dispatch.

### Files to watch on the next sync

These are the high-conflict-likelihood files for the next upstream merge. List is empirical, based on what conflicted in `e197823ee`:

- `tools/server/server-context.{h,cpp}` — large, actively developed upstream, has dense fork additions (session API, classify handler, slot tokens, context shift, route table).
- `tools/server/server-task.{h,cpp}` — every new task result type upstream adds risks collision with the Caelus ones.
- `tools/server/server.cpp` — main() is in flux upstream (new params, new conditional features); merge conflicts focus around the route registration block and clean_up().
- `common/common.h` and `common/arg.cpp` — every new CLI flag upstream lands here too; expect overlapping changes around the embeddings/sessions/transport fields.
- `gguf-py/gguf/constants.py` and `conversion/bert.py` — upstream restructured both during this sync; future smaller changes are likely to keep landing here.
- `src/llama-graph.cpp` (`build_pooling`) and `src/llama-arch.cpp` (`LLM_ARCH_NAMES`) — the activation-switch and arch-name table both need attention if upstream adds new pooling types or BERT variants.

### What was *not* re-ported (intentional)

- The `tools/server-coro/` extension's in-process model registry. The whole point of this sync is to retire it in favor of out-of-process llama-server children + ZMQ. The directory is gitignored for privacy (committed in `f31b88f76`).
- The fork's old `register_route()` direct registration path. Upstream's `register_gcp_compat()` + the data-driven `routes()` table cover the same surface cleanly.

---

## Quick reference: starting and stopping two model servers for testing

You need two endpoints (one LLM, one embedding) and they spawn as separate processes:

```bash
# Terminal 1 — LLM over ZMQ on ipc://
build/bin/llama-server \
    --no-enable-http --enable-zmq \
    --zmq-bind ipc:///tmp/llama-llm.sock \
    -m /path/to/llm.gguf -c 4096 -ngl 999 &
LLM_PID=$!

# Terminal 2 — embedding over a different ZMQ endpoint
build/bin/llama-server \
    --no-enable-http --enable-zmq \
    --zmq-bind ipc:///tmp/llama-emb.sock \
    -m /path/to/embed.gguf --embeddings --pooling mean -c 512 -ngl 0 &
EMB_PID=$!

# Clients connect to either endpoint via DEALER, no model registry needed
# (the model name in the request body is informational; the endpoint is the
# transport identity)

# Graceful stop — the self-pipe signal handler drains in-flight requests
kill -TERM $LLM_PID $EMB_PID
```

From PHP, the equivalent driver is `tools/server/test_multi_model.php`; from C++/CLI directly, just `kill -TERM <pid>` to each child.
