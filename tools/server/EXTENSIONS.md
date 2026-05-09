# llama-server API Extensions

This document covers the additional API endpoints and slot management features added to `llama-server` beyond the standard OpenAI-compatible API surface.

---

## Table of Contents

- [Classification API](#classification-api)
- [Slot State Management](#slot-state-management)
  - [Save State](#save-state)
  - [Restore State](#restore-state)
  - [Get Tokens](#get-tokens)
  - [Context Shift](#context-shift)
- [Slot Info](#slot-info)
- [Server-Side Session Storage](#server-side-session-storage)
  - [Save Session](#save-session)
  - [Restore Session](#restore-session)
  - [Remove Session](#remove-session)
  - [List Sessions](#list-sessions)

---

## Classification API

Adds text classification support for BERT-based classifier models (e.g., `BertForSequenceClassification`, `ModernBERT`, `DistilBERT`). Returns per-label scores (raw logits) for each input. The endpoint follows the TEI (Text Embeddings Inference) input format.

### Requirements

- Server must be started with `--reranking` flag
- Model must be a classifier with multiple output classes (`n_cls_out > 1`)
- Model must use `LLAMA_POOLING_TYPE_RANK`

### Endpoints

| Method | Path |
|--------|------|
| POST | `/classify` |
| POST | `/v1/classify` |

### Request Body

| Field | Type | Description |
|-------|------|-------------|
| `inputs` | `string` or `string[]` | Text(s) to classify |

### Response

```json
{
  "model": "model-name",
  "object": "list",
  "usage": {
    "prompt_tokens": 42,
    "total_tokens": 42
  },
  "predictions": [
    { "label": "POSITIVE", "score": 2.345 },
    { "label": "NEGATIVE", "score": -1.678 }
  ]
}
```

Predictions are sorted by score descending. Scores are raw classifier logits (not softmax-normalized). For batch inputs, `predictions` is an array of arrays.

### curl Examples

**Single input:**

```bash
curl -s http://localhost:8080/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "I love this product, it works great!"
  }' | jq .
```

**Batch input:**

```bash
curl -s http://localhost:8080/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      "I love this product!",
      "This is terrible, worst purchase ever.",
      "It is okay, nothing special."
    ]
  }' | jq .
```

---

## Slot State Management

These endpoints extend the existing `POST /slots/:id_slot` action dispatch with new in-memory operations for KV cache state management. Unlike the existing file-based `save`/`restore` actions (which require `--slot-save-path`), these operate on binary blobs and require only the `--slots` flag.

The binary format used is **SES1**: a simple header followed by the raw KV cache data.

```
SES1 Format:
┌──────────────────┬───────────────┬─────────────────────────┬──────────────┐
│ magic: "SES1"    │ n_tokens: u32 │ tokens: n × llama_token │ kv_data: ... │
│ (4 bytes)        │ (4 bytes)     │ (n × 4 bytes)           │ (variable)   │
└──────────────────┴───────────────┴─────────────────────────┴──────────────┘
```

The blob includes both the prompt token IDs and the serialized KV cache, so a single blob is sufficient to fully restore a slot's state.

### Save State

Serializes a slot's KV cache and prompt tokens into a binary blob. Supports two response formats via content negotiation.

| Method | Path | Query |
|--------|------|-------|
| POST | `/slots/:id_slot` | `?action=save-state` |

**Requires:** `--slots`

#### Response Formats

| Accept Header | Response |
|---------------|----------|
| `application/octet-stream` | Raw binary SES1 blob |
| *(default)* | JSON with base64-encoded `state` field |

#### JSON Response

```json
{
  "id_slot": 0,
  "n_tokens": 156,
  "n_bytes": 4194560,
  "t_ms": 12.5,
  "state": "U0VTMQAAAJ..."
}
```

#### curl Examples

**Save as JSON (base64):**

```bash
curl -s -X POST "http://localhost:8080/slots/0?action=save-state" | jq .
```

**Save as binary file:**

```bash
curl -s -X POST "http://localhost:8080/slots/0?action=save-state" \
  -H "Accept: application/octet-stream" \
  -o slot0.ses1
```

---

### Restore State

Restores a slot's KV cache and prompt tracking from a previously saved SES1 blob. Accepts either raw binary or JSON with base64.

| Method | Path | Query |
|--------|------|-------|
| POST | `/slots/:id_slot` | `?action=restore-state` |

**Requires:** `--slots`

#### Request Formats

| Content-Type | Body |
|--------------|------|
| `application/octet-stream` | Raw binary SES1 blob |
| `application/json` | `{"state": "<base64-encoded SES1 blob>"}` |

#### Response

```json
{
  "id_slot": 0,
  "n_bytes_read": 4194560,
  "success": true,
  "t_ms": 8.3
}
```

#### curl Examples

**Restore from binary file:**

```bash
curl -s -X POST "http://localhost:8080/slots/0?action=restore-state" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @slot0.ses1 | jq .
```

**Restore from JSON (base64):**

```bash
# First save state to a variable
STATE=$(curl -s -X POST "http://localhost:8080/slots/0?action=save-state" | jq -r '.state')

# Then restore to a different slot
curl -s -X POST "http://localhost:8080/slots/1?action=restore-state" \
  -H "Content-Type: application/json" \
  -d "{\"state\": \"$STATE\"}" | jq .
```

---

### Get Tokens

Returns the token IDs currently in a slot's prompt cache. Lightweight operation — does not serialize the KV cache.

| Method | Path | Query |
|--------|------|-------|
| POST | `/slots/:id_slot` | `?action=tokens` |

**Requires:** `--slots`

#### Response

```json
{
  "id_slot": 0,
  "n_tokens": 156,
  "tokens": [1, 9707, 374, 264, ...],
  "n_prompt_tokens_processed": 156
}
```

#### curl Example

```bash
curl -s -X POST "http://localhost:8080/slots/0?action=tokens" | jq .
```

---

### Context Shift

Manually removes a range of tokens from the middle of a slot's KV cache. This is the same operation the server performs automatically when hitting the context limit, but exposed for manual control.

Removes `n_discard` tokens starting at position `n_keep`, then shifts all subsequent tokens left to fill the gap.

```
Before: [kept tokens (n_keep)] [discarded (n_discard)] [remaining tokens...]
After:  [kept tokens (n_keep)] [remaining tokens...]
```

| Method | Path | Query |
|--------|------|-------|
| POST | `/slots/:id_slot` | `?action=context-shift` |

**Requires:** `--slots`

#### Request Body

| Field | Type | Description |
|-------|------|-------------|
| `n_keep` | `int` | Number of tokens to keep from the beginning |
| `n_discard` | `int` | Number of tokens to remove after `n_keep` (must be > 0) |

#### Constraints

- `n_keep >= 0`
- `n_discard > 0`
- `n_keep + n_discard <= n_tokens` (current token count in slot)

#### Response

```json
{
  "success": true,
  "new_n_tokens": 84
}
```

#### curl Example

```bash
# Keep the first 10 tokens (system prompt), discard the next 50 (oldest conversation turns)
curl -s -X POST "http://localhost:8080/slots/0?action=context-shift" \
  -H "Content-Type: application/json" \
  -d '{
    "n_keep": 10,
    "n_discard": 50
  }' | jq .
```

---

## Slot Info

Returns detailed information about a slot's current token state, including message boundary detection. Boundaries are detected by scanning the token array for the model's end-of-turn (EOT) token (e.g., `<|im_end|>`, `<|eot_id|>` — auto-detected by llama.cpp at model load).

This is useful for:
- Determining how much context is used
- Identifying individual message boundaries for selective pruning
- Making informed decisions before calling `context-shift`

| Method | Path |
|--------|------|
| GET | `/v1/slots/:id_slot/info` |

**Requires:** `--slots`

### Response

```json
{
  "n_tokens": 234,
  "boundary_eot": 151645,
  "n_messages": 5,
  "messages": [
    { "index": 0, "start": 0,   "end": 13  },
    { "index": 1, "start": 14,  "end": 31  },
    { "index": 2, "start": 32,  "end": 89  },
    { "index": 3, "start": 90,  "end": 112 },
    { "index": 4, "start": 113, "end": 233 }
  ]
}
```

| Field | Description |
|-------|-------------|
| `n_tokens` | Total tokens in the slot |
| `boundary_eot` | The EOT token ID used for scanning |
| `n_messages` | Number of detected messages |
| `messages[].start` | First token index of this message |
| `messages[].end` | Last token index of this message (inclusive of EOT, or last token if trailing) |

### curl Example

```bash
curl -s http://localhost:8080/v1/slots/0/info | jq .
```

**Practical workflow — inspect then shift:**

```bash
# 1. Check slot state
INFO=$(curl -s http://localhost:8080/v1/slots/0/info)
echo "$INFO" | jq '{n_tokens, n_messages}'

# 2. Find where the 2nd message ends (to discard messages 1-2, keeping system prompt at index 0)
MSG2_END=$(echo "$INFO" | jq '.messages[2].end')

# 3. Keep system prompt (message 0 ends at index 13), discard up through message 2
curl -s -X POST "http://localhost:8080/slots/0?action=context-shift" \
  -H "Content-Type: application/json" \
  -d "{\"n_keep\": 14, \"n_discard\": $((MSG2_END - 13))}" | jq .

# 4. Verify
curl -s http://localhost:8080/v1/slots/0/info | jq '{n_tokens, n_messages}'
```

---

## Server-Side Session Storage

Provides in-memory session storage on the server, eliminating the need to transfer large KV cache blobs (potentially 100MB-4GB) over HTTP for each save/restore operation. Sessions are stored by integer ID and persist until explicitly removed or the server restarts.

This mirrors the session API in `server-coro` PHP extension (`swoole_llama_session_save`, `_restore`, `_remove`, `_list`).

**Requires:** `--slots`

### Save Session

Saves a slot's KV cache state to server-side session storage.

| Method | Path | Query |
|--------|------|-------|
| POST | `/sessions/:id_session` | `?action=save&slot=:id_slot` |

#### Response

```json
{
  "id_session": 42,
  "id_slot": 0,
  "n_tokens": 156,
  "n_bytes": 4194560,
  "t_ms": 12.5
}
```

#### curl Example

```bash
# Save slot 0 to session 42
curl -s -X POST "http://localhost:8080/sessions/42?action=save&slot=0" | jq .
```

---

### Restore Session

Restores a session from server-side storage to a slot. Optionally, can restore directly from a file if `--slot-save-path` is configured.

| Method | Path | Query |
|--------|------|-------|
| POST | `/sessions/:id_session` | `?action=restore&slot=:id_slot` |

#### Request Body (Optional JSON)

| Field | Type | Description |
|-------|------|-------------|
| `file` | string | Filename to restore from (requires `--slot-save-path`) |

When `file` is provided, the session is restored from the specified file in the `--slot-save-path` directory instead of from server-side session storage. This allows for persistent file-based storage.

#### Response

```json
{
  "id_session": 42,
  "id_slot": 0,
  "n_bytes_read": 4194560,
  "success": true,
  "t_ms": 8.3
}
```

#### curl Examples

```bash
# Restore session 42 from server-side storage to slot 1
curl -s -X POST "http://localhost:8080/sessions/42?action=restore&slot=1" | jq .

# Restore from file (requires --slot-save-path)
curl -s -X POST "http://localhost:8080/sessions/42?action=restore&slot=1" \
  -H "Content-Type: application/json" \
  -d '{"file": "session_42.bin"}' | jq .
```

---

### Remove Session

Removes a session from server-side storage. Optionally, saves the session to a file before removing if `--slot-save-path` is configured.

| Method | Path | Query |
|--------|------|-------|
| POST | `/sessions/:id_session` | `?action=remove` |

#### Request Body (Optional JSON)

| Field | Type | Description |
|-------|------|-------------|
| `file` | string | Filename to save session to before removing (requires `--slot-save-path`) |

When `file` is provided, the session blob is saved to the specified file in the `--slot-save-path` directory before being removed from server-side storage. This enables archiving sessions to persistent storage.

#### Response

Without file:
```json
{
  "id_session": 42,
  "removed": true
}
```

With file:
```json
{
  "id_session": 42,
  "removed": true,
  "file": "session_42.bin",
  "n_bytes": 4194560
}
```

#### curl Examples

```bash
# Remove session 42 from memory
curl -s -X POST "http://localhost:8080/sessions/42?action=remove" | jq .

# Save to file before removing (requires --slot-save-path)
curl -s -X POST "http://localhost:8080/sessions/42?action=remove" \
  -H "Content-Type: application/json" \
  -d '{"file": "session_42.bin"}' | jq .
```

---

### List Sessions

Lists all sessions in server-side storage.

| Method | Path |
|--------|------|
| GET | `/sessions` |

#### Response

```json
[
  {"id": 1, "size": 4194560, "created_at": 1739376000, "updated_at": 1739376120},
  {"id": 42, "size": 8388608, "created_at": 1739375900, "updated_at": 1739376100}
]
```

#### curl Example

```bash
curl -s http://localhost:8080/sessions | jq .
```

---

### Multi-Turn Conversation Workflow

```bash
# Turn 1: Initial conversation
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "My name is Alice."}], "id_slot": 0}' | jq .

# Save session
curl -s -X POST "http://localhost:8080/sessions/1?action=save&slot=0" | jq .

# Turn 2: Continue (restore first, then chat)
curl -s -X POST "http://localhost:8080/sessions/1?action=restore&slot=0"
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Hello Alice!"},
    {"role": "user", "content": "What is my name?"}
  ], "id_slot": 0}' | jq .

# Update session with new state
curl -s -X POST "http://localhost:8080/sessions/1?action=save&slot=0"

# Cleanup when done
curl -s -X POST "http://localhost:8080/sessions/1?action=remove"
```

---

## Summary of New Endpoints

| Method | Path | Flag Required | Description |
|--------|------|---------------|-------------|
| POST | `/classify`, `/v1/classify` | `--reranking` | Text classification |
| POST | `/slots/:id?action=save-state` | `--slots` | Serialize slot KV cache to binary blob |
| POST | `/slots/:id?action=restore-state` | `--slots` | Restore slot KV cache from binary blob |
| POST | `/slots/:id?action=tokens` | `--slots` | Get slot's prompt token IDs |
| POST | `/slots/:id?action=context-shift` | `--slots` | Manual KV cache context shift |
| GET | `/v1/slots/:id/info` | `--slots` | Slot token state and message boundaries |
| POST | `/sessions/:id?action=save&slot=N` | `--slots` | Save slot to server-side session storage |
| POST | `/sessions/:id?action=restore&slot=N` | `--slots` | Restore session to slot (optional `{"file":"name"}` with `--slot-save-path`) |
| POST | `/sessions/:id?action=remove` | `--slots` | Remove session (optional `{"file":"name"}` to save first with `--slot-save-path`) |
| GET | `/sessions` | `--slots` | List all server-side sessions |

The existing file-based slot actions (`save`, `restore`, `erase`) continue to require `--slot-save-path` and are unchanged.

The `restore` and `remove` session actions support optional file-based storage when `--slot-save-path` is configured:

- **Restore from file:** `POST /sessions/:id?action=restore&slot=N` with body `{"file": "session.bin"}` reads from file instead of session map
- **Remove to file:** `POST /sessions/:id?action=remove` with body `{"file": "session.bin"}` saves to file before removing from map

This enables a hybrid workflow: use fast server-side storage during active conversations, then archive/restore to files for persistence across server restarts.
