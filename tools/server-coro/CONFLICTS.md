# server-coro upstream merge: conflict inventory

This directory was restored from commit `0196d84` (the parent of `f31b88f` "Remove
server-coro from tracking for privacy") and then 3-way-merged against current
upstream `tools/server`.

- **base**: `tools/server` at `05fa625` (the fork point — Feb 16, 2026)
- **ours**: `tools/server-coro` at `0196d84` (last fork-side state)
- **theirs**: `tools/server` at `upstream/master` (May 9, 2026)

Conflict markers use the `--diff3` form so you can see all three sides:

```
<<<<<<< ours (server-coro)
your coro fork's version
||||||| base (tools/server@fork)
the original fork-point version
=======
upstream's current version
>>>>>>> theirs (tools/server@upstream)
```

## Files needing manual resolution

| File | Conflict regions | Notes |
|------|-----------------:|-------|
| `server-context.cpp` | 16 | Largest — coro hooks layered throughout |
| `server-http.cpp`    | 11 | Httplib integration vs coro http layer |
| `server-task.cpp`    |  8 | Task type additions on both sides |
| `README.md`          |  4 | Doc divergence, mostly trivial |
| `server-context.h`   |  3 | Public API additions (your `get_slot_state`, `set_slot_state`, `get_slot_tokens`, `context_shift` vs upstream's `on_sleeping_changed`) — usually keep both |
| `server-http.h`      |  3 | Httplib types |
| `server-task.h`      |  3 | Task/result struct additions |
| `server.cpp`         |  3 | Routes/main divergence |
| `CMakeLists.txt`     |  2 | Build target list |
| `server-common.cpp`  |  1 | |
| `server-common.h`    |  1 | |
| `server-models.cpp`  |  1 | |

**Total: 56 conflict regions across 12 files.**

## Files merged cleanly (no conflicts)

- `README-dev.md`
- `server-models.h`
- `server-queue.h`

## Files unchanged in upstream (no merge needed)

`chat-llama2.sh`, `chat.mjs`, `chat.sh`, `server-queue.cpp` — restored at
fork-state.

## Coro-specific files (no upstream counterpart, restored as-is)

`DISTILBERT.md`, `config.h.in~`, `config.m4`, `configure~`,
`coro-extension.cpp`, `coro-extension.h`, `httplib_client.cpp`, `httplib_client.h`,
`httplib_server.h`, `make.sh`, `package-lock.json`, `package.json`, all
`test_*.php`, `test_node_client.js`.

## Suggested resolution approach

Most conflicts are concurrent additions where both sides added new things to
the same struct/function/file — just keep both. The harder ones are in
`server-context.cpp` and `server-http.cpp` where upstream restructured code
that your coro fork had also modified for async support; those need
case-by-case judgment.

Once resolved, delete this file.
