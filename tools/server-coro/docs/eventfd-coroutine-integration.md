# Eventfd-Based Coroutine Integration for llama-server-coro

## 1. Problem Statement

The current `server_response_reader::next()` implementation blocks the OS thread when waiting for inference results. This prevents other Swoole coroutines from executing on the same thread.

**Current blocking pattern** (server-queue.cpp, line 268-275):
```cpp
server_task_result_ptr server_response::recv(const std::unordered_set<int> & id_tasks) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        condition_results.wait(lock, [&]{  // <-- BLOCKS OS THREAD
            ...
        });
```

When a Swoole coroutine calls `next()`, it eventually calls `recv()` or `recv_with_timeout()`, which uses `condition_variable::wait`. This blocks the entire OS thread, not just the current coroutine.

**Goal**: Replace the blocking wait with an eventfd-based mechanism that yields the coroutine to Swoole's event loop.

---

## 2. Architecture Overview

### Current Flow (Blocking)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  HTTP Thread (Swoole coroutines)                                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Coroutine A                                                       │ │
│  │    handler() -> rd.next() -> recv_with_timeout()                   │ │
│  │                                  │                                 │ │
│  │                                  ▼                                 │ │
│  │                     condition_results.wait() ◄── BLOCKS THREAD     │ │
│  │                                  │                                 │ │
│  │                                  │ (other coroutines CANNOT run)   │ │
│  └──────────────────────────────────┼─────────────────────────────────┘ │
└─────────────────────────────────────┼───────────────────────────────────┘
                                      │
┌─────────────────────────────────────┼───────────────────────────────────┐
│  Main Thread (Decode Loop)          │                                   │
│    start_loop() -> update_slots()   │                                   │
│         │                           │                                   │
│         ▼                           ▼                                   │
│    send() -> condition_results.notify_all() ───► unblocks               │
└─────────────────────────────────────────────────────────────────────────┘
```

### New Flow (Coroutine-Compatible)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  HTTP Thread (Swoole coroutines)                                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Coroutine A                                                       │ │
│  │    handler() -> rd.next()                                          │ │
│  │                    │                                               │ │
│  │                    ▼                                               │ │
│  │              read(event_fd) ◄── YIELDS to event loop               │ │
│  │                    │                                               │ │
│  │                    │    ┌──────────────────────────────────┐       │ │
│  │                    │    │  Coroutine B, C, D can run here  │       │ │
│  │                    │    └──────────────────────────────────┘       │ │
│  │                    │                                               │ │
│  │              (resumes when eventfd readable)                       │ │
│  │                    │                                               │ │
│  │                    ▼                                               │ │
│  │           grab result from queue                                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │ write(event_fd, 1)
┌─────────────────────────────────────┼───────────────────────────────────┐
│  Main Thread (Decode Loop)          │                                   │
│    start_loop() -> update_slots()   │                                   │
│         │                           │                                   │
│         ▼                           │                                   │
│    send() -> push to queue ─────────┘                                   │
│           -> write(event_fd)                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: Swoole hooks `read()` on file descriptors. When no data is available, the coroutine yields to the event loop. When data arrives (via `write()` from decode thread), only the waiting coroutine resumes.

---

## 3. Data Structure Changes

### 3.1 server_response (server-queue.h, lines 111-155)

Add a map to track which eventfd is waiting for which task:

```cpp
// server-queue.h, add after line 122 (after condition_results declaration)

    // Map task_id -> eventfd for coroutine wakeup
    std::unordered_map<int, int> task_to_eventfd;
```

Add methods to register/unregister eventfds:

```cpp
// server-queue.h, add after line 151 (after terminate() declaration)

    // Register an eventfd for a task (called by server_response_reader)
    void register_eventfd(int id_task, int event_fd);
    
    // Unregister eventfd for a task
    void unregister_eventfd(int id_task);
```

### 3.2 server_response_reader (server-queue.h, lines 157-198)

Add eventfd member:

```cpp
// server-queue.h, add after line 166 (after polling_interval_seconds)

    int event_fd = -1;  // eventfd for coroutine wakeup
```

---

## 4. Implementation Changes

### 4.1 server_response_reader Constructor (server-queue.h, lines 173-175)

Create eventfd in constructor:

```cpp
// server-queue.h, modify lines 173-175

    server_response_reader(server_queue & queue_tasks, server_response & queue_results, int polling_interval_seconds)
        : queue_tasks(queue_tasks), queue_results(queue_results), polling_interval_seconds(polling_interval_seconds) {
        event_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (event_fd < 0) {
            LOG_ERR("failed to create eventfd: %s\n", strerror(errno));
        }
    }
```

### 4.2 server_response_reader Destructor (server-queue.h, lines 176-178)

Close eventfd in destructor:

```cpp
// server-queue.h, modify lines 176-178

    ~server_response_reader() {
        stop();
        if (event_fd >= 0) {
            close(event_fd);
            event_fd = -1;
        }
    }
```

### 4.3 post_task() - Register eventfd (server-queue.cpp, lines 349-359)

Register eventfd when posting task:

```cpp
// server-queue.cpp, modify post_task() around line 356 (before queue_tasks.post)

void server_response_reader::post_task(server_task && task, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_task() can only be called once per reader");
    GGML_ASSERT(!task.is_parent() && "not supported, use post_tasks() instead");
    task.index = 0;
    id_tasks.insert(task.id);
    states.push_back(task.create_state());
    queue_results.add_waiting_task_id(task.id);
    
    // Register eventfd for this task
    if (event_fd >= 0) {
        queue_results.register_eventfd(task.id, event_fd);
    }
    
    queue_tasks.post(std::move(task), front);
}
```

### 4.4 post_tasks() - Register eventfd (server-queue.cpp, lines 361-378)

Register eventfd for all tasks:

```cpp
// server-queue.cpp, modify post_tasks() around line 376 (before queue_tasks.post)

void server_response_reader::post_tasks(std::vector<server_task> && tasks, bool front) {
    // ... existing code ...
    
    queue_results.add_waiting_task_ids(id_tasks);
    
    // Register eventfd for all tasks
    if (event_fd >= 0) {
        for (const auto & id_task : id_tasks) {
            queue_results.register_eventfd(id_task, event_fd);
        }
    }
    
    queue_tasks.post(std::move(tasks), front);
}
```

### 4.5 server_response::send() - Write to eventfd (server-queue.cpp, lines 318-333)

Signal the eventfd when a result is ready:

```cpp
// server-queue.cpp, modify send() - add after line 328 (after queue_results.emplace_back)

void server_response::send(server_task_result_ptr && result) {
    RES_DBG("sending result for task id = %d\n", result->id);
    
    int event_fd_to_signal = -1;

    {
        std::unique_lock<std::mutex> lock(mutex_results);
        for (const auto & id_task : waiting_task_ids) {
            if (result->id == id_task) {
                RES_DBG("task id = %d pushed to result queue\n", result->id);
                
                // Check if there's an eventfd to signal
                auto it = task_to_eventfd.find(result->id);
                if (it != task_to_eventfd.end()) {
                    event_fd_to_signal = it->second;
                }

                queue_results.emplace_back(std::move(result));
                condition_results.notify_all();  // keep for backward compat
                break;
            }
        }
    }
    
    // Signal eventfd outside of lock
    if (event_fd_to_signal >= 0) {
        uint64_t val = 1;
        ssize_t ret = write(event_fd_to_signal, &val, sizeof(val));
        if (ret < 0 && errno != EAGAIN) {
            RES_DBG("failed to write to eventfd: %s\n", strerror(errno));
        }
    }
}
```

### 4.6 server_response::register_eventfd() / unregister_eventfd() (server-queue.cpp)

Add new methods after `send()`:

```cpp
// server-queue.cpp, add after line 333 (after send())

void server_response::register_eventfd(int id_task, int event_fd) {
    std::unique_lock<std::mutex> lock(mutex_results);
    task_to_eventfd[id_task] = event_fd;
    RES_DBG("registered eventfd %d for task %d\n", event_fd, id_task);
}

void server_response::unregister_eventfd(int id_task) {
    std::unique_lock<std::mutex> lock(mutex_results);
    task_to_eventfd.erase(id_task);
    RES_DBG("unregistered eventfd for task %d\n", id_task);
}
```

### 4.7 server_response_reader::next() - Yield on eventfd (server-queue.cpp, lines 384-414)

Replace blocking wait with eventfd read:

```cpp
// server-queue.cpp, replace next() implementation (lines 384-414)

server_task_result_ptr server_response_reader::next(const std::function<bool()> & should_stop) {
    while (true) {
        // First, check if result is already available (non-blocking)
        server_task_result_ptr result = queue_results.recv_with_timeout(id_tasks, 0);
        
        if (result != nullptr) {
            if (result->is_error()) {
                stop();
                SRV_DBG("%s", "received error result, stopping further processing\n");
                return result;
            }
            if (!states.empty()) {
                const size_t idx = result->index;
                GGML_ASSERT(idx < states.size());
                result->update(states[idx]);
            }
            if (result->is_stop()) {
                received_count++;
            }
            return result;
        }
        
        // No result yet - check should_stop before yielding
        if (should_stop()) {
            SRV_DBG("%s", "stopping wait for next result due to should_stop condition\n");
            return nullptr;
        }
        
        // Yield to Swoole event loop by reading from eventfd
        // Swoole hooks read() - this will yield the coroutine until data is available
        if (event_fd >= 0) {
            uint64_t val;
            ssize_t ret = read(event_fd, &val, sizeof(val));
            if (ret < 0 && errno != EAGAIN) {
                SRV_DBG("eventfd read error: %s\n", strerror(errno));
            }
            // After read returns, result should be available - loop back to grab it
        } else {
            // Fallback: no eventfd, use timeout-based polling
            result = queue_results.recv_with_timeout(id_tasks, polling_interval_seconds);
            if (result == nullptr) {
                continue;  // timeout, check should_stop
            }
            // Process result (same as above)
            if (result->is_error()) {
                stop();
                return result;
            }
            if (!states.empty()) {
                const size_t idx = result->index;
                GGML_ASSERT(idx < states.size());
                result->update(states[idx]);
            }
            if (result->is_stop()) {
                received_count++;
            }
            return result;
        }
    }
}
```

### 4.8 server_response_reader::stop() - Unregister eventfds (server-queue.cpp, lines 430-451)

Unregister eventfds when stopping:

```cpp
// server-queue.cpp, modify stop() - add at beginning (after line 431)

void server_response_reader::stop() {
    // Unregister eventfds for all tasks
    for (const auto & id_task : id_tasks) {
        queue_results.unregister_eventfd(id_task);
    }
    
    queue_results.remove_waiting_task_ids(id_tasks);
    // ... rest of existing code ...
}
```

---

## 5. Required Includes

Add to server-queue.cpp (after line 6):

```cpp
#include <sys/eventfd.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
```

Add to server-queue.h (after line 9):

```cpp
#include <unordered_map>
```

---

## 6. Thread Safety Considerations

1. **`task_to_eventfd` map**: Protected by `mutex_results` (same lock as `queue_results`)

2. **eventfd write from decode thread**: `write()` is async-signal-safe and thread-safe

3. **eventfd read from coroutine**: Only one coroutine owns each `server_response_reader`, so no contention

4. **Registration order**: eventfd is registered before task is posted, ensuring it's ready when result arrives

---

## 7. Why This Works with Swoole

Swoole's coroutine runtime hooks standard POSIX I/O functions including `read()`. When you call `read()` on a file descriptor:

1. If data is available → returns immediately
2. If no data (EAGAIN) → coroutine yields to event loop
3. When fd becomes readable → Swoole resumes the coroutine

The eventfd acts as a signaling mechanism:
- Decode thread writes `1` to eventfd when result is ready
- HTTP coroutine's `read()` unblocks and returns
- Coroutine then grabs the result from the queue

Reference: `swoole-src/src/os/async_thread.cc` uses the same pattern for async thread pool → reactor communication.

---

## 8. Testing Strategy

### 8.1 Concurrent Request Test

```bash
# Send 3 requests simultaneously
for i in 1 2 3; do
  curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Count to 3"}]}' &
done
wait
```

Expected: All 3 requests complete (not sequentially blocked).

### 8.2 Streaming Test

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Expected: SSE chunks arrive progressively.

### 8.3 Early Disconnect Test

```bash
timeout 2 curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Write a long story"}],"stream":true}'
```

Expected: Server logs show clean cancellation, no resource leaks.

---

## 9. Summary of Edits

| File | Location | Change |
|------|----------|--------|
| server-queue.h | Line 9 | Add `#include <unordered_map>` |
| server-queue.h | Line 122 | Add `task_to_eventfd` map to `server_response` |
| server-queue.h | Line 151 | Add `register_eventfd()` / `unregister_eventfd()` declarations |
| server-queue.h | Line 166 | Add `event_fd` member to `server_response_reader` |
| server-queue.h | Lines 173-178 | Modify constructor/destructor for eventfd lifecycle |
| server-queue.cpp | Line 6 | Add eventfd includes |
| server-queue.cpp | Lines 318-333 | Modify `send()` to write to eventfd |
| server-queue.cpp | After line 333 | Add `register_eventfd()` / `unregister_eventfd()` implementations |
| server-queue.cpp | Lines 349-359 | Modify `post_task()` to register eventfd |
| server-queue.cpp | Lines 361-378 | Modify `post_tasks()` to register eventfd |
| server-queue.cpp | Lines 384-414 | Replace `next()` with eventfd-based implementation |
| server-queue.cpp | Lines 430-451 | Modify `stop()` to unregister eventfds |

**Total: ~12 edit locations across 2 files**
