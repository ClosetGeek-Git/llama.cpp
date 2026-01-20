# Coroutine-Compatible Result Delivery for llama-server-coro

## An eventfd-based approach to non-blocking inference result retrieval

**Document Version:** 1.0  
**Date:** January 2026  
**Author:** llama-server-coro Development Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background & Problem Statement](#2-background--problem-statement)
3. [Architecture Overview](#3-architecture-overview)
4. [Current Implementation Analysis](#4-current-implementation-analysis)
5. [Proposed Solution: eventfd Integration](#5-proposed-solution-eventfd-integration)
6. [Implementation Guide](#6-implementation-guide)
7. [Code Reference](#7-code-reference)
8. [Testing Strategy](#8-testing-strategy)
9. [Future Considerations](#9-future-considerations)
10. [Appendices](#appendices)

---

## 1. Executive Summary

### 1.1 Overview

llama-server-coro is a variant of the llama.cpp HTTP server that replaces the
cpp-httplib blocking I/O model with Swoole's coroutine-based event-driven
architecture. This enables efficient handling of many concurrent connections
without dedicating an OS thread per request.

### 1.2 The Core Problem

The current implementation uses `std::condition_variable::wait()` to block
HTTP handler coroutines while waiting for inference results from the decode
thread. This blocking call stalls the entire OS thread, preventing other
coroutines from executing and negating the benefits of coroutine-based I/O.

### 1.3 The Solution

Replace the blocking wait with an **eventfd-based signaling mechanism**. The
decode thread writes to an eventfd when results are ready; the HTTP coroutine
yields on a `read()` of that fd. Since Swoole hooks the `read()` syscall, the
coroutine yields control to the event loop rather than blocking the thread.

### 1.4 Impact

- **True concurrency**: Multiple requests progress simultaneously
- **No thread pool overhead**: Single HTTP thread handles all connections
- **Minimal code changes**: ~70 lines modified across 2 files
- **Zero latency overhead**: eventfd is O(1), no polling required

---

## 2. Background & Problem Statement

### 2.1 The Two-Thread Architecture

llama-server-coro operates with two primary threads:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          THREAD 1: Main/Decode                          │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      start_loop()                                │   │
│   │                                                                 │   │
│   │   while (running) {                                             │   │
│   │       wait on condition_tasks for new work                      │   │
│   │       process_single_task()    // assign task to slot           │   │
│   │       update_slots()           // run llama_decode()            │   │
│   │   }                                                             │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   Responsibilities:                                                     │
│   - GPU/CPU inference via llama_decode()                               │
│   - Slot management and scheduling                                      │
│   - Token sampling and stop condition checking                          │
│   - Sending results via queue_results.send()                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          THREAD 2: HTTP/Swoole                          │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                   Swoole Event Loop                              │   │
│   │                                                                 │   │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │   │
│   │   │ Coro #1  │  │ Coro #2  │  │ Coro #3  │  │ Coro #N  │       │   │
│   │   │ (Req A)  │  │ (Req B)  │  │ (Req C)  │  │ (Req N)  │       │   │
│   │   └──────────┘  └──────────┘  └──────────┘  └──────────┘       │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   Responsibilities:                                                     │
│   - Accept HTTP connections                                             │
│   - Parse requests, tokenize prompts                                    │
│   - Post tasks to decode thread                                         │
│   - Wait for results, send HTTP responses                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Task and Result Flow

The flow for a typical chat completion request:

1. **HTTP Request Arrives**: Swoole accepts connection, creates coroutine
2. **Handler Invoked**: `post_chat_completions` lambda executes
3. **Task Created**: Handler tokenizes prompt, creates `server_task`
4. **Task Posted**: `queue_tasks.post()` enqueues work
5. **Wait for Result**: Handler calls `rd.next()` to get tokens
6. **Decode Processes**: Main thread runs inference, generates tokens
7. **Result Sent**: `queue_results.send()` delivers result
8. **Response Sent**: Handler formats and sends HTTP response
9. **Repeat 5-8**: For streaming, loop until generation complete

### 2.3 The Blocking Wait Problem

The critical issue is in step 5. The current implementation:

```cpp
// In server_response::recv()
server_task_result_ptr server_response::recv(const std::unordered_set<int> & id_tasks) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        
        // THIS BLOCKS THE OS THREAD
        condition_results.wait(lock, [&] {
            return !running || has_result_for(id_tasks);
        });
        
        // ... retrieve and return result ...
    }
}
```

When `condition_results.wait()` is called:
- The calling coroutine is **not** yielded
- The entire OS thread blocks on a futex syscall
- All other coroutines on this thread are stalled
- Concurrent requests effectively serialize

### 2.4 Why Swoole Coroutines Cannot Auto-Hook This

Swoole's coroutine system works by hooking blocking syscalls and converting
them to non-blocking operations with coroutine yield/resume:

| Syscall | Swoole Behavior |
|---------|-----------------|
| `read()` | Non-blocking + yield if EAGAIN |
| `write()` | Non-blocking + yield if EAGAIN |
| `connect()` | Non-blocking + yield until connected |
| `sleep()` | Timer-based yield |
| `recv()/send()` | Non-blocking + yield if EAGAIN |

However, `std::condition_variable::wait()` does not use a hookable syscall.
Internally, it uses platform-specific primitives:
- Linux: `futex()` syscall
- macOS: `__psynch_cvwait()` 

These are not part of Swoole's hook list because:
1. They're not network I/O related
2. Hooking them would break general C++ threading
3. Their semantics don't map cleanly to coroutine yield/resume

Therefore, explicit integration with Swoole's event system is required.

### 2.5 Symptoms of the Problem

When running multiple concurrent streaming requests:

```bash
# Terminal 1: Start first request
curl -N http://localhost:8080/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"Count to 100"}],"stream":true}'

# Terminal 2: Start second request (immediately)
curl -N http://localhost:8080/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"Say hello"}],"stream":true}'
```

Expected behavior: Both streams interleave tokens
Actual behavior: Second request waits until first completes

This is because while coroutine #1 is blocked in `condition_variable::wait()`,
coroutine #2 cannot execute at all—the entire thread is blocked.

---

## 3. Architecture Overview

### 3.1 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HTTP Thread (Swoole)                            │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │                    Swoole Event Loop                           │    │
│   │                                                                │    │
│   │   ┌─────────────────────────────────────────────────────────┐ │    │
│   │   │                  Coroutine Context                       │ │    │
│   │   │                                                         │ │    │
│   │   │  ┌──────────────────────────────────────────────────┐  │ │    │
│   │   │  │           HTTP Handler (e.g., /v1/chat)          │  │ │    │
│   │   │  │                                                   │  │ │    │
│   │   │  │  1. Parse request, tokenize                      │  │ │    │
│   │   │  │  2. Create server_task                           │  │ │    │
│   │   │  │  3. rd.post_task() ────────────────────────┐     │  │ │    │
│   │   │  │                                             │     │  │ │    │
│   │   │  │  4. rd.next() ◄─────────────────────────────┼──┐  │  │ │    │
│   │   │  │       │                                     │  │  │  │ │    │
│   │   │  │       │ read(eventfd) ══► YIELDS           │  │  │  │ │    │
│   │   │  │       │                                     │  │  │  │ │    │
│   │   │  │       ▼                                     │  │  │  │ │    │
│   │   │  │  5. Process result, send HTTP chunk        │  │  │  │ │    │
│   │   │  │  6. Loop to step 4 if streaming            │  │  │  │ │    │
│   │   │  └──────────────────────────────────────────────┘  │  │ │    │
│   │   │                                                         │ │    │
│   │   └─────────────────────────────────────────────────────────┘ │    │
│   └───────────────────────────────────────────────────────────────┘    │
│                                      │                      ▲          │
│                                      │                      │          │
└──────────────────────────────────────┼──────────────────────┼──────────┘
                                       │                      │
                           queue_tasks │                      │ eventfd
                               (post)  │                      │ (signal)
                                       │                      │
┌──────────────────────────────────────┼──────────────────────┼──────────┐
│                                      ▼                      │          │
│                          Main Thread (Decode)               │          │
│                                                             │          │
│   ┌─────────────────────────────────────────────────────────┼──────┐   │
│   │                     start_loop()                        │      │   │
│   │                                                         │      │   │
│   │  ┌───────────────────────────────────────────────────┐ │      │   │
│   │  │              process_single_task()                 │ │      │   │
│   │  │  - Assign task to available slot                  │ │      │   │
│   │  │  - Launch slot with task parameters               │ │      │   │
│   │  └───────────────────────────────────────────────────┘ │      │   │
│   │                          │                              │      │   │
│   │                          ▼                              │      │   │
│   │  ┌───────────────────────────────────────────────────┐ │      │   │
│   │  │                 update_slots()                     │ │      │   │
│   │  │                                                   │ │      │   │
│   │  │  - Batch tokens from active slots                 │ │      │   │
│   │  │  - llama_decode(ctx, batch) ◄── GPU/CPU work     │ │      │   │
│   │  │  - Sample next token per slot                     │ │      │   │
│   │  │  - Check stop conditions                          │ │      │   │
│   │  │                                                   │ │      │   │
│   │  │  For each slot with new token:                    │ │      │   │
│   │  │    send_partial_response(slot, token)             │ │      │   │
│   │  │         │                                         │ │      │   │
│   │  │         ▼                                         │ │      │   │
│   │  │    queue_results.send(result) ────────────────────┼─┘      │   │
│   │  │         │                                         │        │   │
│   │  │         └──► write(eventfd, 1) ═══════════════════┘        │   │
│   │  │                                                   │        │   │
│   │  └───────────────────────────────────────────────────┘        │   │
│   │                                                               │   │
│   └───────────────────────────────────────────────────────────────┘   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Sequence Diagram: Streaming Request

```
HTTP Client          HTTP Coroutine         server_response_reader       Decode Thread
    │                      │                         │                        │
    │  POST /v1/chat       │                         │                        │
    │ ───────────────────► │                         │                        │
    │                      │                         │                        │
    │                      │  create reader          │                        │
    │                      │ ──────────────────────► │                        │
    │                      │                         │                        │
    │                      │  post_task(task)        │                        │
    │                      │ ──────────────────────► │                        │
    │                      │                         │  post to queue_tasks   │
    │                      │                         │ ──────────────────────►│
    │                      │                         │                        │
    │                      │  next()                 │                        │
    │                      │ ──────────────────────► │                        │
    │                      │                         │                        │
    │                      │       read(eventfd)     │                        │
    │                      │ ◄─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │                        │
    │                      │       (YIELDS)          │                        │
    │                      │                         │                        │
    │                      │   ═══════════════════════════════════════════    │
    │                      │   ║  Coroutine suspended, thread available  ║    │
    │                      │   ║  for other coroutines to execute        ║    │
    │                      │   ═══════════════════════════════════════════    │
    │                      │                         │                        │
    │                      │                         │  process_single_task   │
    │                      │                         │ ◄──────────────────────│
    │                      │                         │                        │
    │                      │                         │  update_slots          │
    │                      │                         │  llama_decode()        │
    │                      │                         │  sample token          │
    │                      │                         │                        │
    │                      │                         │  send(result)          │
    │                      │                         │ ◄──────────────────────│
    │                      │                         │                        │
    │                      │                         │  write(eventfd, 1)     │
    │                      │                         │ ◄──────────────────────│
    │                      │                         │                        │
    │                      │   ═══════════════════════════════════════════    │
    │                      │   ║  Swoole reactor sees eventfd readable   ║    │
    │                      │   ║  Resumes coroutine                      ║    │
    │                      │   ═══════════════════════════════════════════    │
    │                      │                         │                        │
    │                      │  result                 │                        │
    │                      │ ◄─────────────────────── │                        │
    │                      │                         │                        │
    │  SSE: data: {...}    │                         │                        │
    │ ◄─────────────────── │                         │                        │
    │                      │                         │                        │
    │                      │  next() [loop]          │                        │
    │                      │ ──────────────────────► │                        │
    │                      │       ...               │        ...             │
```

### 3.3 Why eventfd is the Right Choice

Several mechanisms could signal the coroutine:

| Mechanism | Pros | Cons |
|-----------|------|------|
| **Polling** | Simple to implement | Wastes CPU, adds latency |
| **Pipe** | Standard Unix | 2 fds per reader, buffer management |
| **socketpair** | Bidirectional | Overkill, 2 fds |
| **eventfd** | Single fd, atomic, zero-copy | Linux-only |
| **Swoole Channel** | Clean C++ API | Needs object sharing across threads |

**eventfd advantages:**
1. **Single file descriptor** per reader
2. **Atomic counter** semantics (no buffer management)
3. **Zero-copy** signaling (just counter increment)
4. **Integrates with Swoole** via hooked `read()` syscall
5. **O(1) overhead** for both signal and wait

The Linux-only limitation is acceptable because llama-server-coro is
primarily deployed on Linux servers with CUDA/ROCm GPU support.

### 3.4 eventfd Semantics

eventfd provides a simple counter-based signaling mechanism:

```
                    ┌─────────────────────────────┐
                    │         eventfd             │
                    │                             │
  write(fd, &val) ──┼──► counter += val           │
                    │                             │
  read(fd, &buf) ◄──┼─── buf = counter            │
                    │    counter = 0              │
                    │                             │
                    │    If counter == 0:         │
                    │      EAGAIN (non-blocking)  │
                    │      or block (blocking)    │
                    └─────────────────────────────┘
```

For our use case:
- Decode thread: `write(fd, 1)` increments counter
- HTTP coroutine: `read(fd)` returns counter, resets to 0
- With `EFD_NONBLOCK`: `read()` returns EAGAIN if counter is 0
- Swoole hooks the `read()`: on EAGAIN, yields coroutine

---

## 4. Current Implementation Analysis

### 4.1 Class: `server_queue`

**Location:** `tools/server-coro/server-queue.h`, `server-queue.cpp`

**Purpose:** Manages the task queue consumed by the decode thread. HTTP
handlers post tasks here; the decode thread's `start_loop()` consumes them.

#### 4.1.1 Key Data Members

```cpp
struct server_queue {
private:
    int id = 0;                                      // Next task ID counter
    bool running  = false;                           // Loop running flag
    bool sleeping = false;                           // Idle sleep state
    bool req_stop_sleeping = false;                  // Wake request flag
    int64_t time_last_task = 0;                      // For idle detection

    std::deque<server_task> queue_tasks;             // Main task queue
    std::deque<server_task> queue_tasks_deferred;    // Overflow queue

    std::mutex mutex_tasks;                          // Protects queues
    std::condition_variable condition_tasks;         // Signals new tasks

    // Callbacks set before start_loop()
    std::function<void(server_task &&)> callback_new_task;
    std::function<void(void)>           callback_update_slots;
    std::function<void(bool)>           callback_sleeping_state;
};
```

#### 4.1.2 Method: `post()`

Adds a task to the queue and notifies the decode thread.

```cpp
int server_queue::post(server_task && task, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    
    if (task.id == -1) {
        task.id = id++;          // Assign ID if not set
    }
    int id_task = task.id;
    
    time_last_task = ggml_time_ms();
    
    if (front) {
        queue_tasks.push_front(std::move(task));    // High priority
    } else {
        queue_tasks.push_back(std::move(task));     // Normal priority
    }
    
    lock.unlock();
    condition_tasks.notify_one();    // Wake decode thread
    
    return id_task;
}
```

#### 4.1.3 Method: `start_loop()`

Main decode thread entry point. Waits for tasks and processes them.

```cpp
void server_queue::start_loop(int64_t idle_sleep_ms) {
    running = true;
    
    while (running) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        
        // Wait for tasks (with optional timeout for sleep detection)
        if (idle_sleep_ms > 0) {
            bool has_task = condition_tasks.wait_for(lock,
                std::chrono::milliseconds(idle_sleep_ms),
                [this] { return !queue_tasks.empty() || !running; });
            
            if (!has_task && running) {
                // Enter sleeping state
                enter_sleep_mode();
                continue;
            }
        } else {
            condition_tasks.wait(lock, [this] {
                return !queue_tasks.empty() || !running;
            });
        }
        
        // Process all available tasks
        while (!queue_tasks.empty()) {
            server_task task = std::move(queue_tasks.front());
            queue_tasks.pop_front();
            lock.unlock();
            
            callback_new_task(std::move(task));    // process_single_task()
            
            lock.lock();
        }
        
        lock.unlock();
        callback_update_slots();    // update_slots() - runs inference
    }
}
```

#### 4.1.4 Method: `defer()`

Queues tasks when no slots are available. They're moved back to main queue
when a slot becomes free.

```cpp
void server_queue::defer(server_task && task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    queue_tasks_deferred.push_back(std::move(task));
}
```

### 4.2 Class: `server_response`

**Purpose:** Holds results from the decode thread and allows HTTP handlers
to retrieve them. This is where the blocking wait occurs.

#### 4.2.1 Key Data Members

```cpp
struct server_response {
private:
    bool running = true;
    
    // Tasks currently waiting for results
    std::unordered_set<int> waiting_task_ids;
    
    // Result storage (polymorphic via unique_ptr)
    std::vector<server_task_result_ptr> queue_results;
    
    std::mutex mutex_results;
    std::condition_variable condition_results;    // THE BLOCKING PRIMITIVE
};
```

#### 4.2.2 Method: `send()`

Called by decode thread to deliver results. This is where we'll add
eventfd signaling.

```cpp
void server_response::send(server_task_result_ptr && result) {
    std::unique_lock<std::mutex> lock(mutex_results);
    
    // Only queue if someone is waiting for this task
    if (waiting_task_ids.find(result->id) != waiting_task_ids.end()) {
        queue_results.push_back(std::move(result));
    }
    
    lock.unlock();
    condition_results.notify_all();    // Wake all waiters
    
    // NEW: Signal eventfd here
}
```

#### 4.2.3 Method: `recv()`

Blocking wait for results. **This is the problematic method.**

```cpp
server_task_result_ptr server_response::recv(
        const std::unordered_set<int> & id_tasks) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        
        // BLOCKS THE OS THREAD
        condition_results.wait(lock, [&] {
            if (!running) return true;
            
            // Check if any requested task has a result
            for (auto & result : queue_results) {
                if (id_tasks.find(result->id) != id_tasks.end()) {
                    return true;
                }
            }
            return false;
        });
        
        if (!running) {
            return nullptr;
        }
        
        // Find and return the result
        for (auto it = queue_results.begin(); it != queue_results.end(); ++it) {
            if (id_tasks.find((*it)->id) != id_tasks.end()) {
                auto result = std::move(*it);
                queue_results.erase(it);
                return result;
            }
        }
    }
}
```

#### 4.2.4 Method: `recv_with_timeout()`

Same as `recv()` but with timeout. Returns nullptr on timeout.

```cpp
server_task_result_ptr server_response::recv_with_timeout(
        const std::unordered_set<int> & id_tasks,
        int timeout) {
    std::unique_lock<std::mutex> lock(mutex_results);
    
    bool has_result = condition_results.wait_for(lock,
        std::chrono::seconds(timeout),
        [&] {
            if (!running) return true;
            for (auto & result : queue_results) {
                if (id_tasks.find(result->id) != id_tasks.end()) {
                    return true;
                }
            }
            return false;
        });
    
    if (!has_result || !running) {
        return nullptr;
    }
    
    // Find and return result (same as recv)
    // ...
}
```

### 4.3 Class: `server_response_reader`

**Purpose:** High-level generator-like API that wraps `server_queue` and
`server_response`. Each HTTP handler creates one of these to manage its
request lifecycle.

#### 4.3.1 Key Data Members

```cpp
struct server_response_reader {
    std::unordered_set<int> id_tasks;      // Task IDs this reader tracks
    server_queue & queue_tasks;             // Reference to task queue
    server_response & queue_results;        // Reference to result queue
    size_t received_count = 0;              // Results received so far
    bool cancelled = false;                 // Cancellation flag
    int polling_interval_seconds;           // Timeout for polling
    
    std::vector<task_result_state> states;  // Per-task state tracking
    
    // NEW: eventfd for this reader
    // int event_fd = -1;
};
```

#### 4.3.2 Method: `post_task()`

Posts a task and registers for results.

```cpp
void server_response_reader::post_task(server_task && task, bool front) {
    id_tasks.insert(task.id);
    queue_results.add_waiting_task_id(task.id);
    queue_tasks.post(std::move(task), front);
    
    // NEW: Register eventfd with task ID
}
```

#### 4.3.3 Method: `next()`

Generator-style method to get next result. **This is where we'll add
eventfd-based yielding.**

```cpp
server_task_result_ptr server_response_reader::next(
        const std::function<bool()> & should_stop) {
    while (!should_stop()) {
        // Current: blocks via recv_with_timeout
        auto result = queue_results.recv_with_timeout(
            id_tasks, 
            polling_interval_seconds);
        
        if (result) {
            received_count++;
            
            // Handle error results
            if (result->is_error()) {
                cancelled = true;
                // Cancel remaining tasks
                for (int id : id_tasks) {
                    server_task cancel_task(SERVER_TASK_TYPE_CANCEL);
                    cancel_task.id_target = id;
                    queue_tasks.post(std::move(cancel_task));
                }
                return result;
            }
            
            // Check if this result is final
            if (result->is_stop()) {
                id_tasks.erase(result->id);
            }
            
            return result;
        }
        
        // Timeout - check should_stop and retry
    }
    
    return nullptr;    // Cancelled
}
```

#### 4.3.4 Method: `wait_for_all()`

Collects all results until all tasks complete.

```cpp
server_response_reader::batch_response 
server_response_reader::wait_for_all(const std::function<bool()> & should_stop) {
    batch_response response;
    
    while (!id_tasks.empty()) {
        auto result = next(should_stop);
        
        if (!result) {
            response.is_terminated = true;
            return response;
        }
        
        if (result->is_error()) {
            response.error = std::move(result);
            return response;
        }
        
        response.results.push_back(std::move(result));
    }
    
    return response;
}
```

---

## 5. Proposed Solution: eventfd Integration

### 5.1 Design Goals

1. **Minimal invasiveness**: Change as few lines as possible
2. **Backward compatibility**: Existing code paths still work
3. **No server_context changes**: Keep modifications in server-queue only
4. **Thread safety**: Proper synchronization for eventfd registration

### 5.2 New Data Structures

#### 5.2.1 In `server_response`

```cpp
struct server_response {
private:
    // ... existing members ...
    
    // NEW: Map task IDs to eventfds for signaling
    std::mutex eventfd_mutex;
    std::unordered_map<int, int> task_eventfds;  // task_id -> eventfd
};
```

#### 5.2.2 In `server_response_reader`

```cpp
struct server_response_reader {
    // ... existing members ...
    
    // NEW: eventfd for this reader instance
    int event_fd = -1;
};
```

### 5.3 New Methods

#### 5.3.1 `server_response::register_eventfd()`

```cpp
void server_response::register_eventfd(int task_id, int fd) {
    std::lock_guard<std::mutex> lock(eventfd_mutex);
    task_eventfds[task_id] = fd;
}
```

#### 5.3.2 `server_response::unregister_eventfd()`

```cpp
void server_response::unregister_eventfd(int task_id) {
    std::lock_guard<std::mutex> lock(eventfd_mutex);
    task_eventfds.erase(task_id);
}
```

### 5.4 Modified Methods

#### 5.4.1 `server_response::send()` - Add eventfd Signal

```cpp
void server_response::send(server_task_result_ptr && result) {
    int id = result->id;
    
    {
        std::unique_lock<std::mutex> lock(mutex_results);
        if (waiting_task_ids.find(id) != waiting_task_ids.end()) {
            queue_results.push_back(std::move(result));
        }
    }
    
    condition_results.notify_all();  // Keep for compatibility
    
    // NEW: Signal eventfd if registered
    {
        std::lock_guard<std::mutex> lock(eventfd_mutex);
        auto it = task_eventfds.find(id);
        if (it != task_eventfds.end()) {
            uint64_t val = 1;
            ::write(it->second, &val, sizeof(val));
        }
    }
}
```

#### 5.4.2 `server_response_reader` Constructor - Create eventfd

```cpp
server_response_reader::server_response_reader(
        server_queue & queue_tasks,
        server_response & queue_results,
        int polling_interval_seconds)
    : queue_tasks(queue_tasks)
    , queue_results(queue_results)
    , polling_interval_seconds(polling_interval_seconds) 
{
    // NEW: Create eventfd for this reader
    event_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (event_fd < 0) {
        // Log warning, fall back to polling mode
        SRV_WRN("eventfd creation failed: %s, using polling fallback", 
                strerror(errno));
    }
}
```

#### 5.4.3 `server_response_reader` Destructor - Close eventfd

```cpp
server_response_reader::~server_response_reader() {
    stop();
    
    // NEW: Close eventfd
    if (event_fd >= 0) {
        ::close(event_fd);
        event_fd = -1;
    }
}
```

#### 5.4.4 `server_response_reader::post_task()` - Register eventfd

```cpp
void server_response_reader::post_task(server_task && task, bool front) {
    id_tasks.insert(task.id);
    queue_results.add_waiting_task_id(task.id);
    
    // NEW: Register eventfd for this task
    if (event_fd >= 0) {
        queue_results.register_eventfd(task.id, event_fd);
    }
    
    queue_tasks.post(std::move(task), front);
}
```

#### 5.4.5 `server_response_reader::next()` - Yield on eventfd

```cpp
server_task_result_ptr server_response_reader::next(
        const std::function<bool()> & should_stop) {
    while (!should_stop()) {
        // Non-blocking check for result
        auto result = queue_results.recv_with_timeout(id_tasks, 0);
        
        if (result) {
            received_count++;
            
            // Unregister eventfd for completed task
            if (event_fd >= 0) {
                queue_results.unregister_eventfd(result->id);
            }
            
            // ... existing result handling ...
            return result;
        }
        
        // No result available - yield via eventfd
        if (event_fd >= 0) {
            uint64_t val;
            // Swoole hooks read() - this yields the coroutine
            ssize_t n = ::read(event_fd, &val, sizeof(val));
            
            if (n < 0 && errno != EAGAIN) {
                // Real error - fall through to retry
                SRV_WRN("eventfd read error: %s", strerror(errno));
            }
            // On success or EAGAIN, loop and check for result
        } else {
            // Fallback: Swoole coroutine sleep
            swoole::Coroutine::sleep(0.001);
        }
    }
    
    return nullptr;  // Cancelled
}
```

#### 5.4.6 `server_response_reader::stop()` - Unregister All

```cpp
void server_response_reader::stop() {
    // NEW: Unregister all eventfds
    if (event_fd >= 0) {
        for (int id : id_tasks) {
            queue_results.unregister_eventfd(id);
        }
    }
    
    // ... existing cleanup ...
}
```

---

## 6. Implementation Guide

### 6.1 Required Includes

Add to `server-queue.cpp`:

```cpp
#include <sys/eventfd.h>
#include <unistd.h>
#include <errno.h>
```

### 6.2 Complete Diff for server-queue.h

```diff
 struct server_response {
 private:
     bool running = true;
     std::unordered_set<int> waiting_task_ids;
     std::vector<server_task_result_ptr> queue_results;
     std::mutex mutex_results;
     std::condition_variable condition_results;
+
+    // eventfd registration for coroutine signaling
+    std::mutex eventfd_mutex;
+    std::unordered_map<int, int> task_eventfds;
 
 public:
     void add_waiting_task_id(int id_task);
     void add_waiting_task_ids(const std::unordered_set<int> & id_tasks);
     void remove_waiting_task_id(int id_task);
     void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks);
     server_task_result_ptr recv(const std::unordered_set<int> & id_tasks);
     server_task_result_ptr recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout);
     server_task_result_ptr recv(int id_task);
     void send(server_task_result_ptr && result);
     void terminate();
+
+    // eventfd registration API
+    void register_eventfd(int task_id, int fd);
+    void unregister_eventfd(int task_id);
 };


 struct server_response_reader {
     std::unordered_set<int> id_tasks;
     server_queue & queue_tasks;
     server_response & queue_results;
     size_t received_count = 0;
     bool cancelled = false;
     int polling_interval_seconds;
     std::vector<task_result_state> states;
+
+    // eventfd for coroutine-compatible waiting
+    int event_fd = -1;
```

### 6.3 Thread Safety Analysis

| Operation | Thread | Locks Held | Notes |
|-----------|--------|------------|-------|
| `register_eventfd()` | HTTP | `eventfd_mutex` | Called from `post_task()` |
| `unregister_eventfd()` | HTTP | `eventfd_mutex` | Called from `next()` or `stop()` |
| `send()` write | Decode | `eventfd_mutex` | After `mutex_results` released |
| `read()` | HTTP | None | Lock-free, atomic eventfd op |

Key insight: `mutex_results` and `eventfd_mutex` are never held simultaneously
in the decode thread. The sequence is:
1. Lock `mutex_results`, push result, unlock
2. Lock `eventfd_mutex`, write to eventfd, unlock

This prevents deadlock and minimizes contention.

---

## 7. Code Reference

### 7.1 Linux eventfd API

```cpp
#include <sys/eventfd.h>

// Create eventfd
// initval: initial counter value (usually 0)
// flags: EFD_NONBLOCK, EFD_CLOEXEC, EFD_SEMAPHORE
int eventfd(unsigned int initval, int flags);

// Returns: file descriptor on success, -1 on error
```

**Flags:**
- `EFD_NONBLOCK`: `read()` returns EAGAIN instead of blocking
- `EFD_CLOEXEC`: Close fd on exec()
- `EFD_SEMAPHORE`: Decrement by 1 instead of returning full counter

**Operations:**
```cpp
// Write: add value to counter
uint64_t val = 1;
write(fd, &val, sizeof(val));  // Returns 8 on success

// Read: get counter, reset to 0
uint64_t val;
read(fd, &val, sizeof(val));   // Returns 8 on success, val = counter
                                // Returns -1 with EAGAIN if counter is 0
```

### 7.2 Swoole Coroutine Hooks

Swoole intercepts these syscalls to enable coroutine yielding:

```cpp
// From swoole_coroutine_system.cc
// When read() would block, Swoole:
// 1. Sets fd to non-blocking
// 2. Adds fd to reactor for read events
// 3. Yields coroutine
// 4. When fd becomes readable, resumes coroutine
// 5. Performs the actual read()

ssize_t swoole_coroutine_read(int fd, void *buf, size_t count);
ssize_t swoole_coroutine_write(int fd, const void *buf, size_t count);
```

### 7.3 Coroutine State Transitions

```
                     ┌──────────────────┐
                     │                  │
                     │  SW_CORO_INIT    │
                     │                  │
                     └────────┬─────────┘
                              │
                              │ Coroutine::create()
                              ▼
                     ┌──────────────────┐
           ┌────────►│                  │◄────────┐
           │         │ SW_CORO_RUNNING  │         │
           │         │                  │         │
           │         └────────┬─────────┘         │
           │                  │                   │
           │                  │ yield()           │ resume()
           │                  ▼                   │
           │         ┌──────────────────┐         │
           │         │                  │─────────┘
           │         │ SW_CORO_WAITING  │
           │         │                  │
           │         └────────┬─────────┘
           │                  │
           │                  │ (completed)
           │                  ▼
           │         ┌──────────────────┐
           └─────────│                  │
                     │   SW_CORO_END    │
                     │                  │
                     └──────────────────┘
```

---

## 8. Testing Strategy

### 8.1 Unit Test: eventfd Basic Operations

```cpp
TEST(eventfd, basic_operations) {
    int fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    ASSERT_GE(fd, 0);
    
    // Read should fail with EAGAIN (counter is 0)
    uint64_t val;
    ASSERT_EQ(read(fd, &val, sizeof(val)), -1);
    ASSERT_EQ(errno, EAGAIN);
    
    // Write increments counter
    val = 5;
    ASSERT_EQ(write(fd, &val, sizeof(val)), sizeof(val));
    
    // Read returns counter value
    ASSERT_EQ(read(fd, &val, sizeof(val)), sizeof(val));
    ASSERT_EQ(val, 5);
    
    // Counter is now 0 again
    ASSERT_EQ(read(fd, &val, sizeof(val)), -1);
    ASSERT_EQ(errno, EAGAIN);
    
    close(fd);
}
```

### 8.2 Unit Test: eventfd with Coroutine Yield

```cpp
TEST(eventfd, coroutine_yield_resume) {
    using swoole::test::coroutine;
    
    int fd = eventfd(0, EFD_NONBLOCK);
    std::atomic<int> stage{0};
    
    coroutine::run({
        // Coroutine 1: waits on eventfd
        make_pair([](void* arg) {
            auto* ctx = static_cast<std::pair<int, std::atomic<int>*>*>(arg);
            int fd = ctx->first;
            auto* stage = ctx->second;
            
            stage->store(1);  // Started waiting
            
            uint64_t val;
            read(fd, &val, sizeof(val));  // Should yield
            
            stage->store(3);  // Resumed and completed
        }, new std::pair<int, std::atomic<int>*>(fd, &stage)),
        
        // Coroutine 2: signals eventfd
        make_pair([](void* arg) {
            auto* ctx = static_cast<std::pair<int, std::atomic<int>*>*>(arg);
            int fd = ctx->first;
            auto* stage = ctx->second;
            
            // Wait until coroutine 1 is waiting
            while (stage->load() < 1) {
                swoole::Coroutine::sleep(0.001);
            }
            
            stage->store(2);  // About to signal
            
            uint64_t val = 1;
            write(fd, &val, sizeof(val));
        }, new std::pair<int, std::atomic<int>*>(fd, &stage))
    });
    
    ASSERT_EQ(stage.load(), 3);
    close(fd);
}
```

### 8.3 Integration Test: Concurrent Streaming

```bash
#!/bin/bash
# test_concurrent_streaming.sh

SERVER_URL="http://localhost:8080/v1/chat/completions"
CONCURRENT=5
PIDS=()

# Start concurrent streaming requests
for i in $(seq 1 $CONCURRENT); do
    curl -s -N "$SERVER_URL" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 10, request $i\"}],\"stream\":true}" \
        > /tmp/stream_$i.log 2>&1 &
    PIDS+=($!)
done

echo "Started $CONCURRENT concurrent requests"

# Wait for all to complete
for pid in "${PIDS[@]}"; do
    wait $pid
done

# Verify all completed successfully
SUCCESS=0
for i in $(seq 1 $CONCURRENT); do
    if grep -q '"finish_reason":"stop"' /tmp/stream_$i.log; then
        ((SUCCESS++))
    else
        echo "Request $i failed:"
        tail -5 /tmp/stream_$i.log
    fi
done

echo "Completed: $SUCCESS / $CONCURRENT"
[ $SUCCESS -eq $CONCURRENT ] && exit 0 || exit 1
```

### 8.4 Integration Test: Early Disconnect

```bash
#!/bin/bash
# test_early_disconnect.sh

# Start a long-running request and disconnect after 2 seconds
timeout 2 curl -s -N http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Write a 1000 word essay about coroutines"}],"stream":true}' \
    2>&1

# Give server time to process disconnect
sleep 1

# Check server logs for clean cancellation
if grep -q "srv stop: cancel task" /tmp/llama-server.log | tail -5; then
    echo "PASS: Task cancellation logged"
else
    echo "WARN: No cancellation log found"
fi

# Verify server is still healthy
if curl -s http://localhost:8080/health | grep -q '"status":"ok"'; then
    echo "PASS: Server still healthy"
    exit 0
else
    echo "FAIL: Server unhealthy after disconnect"
    exit 1
fi
```

### 8.5 Performance Benchmark

```bash
#!/bin/bash
# benchmark_concurrent.sh

measure_latency() {
    local concurrent=$1
    local start=$(date +%s.%N)
    
    pids=()
    for i in $(seq 1 $concurrent); do
        curl -s http://localhost:8080/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"test","messages":[{"role":"user","content":"Say hi"}]}' \
            > /dev/null &
        pids+=($!)
    done
    
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    local end=$(date +%s.%N)
    echo "scale=3; $end - $start" | bc
}

echo "Concurrent requests,Total time (s)"
for n in 1 5 10 20 50; do
    time=$(measure_latency $n)
    echo "$n,$time"
done
```

---

## 9. Future Considerations

### 9.1 Batch Results with eventfd Counter

Currently, we write `1` to eventfd for each result. The counter could encode
how many results are available:

```cpp
// In send():
uint64_t val = 1;
write(event_fd, &val, sizeof(val));  // Counter += 1

// In next():
uint64_t val;
read(event_fd, &val, sizeof(val));   // val = total signals since last read
// Could process up to `val` results in a batch
```

This would reduce reactor overhead for high-throughput scenarios.

### 9.2 Per-Task eventfd vs Shared

Current design: One eventfd per `server_response_reader`
- Pros: Simple, minimal fds
- Cons: Spurious wakeups if reader tracks multiple tasks

Alternative: One eventfd per task
- Pros: Precise wakeups
- Cons: More file descriptors, more registration overhead

For typical use (1 task per reader for completions), current design is optimal.

### 9.3 Cross-Platform Support

eventfd is Linux-only. For macOS/FreeBSD:

```cpp
#ifdef __linux__
    event_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
#else
    // Use pipe as fallback
    int pipefd[2];
    if (pipe(pipefd) == 0) {
        fcntl(pipefd[0], F_SETFL, O_NONBLOCK);
        fcntl(pipefd[1], F_SETFL, O_NONBLOCK);
        event_fd_read = pipefd[0];
        event_fd_write = pipefd[1];
    }
#endif
```

### 9.4 Timeout Integration

Could integrate with Swoole's timer for deadline-based cancellation:

```cpp
// In next() with timeout:
auto timer_id = swoole_timer_after(timeout_ms, [co = Coroutine::get_current()](TIMER_PARAMS) {
    co->resume();  // Wake on timeout
});

uint64_t val;
read(event_fd, &val, sizeof(val));

swoole_timer_del(timer_id);
```

### 9.5 Graceful Degradation

If eventfd creation fails:
1. Log warning (not error - server still functional)
2. Fall back to polling via `Coroutine::sleep()`
3. Performance degrades but correctness maintained

---

## Appendices

### Appendix A: Complete Modified server-queue.h

```cpp
#pragma once

#include "server-task.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>
#include <unordered_set>
#include <unordered_map>

struct server_queue {
private:
    int id = 0;
    bool running  = false;
    bool sleeping = false;
    bool req_stop_sleeping = false;
    int64_t time_last_task = 0;

    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    std::function<void(server_task &&)> callback_new_task;
    std::function<void(void)>           callback_update_slots;
    std::function<void(bool)>           callback_sleeping_state;

public:
    int post(server_task && task, bool front = false);
    int post(std::vector<server_task> && tasks, bool front = false);
    void defer(server_task && task);
    int get_new_id();
    void pop_deferred_task(int id_slot);
    void wait_until_no_sleep();
    bool is_sleeping();
    void terminate();
    void start_loop(int64_t idle_sleep_ms = -1);
    size_t queue_tasks_deferred_size();

    void on_new_task(std::function<void(server_task &&)> callback);
    void on_update_slots(std::function<void(void)> callback);
    void on_sleeping_state(std::function<void(bool)> callback);

private:
    void cleanup_pending_task(int id_target);
};

struct server_response {
private:
    bool running = true;
    std::unordered_set<int> waiting_task_ids;
    std::vector<server_task_result_ptr> queue_results;
    std::mutex mutex_results;
    std::condition_variable condition_results;

    // NEW: eventfd registration for coroutine signaling
    std::mutex eventfd_mutex;
    std::unordered_map<int, int> task_eventfds;

public:
    void add_waiting_task_id(int id_task);
    void add_waiting_task_ids(const std::unordered_set<int> & id_tasks);
    void remove_waiting_task_id(int id_task);
    void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks);
    server_task_result_ptr recv(const std::unordered_set<int> & id_tasks);
    server_task_result_ptr recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout);
    server_task_result_ptr recv(int id_task);
    void send(server_task_result_ptr && result);
    void terminate();

    // NEW: eventfd registration API
    void register_eventfd(int task_id, int fd);
    void unregister_eventfd(int task_id);
};

struct server_response_reader {
    std::unordered_set<int> id_tasks;
    server_queue & queue_tasks;
    server_response & queue_results;
    size_t received_count = 0;
    bool cancelled = false;
    int polling_interval_seconds;
    std::vector<task_result_state> states;

    // NEW: eventfd for coroutine-compatible waiting
    int event_fd = -1;

    server_response_reader(server_queue & queue_tasks, server_response & queue_results, int polling_interval_seconds);
    ~server_response_reader();

    int get_new_id();
    void post_task(server_task && task, bool front = false);
    void post_tasks(std::vector<server_task> && tasks, bool front = false);
    bool has_next() const;
    server_task_result_ptr next(const std::function<bool()> & should_stop);

    struct batch_response {
        bool is_terminated = false;
        std::vector<server_task_result_ptr> results;
        server_task_result_ptr error;
    };
    batch_response wait_for_all(const std::function<bool()> & should_stop);
    void stop();
};
```

### Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Coroutine** | Lightweight cooperative thread managed by Swoole's scheduler |
| **eventfd** | Linux kernel mechanism for event notification via file descriptor |
| **Reactor** | Event loop that monitors file descriptors for readiness |
| **Yield** | Suspend coroutine execution, return control to scheduler |
| **Resume** | Continue suspended coroutine from its yield point |
| **Slot** | llama.cpp inference slot for parallel token generation |
| **Task** | Work unit representing an inference request |
| **Result** | Generated token(s) or completion from decode thread |
| **Decode thread** | Thread running `start_loop()` and `llama_decode()` |
| **HTTP thread** | Thread running Swoole event loop and coroutines |

### Appendix C: References

1. **eventfd(2)** - Linux manual page
   - https://man7.org/linux/man-pages/man2/eventfd.2.html

2. **Swoole Coroutine Documentation**
   - https://wiki.swoole.com/en/#/coroutine

3. **llama.cpp Server Documentation**
   - https://github.com/ggerganov/llama.cpp/tree/master/tools/server

4. **Swoole Source Code** - Coroutine implementation
   - `swoole-src/src/coroutine/`
   - `swoole-src/include/swoole_coroutine.h`

---

*Document generated for llama-server-coro eventfd integration project*
