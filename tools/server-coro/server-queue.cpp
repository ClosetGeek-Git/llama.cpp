#include "server-task.h"
#include "server-queue.h"

#include "log.h"

// Swoole headers for coroutine-aware waiting
#include "swoole.h"
#include "swoole_coroutine.h"
#include "swoole_coroutine_api.h"
#include "swoole_pipe.h"

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define RES_INF(fmt, ...) LOG_INF("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_WRN(fmt, ...) LOG_WRN("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_ERR(fmt, ...) LOG_ERR("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_DBG(fmt, ...) LOG_DBG("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

// Coroutine debug macro - uses coro_debug_enabled() from server-queue.h
#define CORO_DBG(fmt, ...) \
    do { if (coro_debug_enabled()) { fprintf(stderr, "[CORO_DBG] %s: " fmt "\n", __func__, ##__VA_ARGS__); fflush(stderr); } } while(0)

// Yield to event loop using defer+yield pattern.
// Unlike swoole_coroutine_usleep(1), this works correctly when other coroutines
// are blocked on swoole_coroutine_socket_wait_event() because adding a defer task
// causes epoll_wait timeout to become 0, allowing immediate return.
static void coroutine_reschedule() {
    auto *co = swoole::Coroutine::get_current();
    if (!co) return;
    swoole_event_defer([](void *arg) {
        static_cast<swoole::Coroutine*>(arg)->resume();
    }, co);
    co->yield();
}

//
// server_queue
//

int server_queue::post(server_task && task, bool front) {
    timing_log("QUEUE_POST_ENTER", task.id, {{"front", front}});
    std::unique_lock<std::mutex> lock(mutex_tasks);
    timing_log("QUEUE_POST_LOCKED", task.id);
    GGML_ASSERT(task.id != -1);
    // if this is cancel task make sure to clean up pending tasks
    if (task.type == SERVER_TASK_TYPE_CANCEL) {
        cleanup_pending_task(task.id_target);
    }
    const int task_id = task.id;
    QUE_DBG("new task, id = %d, front = %d\n", task_id, front);
    if (front) {
        queue_tasks.push_front(std::move(task));
    } else {
        queue_tasks.push_back(std::move(task));
    }
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
    timing_log("QUEUE_POST_EXIT", task_id);
    return task_id;
}

int server_queue::post(std::vector<server_task> && tasks, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    for (auto & task : tasks) {
        if (task.id == -1) {
            task.id = id++;
        }
        // if this is cancel task make sure to clean up pending tasks
        if (task.type == SERVER_TASK_TYPE_CANCEL) {
            cleanup_pending_task(task.id_target);
        }
        QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int) tasks.size(), front);
        if (front) {
            queue_tasks.push_front(std::move(task));
        } else {
            queue_tasks.push_back(std::move(task));
        }
    }
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
    return 0;
}

void server_queue::defer(server_task && task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    QUE_DBG("defer task, id = %d\n", task.id);
    queue_tasks_deferred.push_back(std::move(task));
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
}

int server_queue::get_new_id() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    int new_id = id++;
    return new_id;
}

void server_queue::pop_deferred_task(int id_slot) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!queue_tasks_deferred.empty()) {
        // try to find a task that uses the specified slot
        bool found = false;
        for (auto it = queue_tasks_deferred.begin(); it != queue_tasks_deferred.end(); ++it) {
            if (it->id_slot == id_slot) {
                QUE_DBG("pop deferred task (use slot %d), id_task = %d\n", id_slot, it->id);
                queue_tasks.emplace_front(std::move(*it));
                queue_tasks_deferred.erase(it);
                found = true;
                break;
            }
        }
        // if not tasks found using the slot, just pop the first deferred task (default behavior)
        if (!found) {
            QUE_DBG("pop deferred task, id_task = %d\n", queue_tasks_deferred.front().id);
            queue_tasks.emplace_front(std::move(queue_tasks_deferred.front()));
            queue_tasks_deferred.pop_front();
        }
    }
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
}

void server_queue::wait_until_no_sleep() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!sleeping) {
        return;
    } else {
        if (!req_stop_sleeping) {
            QUE_DBG("%s", "requesting to stop sleeping\n");
            req_stop_sleeping = true;
            condition_tasks.notify_one(); // only main thread is waiting on this
        }
        QUE_DBG("%s", "waiting until no sleep\n");
        condition_tasks.wait(lock, [&]{
            return !sleeping;
        });
    }
}

void server_queue::terminate() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    running = false;
    condition_tasks.notify_all();
}

void server_queue::start_loop(int64_t idle_sleep_ms) {
    running = true;
    time_last_task = ggml_time_ms();
    CORO_DBG("start_loop ENTER: idle_sleep_ms=%ld", (long)idle_sleep_ms);

    constexpr auto max_wait_time = std::chrono::seconds(1);
    auto should_sleep = [&]() -> bool {
        // caller must hold mutex_tasks
        if (idle_sleep_ms < 0) {
            return false;
        }
        int64_t now = ggml_time_ms();
        return (now - time_last_task) >= idle_sleep_ms;
    };

    while (true) {
        QUE_DBG("%s", "processing new tasks\n");

        int tasks_drained = 0;
        int64_t drain_start_us = ggml_time_us();
        size_t queue_size_at_start = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            queue_size_at_start = queue_tasks.size();
        }
        CORO_DBG("start_loop: CYCLE_START queue_size=%zu timestamp_us=%lld", queue_size_at_start, (long long)drain_start_us);

        while (true) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (!running) {
                CORO_DBG("start_loop: terminating");
                QUE_DBG("%s", "terminate\n");
                return;
            }
            if (queue_tasks.empty()) {
                lock.unlock();
                break;
            }
            server_task task = std::move(queue_tasks.front());
            queue_tasks.pop_front();
            size_t remaining = queue_tasks.size();
            lock.unlock();
            tasks_drained++;

            timing_log("TASK_PICKUP", task.id, {{"remaining", remaining}, {"task_type", (int)task.type}});
            CORO_DBG("start_loop: processing task id=%d type=%d (drained=%d remaining=%zu)", task.id, (int)task.type, tasks_drained, remaining);
            QUE_DBG("processing task, id = %d\n", task.id);
            callback_new_task(std::move(task));
            CORO_DBG("start_loop: task id=%d processed", task.id);
        }
        int64_t drain_end_us = ggml_time_us();
        CORO_DBG("start_loop: DRAIN_COMPLETE tasks_drained=%d drain_time_us=%lld", tasks_drained, (long long)(drain_end_us - drain_start_us));
        // all tasks in the current loop is processed, slots data is now ready
        CORO_DBG("start_loop: calling update_slots");
        QUE_DBG("%s", "update slots\n");

        // this will run the main inference process for all slots
        int64_t update_slots_start_us = ggml_time_us();
        timing_log("UPDATE_SLOTS_ENTER", -1, {{"tasks_drained", tasks_drained}});
        callback_update_slots();
        int64_t update_slots_end_us = ggml_time_us();
        timing_log("UPDATE_SLOTS_EXIT", -1, {{"duration_us", update_slots_end_us - update_slots_start_us}});
        CORO_DBG("start_loop: update_slots returned (took %lld us)", (long long)(update_slots_end_us - update_slots_start_us));
        {
            // update_slots() may take a while to finish, we need to make sure it's not counted as idle
            std::unique_lock<std::mutex> lock(mutex_tasks);
            time_last_task = ggml_time_ms();
        }

        CORO_DBG("start_loop: waiting for new tasks");
        QUE_DBG("%s", "waiting for new tasks\n");
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (!running || !queue_tasks.empty()) {
                break; // go back to process new tasks or terminate
            }

            // no tasks, check for sleeping state
            if (should_sleep()) {
                QUE_INF("%s", "entering sleeping state\n");
                sleeping = true;
                callback_sleeping_state(true);
                req_stop_sleeping = false;
                // wait until we are requested to exit sleeping state
                condition_tasks.wait(lock, [&]{
                    return (!running || req_stop_sleeping);
                });
                if (!running) { // may changed during sleep
                    break; // terminate
                }
                QUE_INF("%s", "exiting sleeping state\n");
                req_stop_sleeping = false;
                callback_sleeping_state(false);
                sleeping = false;
                time_last_task = ggml_time_ms();
                condition_tasks.notify_all(); // notify wait_until_no_sleep()
                break; // process new tasks
            } else {
                // wait for new tasks or timeout for checking sleeping condition
                bool res = condition_tasks.wait_for(lock, max_wait_time, [&]{
                    return (!queue_tasks.empty() || !running);
                });
                if (res) {
                    break; // new task arrived or terminate
                }
                // otherwise, loop again to check sleeping condition
            }
        }
    }
}

void server_queue::cleanup_pending_task(int id_target) {
    // no need lock because this is called exclusively by post()
    auto rm_func = [id_target](const server_task & task) {
        return task.id == id_target;
    };
    queue_tasks.erase(
        std::remove_if(queue_tasks.begin(),          queue_tasks.end(),          rm_func),
        queue_tasks.end());
    queue_tasks_deferred.erase(
        std::remove_if(queue_tasks_deferred.begin(), queue_tasks_deferred.end(), rm_func),
        queue_tasks_deferred.end());
}

//
// server_response
//

void server_response::register_pipe(int id_task, int pipe_write_fd) {
    std::unique_lock<std::mutex> lock(mutex_pipe);
    task_to_pipe_fd[id_task] = pipe_write_fd;
    RES_DBG("registered pipe fd %d for task %d\n", pipe_write_fd, id_task);
}

void server_response::unregister_pipe(int id_task) {
    std::unique_lock<std::mutex> lock(mutex_pipe);
    task_to_pipe_fd.erase(id_task);
    RES_DBG("unregistered pipe for task %d\n", id_task);
}

void server_response::add_waiting_task_id(int id_task) {
    RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.insert(id_task);
}

void server_response::add_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto & id_task : id_tasks) {
        RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());
        waiting_task_ids.insert(id_task);
    }
}

void server_response::remove_waiting_task_id(int id_task) {
    RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.erase(id_task);
    // make sure to clean up all pending results
    queue_results.erase(
        std::remove_if(queue_results.begin(), queue_results.end(), [id_task](const server_task_result_ptr & res) {
            return res->id == id_task;
        }),
        queue_results.end());
}

void server_response::remove_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto & id_task : id_tasks) {
        RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());
        waiting_task_ids.erase(id_task);
    }
}

server_task_result_ptr server_response::recv(const std::unordered_set<int> & id_tasks) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        condition_results.wait(lock, [&]{
            if (!running) {
                RES_DBG("%s : queue result stop\n", "recv");
                std::terminate(); // we cannot return here since the caller is HTTP code
            }
            return !queue_results.empty();
        });

        for (size_t i = 0; i < queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                server_task_result_ptr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }
    }

    // should never reach here
}

server_task_result_ptr server_response::recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);

        // Check running state first (for clean shutdown)
        if (!running) {
            RES_DBG("%s : queue shutting down, returning nullptr\n", __func__);
            return nullptr;
        }

        for (int i = 0; i < (int) queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                server_task_result_ptr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }

        std::cv_status cr_res = condition_results.wait_for(lock, std::chrono::seconds(timeout));
        if (!running) {
            RES_DBG("%s : queue result stop\n", __func__);
            return nullptr;  // Return nullptr instead of terminate for clean shutdown
        }
        if (cr_res == std::cv_status::timeout) {
            return nullptr;
        }
    }

    // should never reach here
}

server_task_result_ptr server_response::recv(int id_task) {
    std::unordered_set<int> id_tasks = {id_task};
    return recv(id_tasks);
}

void server_response::send(server_task_result_ptr && result) {
    RES_DBG("sending result for task id = %d\n", result->id);
    timing_log("SEND_ENTER", result->id, {{"is_stop", result->is_stop()}, {"is_error", result->is_error()}});
    CORO_DBG("ENTER: result id=%d is_stop=%d is_error=%d", result->id, result->is_stop(), result->is_error());

    int result_id = result->id;

    {
        std::unique_lock<std::mutex> lock(mutex_results);
        timing_log("SEND_LOCKED_RESULTS", result_id);
        CORO_DBG("checking waiting_task_ids, size=%zu", waiting_task_ids.size());
        bool found = false;
        for (const auto & id_task : waiting_task_ids) {
            if (result_id == id_task) {
                RES_DBG("task id = %d pushed to result queue\n", result_id);
                CORO_DBG("task id=%d matched waiting task, pushing to queue", result_id);

                queue_results.emplace_back(std::move(result));
                condition_results.notify_all();
                found = true;
                break;
            }
        }
        if (!found) {
            CORO_DBG("task id=%d NOT found in waiting_task_ids!", result_id);
        }
    }

    // Signal pipe to wake coroutine (outside of mutex_results lock)
    // Write notification byte to pipe - coroutine will wake via swoole_coroutine_socket_wait_event
    {
        int64_t mutex_start_us = ggml_time_us();
        std::unique_lock<std::mutex> lock(mutex_pipe);
        int64_t mutex_end_us = ggml_time_us();
        timing_log("SEND_PIPE_MUTEX", result_id, {{"wait_us", mutex_end_us - mutex_start_us}});
        
        auto it = task_to_pipe_fd.find(result_id);
        if (it != task_to_pipe_fd.end()) {
            timing_log("SEND_PIPE_WRITE", result_id, {{"fd", it->second}});
            CORO_DBG("writing notification to pipe fd %d for task %d", it->second, result_id);
            // Write a simple notification byte - the result is already in queue_results
            uint8_t notify = 1;
            ssize_t r = write(it->second, &notify, sizeof(notify));
            if (r < 0) {
                RES_WRN("failed to write to pipe for task %d: %s\n", result_id, strerror(errno));
            } else {
                RES_DBG("wrote notification to pipe for task %d\n", result_id);
                timing_log("SEND_PIPE_WRITTEN", result_id);
                CORO_DBG("pipe write successful for task %d", result_id);
            }
        } else {
            timing_log("SEND_NO_PIPE", result_id, {{"map_size", task_to_pipe_fd.size()}});
            CORO_DBG("NO pipe registered for task %d (map size=%zu)", result_id, task_to_pipe_fd.size());
        }
    }
}

void server_response::terminate() {
    RES_DBG("%s", "terminating response queue\n");
    
    // Set shutdown flag BEFORE modifying state - readers check this to avoid use-after-free
    shutdown_flag->store(true);
    
    running = false;
    condition_results.notify_all();

    // Signal all registered pipes to wake up blocked coroutines
    {
        std::unique_lock<std::mutex> lock(mutex_pipe);
        for (const auto & pair : task_to_pipe_fd) {
            uint8_t notify = 1;
            ssize_t r = write(pair.second, &notify, sizeof(notify));
            if (r < 0) {
                RES_WRN("failed to write to pipe during terminate for task %d: %s\n", pair.first, strerror(errno));
            }
        }
    }
}

//
// server_response_reader
//

// Pipe-based notification: decode thread writes to pipe, coroutine waits via swoole_coroutine_socket_wait_event

server_response_reader::server_response_reader(server_queue & queue_tasks, server_response & queue_results, int polling_interval_seconds)
    : queue_tasks(queue_tasks), queue_results(queue_results), polling_interval_seconds(polling_interval_seconds),
      shutdown_flag(queue_results.get_shutdown_flag()) {
    
    // Create pipe for cross-thread notification (decode thread -> coroutine)
    pipe = new swoole::Pipe(false);  // non-blocking
    if (!pipe->ready()) {
        LOG_WRN("Pipe creation failed, falling back to polling mode\n");
        delete pipe;
        pipe = nullptr;
    }
    
    if (pipe) {
        // Get the pipe read fd for swoole_coroutine_socket_wait_event
        auto* read_socket = pipe->get_socket(false);  // false = worker/read side
        if (read_socket) {
            pipe_read_fd = read_socket->get_fd();
            LOG_DBG("Pipe created for reader: read_fd=%d, write_fd=%d\n", 
                    pipe_read_fd, pipe->get_socket(true)->get_fd());
        }
    }
}

server_response_reader::~server_response_reader() {
    stop();
    
    if (pipe) {
        delete pipe;
        pipe = nullptr;
        pipe_read_fd = -1;
    }
}

void server_response_reader::post_task(server_task && task, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_task() can only be called once per reader");
    GGML_ASSERT(!task.is_parent() && "not supported, use post_tasks() instead");
    
    int task_id = task.id;
    int pipe_write_fd = pipe ? pipe->get_socket(true)->get_fd() : -1;
    timing_log("POST_TASK_ENTER", task_id, {{"front", front}, {"pipe_fd", pipe_write_fd}});
    CORO_DBG("ENTER: task_id=%d front=%d pipe_fd=%d", task_id, front, pipe_write_fd);
    
    task.index = 0;
    id_tasks.insert(task.id);
    states.push_back(task.create_state());
    queue_results.add_waiting_task_id(task.id);
    // Register pipe for coroutine-yielding wait
    if (pipe_write_fd >= 0) {
        queue_results.register_pipe(task.id, pipe_write_fd);
    }
    CORO_DBG("task_id=%d: posting to queue...", task_id);
    timing_log("POST_TASK_QUEUE_POST", task_id);
    queue_tasks.post(std::move(task), front);

    // Yield to let other coroutines post their tasks before we wait for results
    // This enables batching of concurrent requests
    timing_log("POST_TASK_YIELD_ENTER", task_id);
    CORO_DBG("task_id=%d: yielding coroutine (coroutine_reschedule)...", task_id);
    coroutine_reschedule();
    timing_log("POST_TASK_YIELD_EXIT", task_id);
    CORO_DBG("task_id=%d: EXIT after yield", task_id);
}

void server_response_reader::post_tasks(std::vector<server_task> && tasks, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_tasks() can only be called once per reader");
    
    int first_task_id = tasks.empty() ? -1 : tasks[0].id;
    int pipe_write_fd = pipe ? pipe->get_socket(true)->get_fd() : -1;
    timing_log("POST_TASKS_ENTER", first_task_id, {{"num_tasks", tasks.size()}, {"front", front}, {"pipe_fd", pipe_write_fd}});
    CORO_DBG("ENTER: num_tasks=%zu front=%d pipe_fd=%d", tasks.size(), front, pipe_write_fd);
    
    id_tasks = server_task::get_list_id(tasks);
    states.reserve(tasks.size());
    size_t index = 0;
    for (auto & task : tasks) {
        task.index = index++;
        states.push_back(task.create_state());
        // for child tasks
        for (auto & child_task : task.child_tasks) {
            child_task.index = index++;
            states.push_back(child_task.create_state());
        }
    }
    GGML_ASSERT(states.size() == id_tasks.size());
    queue_results.add_waiting_task_ids(id_tasks);
    // Register pipe for coroutine-yielding wait
    if (pipe_write_fd >= 0) {
        for (const auto & id_task : id_tasks) {
            queue_results.register_pipe(id_task, pipe_write_fd);
        }
    }
    CORO_DBG("posting %zu tasks to queue...", tasks.size());
    timing_log("POST_TASKS_QUEUE_POST", first_task_id, {{"num_tasks", id_tasks.size()}});
    queue_tasks.post(std::move(tasks), front);

    // Yield to let other coroutines post their tasks before we wait for results
    // This enables batching of concurrent requests
    timing_log("POST_TASKS_YIELD_ENTER", first_task_id);
    CORO_DBG("yielding coroutine (coroutine_reschedule)...");
    coroutine_reschedule();
    timing_log("POST_TASKS_YIELD_EXIT", first_task_id);
    CORO_DBG("EXIT after yield");
}

bool server_response_reader::has_next() const {
    return !cancelled && received_count < id_tasks.size();
}

// return nullptr if should_stop() is true before receiving a result
// note: if one error is received, it will stop further processing and return error result
server_task_result_ptr server_response_reader::next(const std::function<bool()> & should_stop) {
    int first_task_id = id_tasks.empty() ? -1 : *id_tasks.begin();
    bool has_pipe = (pipe_read_fd >= 0);
    timing_log("NEXT_ENTER", first_task_id, {{"num_id_tasks", id_tasks.size()}, {"has_pipe", has_pipe}, {"pipe_read_fd", pipe_read_fd}});
    CORO_DBG("ENTER: num_id_tasks=%zu has_pipe=%d pipe_read_fd=%d", id_tasks.size(), has_pipe, pipe_read_fd);
    int loop_count = 0;
    
    while (true) {
        loop_count++;
        
        // Check if the queue is shutting down
        if (!queue_results.is_running()) {
            timing_log("NEXT_EXIT_SHUTDOWN", first_task_id);
            CORO_DBG("EXIT: queue is shutting down");
            SRV_DBG("%s", "queue is shutting down, returning nullptr\n");
            return nullptr;
        }

        // Try non-blocking fetch first
        timing_log("NEXT_RECV_TRY", first_task_id, {{"loop", loop_count}});
        CORO_DBG("loop=%d: trying recv_with_timeout (non-blocking)...", loop_count);
        server_task_result_ptr result = queue_results.recv_with_timeout(id_tasks, 0);
        
        if (result == nullptr) {
            CORO_DBG("loop=%d: no result yet", loop_count);
            
            // No result yet, check stop condition
            if (should_stop()) {
                timing_log("NEXT_EXIT_SHOULD_STOP", first_task_id);
                CORO_DBG("EXIT: should_stop returned true");
                SRV_DBG("%s", "stopping wait for next result due to should_stop condition\n");
                return nullptr;
            }
            
            // Yield coroutine by waiting on pipe read fd with Swoole's coroutine-aware API
            if (pipe_read_fd >= 0) {
                uint8_t buf[64];
                ssize_t r = read(pipe_read_fd, buf, sizeof(buf));
                if (r < 0) {
                    if (errno == EAGAIN) {
                        // No data available - yield coroutine until pipe becomes readable
                        // swoole_coroutine_socket_wait_event will suspend this coroutine
                        // and resume it when the pipe is signaled (written to)
                        timing_log("NEXT_PIPE_WAIT_ENTER", first_task_id, {{"loop", loop_count}});
                        CORO_DBG("loop=%d: yielding coroutine (swoole_coroutine_socket_wait_event on pipe)...", loop_count);
                        int wait_result = swoole_coroutine_socket_wait_event(pipe_read_fd, SW_EVENT_READ, -1);
                        timing_log("NEXT_PIPE_WAIT_EXIT", first_task_id, {{"loop", loop_count}, {"wait_result", wait_result}});
                        CORO_DBG("loop=%d: coroutine resumed, wait_result=%d", loop_count, wait_result);
                        if (wait_result < 0) {
                            // Wait failed - could be interrupted or fd closed
                            if (!queue_results.is_running()) {
                                CORO_DBG("EXIT: queue stopped during pipe wait");
                                SRV_DBG("%s", "queue stopped during pipe wait\n");
                                return nullptr;
                            }
                            // Otherwise just continue the loop to retry
                        }
                        // After waking, try to drain the pipe
                        read(pipe_read_fd, buf, sizeof(buf));
                    } else {
                        SRV_WRN("pipe read error: %s\n", strerror(errno));
                    }
                } else {
                    CORO_DBG("loop=%d: pipe had data, bytes=%ld", loop_count, (long)r);
                }
                // After waking, check if queue is still running
                if (!queue_results.is_running()) {
                    CORO_DBG("EXIT: queue stopped while waiting on pipe");
                    SRV_DBG("%s", "queue stopped while waiting on pipe\n");
                    return nullptr;
                }
            } else {
                // Fallback: use polling with condition_variable timeout
                CORO_DBG("loop=%d: no pipe, using polling fallback", loop_count);
                result = queue_results.recv_with_timeout(id_tasks, polling_interval_seconds);
                if (result == nullptr) {
                    continue; // timeout, loop to check should_stop
                }
            }
        }
        
        if (result != nullptr) {
            int result_id = result->id;
            bool is_error = result->is_error();
            bool is_stop = result->is_stop();
            timing_log("NEXT_RESULT_RECEIVED", result_id, {{"is_error", is_error}, {"is_stop", is_stop}, {"loop", loop_count}});
            CORO_DBG("loop=%d: got result id=%d is_error=%d is_stop=%d", loop_count, result_id, is_error, is_stop);
            
            if (is_error) {
                stop(); // cancel remaining tasks
                CORO_DBG("EXIT: error result");
                SRV_DBG("%s", "received error result, stopping further processing\n");
                return result;
            }
            if (!states.empty()) {
                // update the generation state if needed
                const size_t idx = result->index;
                GGML_ASSERT(idx < states.size());
                result->update(states[idx]);
            }
            if (is_stop) {
                received_count++;
                CORO_DBG("EXIT: stop result, received_count=%zu", received_count);
            } else {
                CORO_DBG("EXIT: returning streaming chunk");
            }
            return result;
        }
    }

    // should not reach here
}

server_response_reader::batch_response server_response_reader::wait_for_all(const std::function<bool()> & should_stop) {
    batch_response batch_res;
    batch_res.results.clear();
    batch_res.results.resize(id_tasks.size());
    while (has_next()) {
        auto res = next(should_stop);
        if (res == nullptr) {
            batch_res.is_terminated = true;
            return batch_res;
        }
        if (res->is_error()) {
            batch_res.error = std::move(res);
            return batch_res;
        }
        const size_t idx = res->index;
        GGML_ASSERT(idx < batch_res.results.size() && "index out of range");
        GGML_ASSERT(batch_res.results[idx] == nullptr && "duplicate result received");
        batch_res.results[idx] = std::move(res);
    }
    return batch_res;
}

void server_response_reader::stop() {
    // Guard against calling stop() after server_response has been destroyed
    // This can happen if Request objects are GC'd after swoole_llama_shutdown()
    if (shutdown_flag->load()) {
        return;
    }

    // Unregister pipe for all tasks
    int pipe_write_fd = pipe ? pipe->get_socket(true)->get_fd() : -1;
    if (pipe_write_fd >= 0) {
        for (const auto & id_task : id_tasks) {
            queue_results.unregister_pipe(id_task);
        }
        // Write to pipe to wake up any waiting coroutine
        uint8_t buf = 1;
        ssize_t w = write(pipe_write_fd, &buf, 1);
        (void)w;
    }

    queue_results.remove_waiting_task_ids(id_tasks);
    if (has_next() && !cancelled) {
        // if tasks is not finished yet, cancel them
        cancelled = true;
        std::vector<server_task> cancel_tasks;
        cancel_tasks.reserve(id_tasks.size());
        for (const auto & id_task : id_tasks) {
            SRV_WRN("cancel task, id_task = %d\n", id_task);
            server_task task(SERVER_TASK_TYPE_CANCEL);
            task.id_target = id_task;
            queue_results.remove_waiting_task_id(id_task);
            cancel_tasks.push_back(std::move(task));
        }
        // push to beginning of the queue, so it has highest priority
        queue_tasks.post(std::move(cancel_tasks), true);
    } else {
        SRV_DBG("%s", "all tasks already finished, no need to cancel\n");
    }
}
