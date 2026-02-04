#pragma once

#include "server-task.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <sys/eventfd.h>
#include <unistd.h>

// Forward declaration for Swoole Pipe
namespace swoole {
    class Pipe;
}

// Debug logging for coroutine integration - enabled by LLAMA_CORO_DEBUG_JSON env var
inline bool coro_debug_enabled() {
    static std::atomic<int> cached{-1};
    int val = cached.load(std::memory_order_acquire);
    if (val < 0) {
        const char* env = getenv("LLAMA_CORO_DEBUG_JSON");
        val = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
        cached.store(val, std::memory_order_release);
    }
    return val == 1;
}

// Timing debug - enabled by LLAMA_TIMING_DEBUG env var
// Outputs microsecond timestamps at all synchronization points
inline bool timing_debug_enabled() {
    static std::atomic<int> cached{-1};
    int val = cached.load(std::memory_order_acquire);
    if (val < 0) {
        const char* env = getenv("LLAMA_TIMING_DEBUG");
        val = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
        cached.store(val, std::memory_order_release);
    }
    return val == 1;
}

#include <sys/syscall.h>
#include <nlohmann/json.hpp>

inline pid_t get_tid() {
    return syscall(SYS_gettid);
}

// Timing log function - outputs proper JSON using nlohmann::json
inline void timing_log(const char* event, int task_id, nlohmann::ordered_json extra = {}) {
    if (!timing_debug_enabled()) return;
    nlohmann::ordered_json j;
    j["timing"] = true;
    j["tid"] = get_tid();
    j["task"] = task_id;
    j["event"] = event;
    j["ts_us"] = ggml_time_us();
    for (auto& [k, v] : extra.items()) {
        j[k] = v;
    }
    fprintf(stderr, "%s\n", j.dump().c_str());
    fflush(stderr);
}

// struct for managing server tasks
// in most cases, use server_response_reader to post new tasks and retrieve results
struct server_queue {
private:
    int id = 0;
    bool running  = false;
    bool sleeping = false;
    bool req_stop_sleeping = false;
    int64_t time_last_task = 0;

    // queues
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // callback functions
    std::function<void(server_task &&)> callback_new_task;
    std::function<void(void)>           callback_update_slots;
    std::function<void(bool)>           callback_sleeping_state;

public:
    // Add a new task to the end of the queue
    int post(server_task && task, bool front = false);

    // multi-task version of post()
    int post(std::vector<server_task> && tasks, bool front = false);

    // Add a new task, but defer until one slot is available
    void defer(server_task && task);

    // Get the next id for creating a new task
    int get_new_id();

    // Call when the state of one slot is changed, it will move one task from deferred to main queue
    // prioritize tasks that use the specified slot (otherwise, pop the first deferred task)
    void pop_deferred_task(int id_slot);

    // if sleeping, request exiting sleep state and wait until it is done
    // returns immediately if not sleeping
    void wait_until_no_sleep();

    bool is_sleeping() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return sleeping;
    }

    // end the start_loop routine
    void terminate();

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     *
     * Sleeping procedure (disabled if idle_sleep_ms < 0):
     * - If there is no task after idle_sleep_ms, enter sleeping state
     * - Call callback_sleeping_state(true)
     * - Wait until req_stop_sleeping is set to true
     * - Call callback_sleeping_state(false)
     * - Exit sleeping state
     */
    void start_loop(int64_t idle_sleep_ms = -1);

    // for metrics
    size_t queue_tasks_deferred_size() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return queue_tasks_deferred.size();
    }

    //
    // Functions below are not thread-safe, must only be used before start_loop() is called
    //

    // Register function to process a new task
    void on_new_task(std::function<void(server_task &&)> callback) {
        callback_new_task = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Register callback for sleeping state change
    // note: when entering sleeping state, the callback is called AFTER sleeping is set to true
    //       when leaving sleeping state, the callback is called BEFORE sleeping is set to false
    void on_sleeping_state(std::function<void(bool)> callback) {
        callback_sleeping_state = std::move(callback);
    }

private:
    void cleanup_pending_task(int id_target);
};

// struct for managing server responses
// in most cases, use server_response_reader to retrieve results
struct server_response {
private:
    bool running = true;

    // Shared shutdown flag - readers hold this to detect when queue is destroyed
    std::shared_ptr<std::atomic<bool>> shutdown_flag = std::make_shared<std::atomic<bool>>(false);

    // for keeping track of all tasks waiting for the result
    std::unordered_set<int> waiting_task_ids;

    // the main result queue (using ptr for polymorphism)
    std::vector<server_task_result_ptr> queue_results;

    std::mutex mutex_results;
    std::condition_variable condition_results;

    // Pipe-based notification for coroutine-yielding wait
    // maps task_id -> pipe write fd (dedicated lock for cross-thread access from decode thread)
    std::unordered_map<int, int> task_to_pipe_fd;
    std::mutex mutex_pipe;  // Mutex for cross-thread pipe writes

public:
    // Register a pipe write fd for a task to enable coroutine-yielding wait
    void register_pipe(int id_task, int pipe_write_fd);

    // Unregister pipe when task is done or cancelled
    void unregister_pipe(int id_task);
    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task);

    void add_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task);

    // remove multiple tasks from waiting list
    void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // This function blocks the thread until there is a response for one of the id_tasks
    server_task_result_ptr recv(const std::unordered_set<int> & id_tasks);

    // same as recv(), but have timeout in seconds
    // if timeout is reached, nullptr is returned
    server_task_result_ptr recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout);

    // single-task version of recv()
    server_task_result_ptr recv(int id_task);

    // Send a new result to a waiting id_task
    void send(server_task_result_ptr && result);

    // terminate the waiting loop
    void terminate();

    // check if the response queue is still running
    bool is_running() const {
        return running;
    }

    // Get shared shutdown flag for readers to detect when queue is destroyed
    std::shared_ptr<std::atomic<bool>> get_shutdown_flag() const {
        return shutdown_flag;
    }
};

// utility class to make working with server_queue and server_response easier
// it provides a generator-like API for server responses
// support pooling connection state and aggregating multiple results
struct server_response_reader {
    std::unordered_set<int> id_tasks;
    server_queue & queue_tasks;
    server_response & queue_results;
    size_t received_count = 0;
    bool cancelled = false;
    int polling_interval_seconds;

    // Holds shared shutdown flag to detect when queue is destroyed (avoids use-after-free in stop())
    std::shared_ptr<std::atomic<bool>> shutdown_flag;

    // Pipe for cross-thread notification (decode thread writes, coroutine waits)
    swoole::Pipe* pipe = nullptr;              // Per-reader pipe for notification
    int pipe_read_fd = -1;                     // Cached pipe read fd for swoole_coroutine_socket_wait_event

    // tracking generation state and partial tool calls
    // only used by streaming completions
    std::vector<task_result_state> states;

    // should_stop function will be called each polling_interval_seconds
    // Constructor and destructor implemented in .cpp file (needs Swoole includes)
    server_response_reader(server_queue & queue_tasks, server_response & queue_results, int polling_interval_seconds);
    ~server_response_reader();

    int get_new_id() {
        return queue_tasks.get_new_id();
    }

    // if front = true, the task will be posted to the front of the queue (high priority)
    void post_task(server_task && task, bool front = false);
    void post_tasks(std::vector<server_task> && tasks, bool front = false);
    bool has_next() const;

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop);

    struct batch_response {
        bool is_terminated = false; // if true, indicates that processing was stopped before all results were received
        std::vector<server_task_result_ptr> results;
        server_task_result_ptr error; // nullptr if no error
    };
    // aggregate multiple results
    batch_response wait_for_all(const std::function<bool()> & should_stop);

    void stop();
};
