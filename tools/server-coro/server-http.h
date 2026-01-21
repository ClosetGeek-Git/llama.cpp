#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <memory>
#include <vector>

struct common_params;

// generator-like API for HTTP response generation
// in ZMQ transport, responses are serialized to JSON and carried over a ROUTER socket.
// streaming is internally consumed and sent back as one combined payload.
struct server_http_res {
    std::string content_type = "application/json; charset=utf-8";
    int status = 200;
    std::string data;
    std::map<std::string, std::string> headers;

    // streaming is not externally chunked in ZMQ mode; if set, chunks are pulled and concatenated
    std::function<bool(std::string &)> next = nullptr;
    bool is_stream() const {
        return next != nullptr;
    }

    virtual ~server_http_res() = default;
};

// unique pointer, used by transport
using server_http_res_ptr = std::unique_ptr<server_http_res>;

struct server_http_req {
    std::map<std::string, std::string> params;  // path params + query/body "params"
    std::map<std::string, std::string> headers; // as provided by client JSON
    std::string path;                            // absolute path from client JSON
    std::string body;                            // raw body string
    
    // General interrupt signal for this request's generation.
    // Returns true when generation should stop. Can be triggered by:
    // - Client disconnection (socket no longer writable)
    // - User clicking "Stop" button (future: cancel API)
    // - Timeout or other interrupt conditions
    // Stored by value to avoid dangling reference issues.
    std::function<bool()> should_stop;

    std::string get_param(const std::string & key, const std::string & def = "") const {
        auto it = params.find(key);
        if (it != params.end()) return it->second;
        return def;
    }
};

struct server_http_context {
    class Impl;
    std::unique_ptr<Impl> pimpl;

    std::thread thread; // server thread
    std::atomic<bool> is_ready = false;

    std::string path_prefix;
    std::string hostname;
    int port;

    // Shutdown callback - called from signal handler in reactor context
    // Used to notify main thread (e.g., ctx_server.terminate())
    std::function<void()> on_shutdown;

    server_http_context();
    ~server_http_context();

    bool init(const common_params & params);
    bool start();
    void stop() const;

    using handler_t = std::function<server_http_res_ptr(const server_http_req & req)>;

    void get(const std::string & path, const handler_t & handler) const;
    void post(const std::string & path, const handler_t & handler) const;

    // for debugging
    std::string listening_address;
};
