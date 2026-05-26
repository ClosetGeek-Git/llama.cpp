#pragma once

#include <atomic>
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <string>
#include <thread>
<<<<<<< ours (server-coro)
#include <memory>
#include <vector>
#include <cstdint>
#include <queue>
||||||| base (tools/server@fork)
=======
#include <vector>
#include <cstdint>
>>>>>>> theirs (tools/server@upstream)

struct common_params;
struct server_context;

// Case-insensitive string comparator for headers (PSR-7 requirement)
struct case_insensitive_compare {
    bool operator()(const std::string & a, const std::string & b) const {
        return std::lexicographical_compare(
            a.begin(), a.end(), b.begin(), b.end(),
            [](char c1, char c2) { return std::tolower(static_cast<unsigned char>(c1)) < std::tolower(static_cast<unsigned char>(c2)); }
        );
    }
};

// Multi-value header map with case-insensitive keys (PSR-7 compatible)
using http_headers_t = std::map<std::string, std::vector<std::string>, case_insensitive_compare>;

// generator-like API for HTTP response generation
struct server_http_res {
    std::string content_type = "application/json; charset=utf-8";
    int status = 200;
    std::string reason_phrase;           // PSR-7: "OK", "Not Found", etc. (optional)
    std::string protocol_version = "1.1"; // PSR-7: "1.0", "1.1", "2"
    std::string data;
    std::queue<std::string> pending_chunks;  // For multi-element RAW chunks
    http_headers_t headers;

    // streaming: if set, chunks are pulled via next()
    std::function<bool(std::string &)> next = nullptr;
    bool is_stream() const {
        return next != nullptr;
    }

    // PSR-7 helper: set header (replaces all values)
    void set_header(const std::string & name, const std::string & value) {
        headers[name] = {value};
    }

    // PSR-7 helper: add header value (appends to existing)
    void add_header(const std::string & name, const std::string & value) {
        headers[name].push_back(value);
    }

    // PSR-7 helper: get first header value
    std::string get_header(const std::string & name, const std::string & def = "") const {
        auto it = headers.find(name);
        if (it != headers.end() && !it->second.empty()) {
            return it->second[0];
        }
        return def;
    }

    // PSR-7 helper: get comma-joined header line
    std::string get_header_line(const std::string & name) const {
        auto it = headers.find(name);
        if (it != headers.end() && !it->second.empty()) {
            std::string result;
            for (size_t i = 0; i < it->second.size(); ++i) {
                if (i > 0) result += ", ";
                result += it->second[i];
            }
            return result;
        }
        return "";
    }

    virtual ~server_http_res() = default;
};

// unique pointer, used by transport
using server_http_res_ptr = std::unique_ptr<server_http_res>;
using raw_buffer = std::vector<uint8_t>;

struct uploaded_file {
    raw_buffer data;
    std::string filename;
    std::string content_type;
};

struct server_http_req {
<<<<<<< ours (server-coro)
    // PSR-7 core fields
    std::string method;                   // "GET", "POST", etc.
    std::string protocol_version = "1.1"; // "1.0", "1.1", "2"
    std::string request_target;           // Full request target (path + query)
    std::string path;                     // Path component only
    std::string query_string;             // Query string without leading '?'
    std::string body;                     // Raw body string

    // PSR-7 server request fields
    std::string scheme = "http";          // "http" or "https"
    std::string host;                     // Host from Host header or URI
    int port = 0;                         // Port from Host header or URI
    std::string remote_addr;              // Client IP address
    int remote_port = 0;                  // Client port

    // Headers with multi-value support (PSR-7)
    http_headers_t headers;

    // Params: path params + query params (convenience, single-value)
    std::map<std::string, std::string> params;

    // Interrupt signal for request generation
    std::function<bool()> should_stop;

    // Get single param value
||||||| base (tools/server@fork)
    std::map<std::string, std::string> params; // path_params + query_params
    std::map<std::string, std::string> headers; // reserved for future use
    std::string path; // reserved for future use
    std::string body;
    const std::function<bool()> & should_stop;

=======
    std::map<std::string, std::string> params; // path_params + query_params
    std::map<std::string, std::string> headers; // used by MCP proxy
    std::string path;
    std::string query_string; // query parameters string (e.g. "action=save")
    std::string body;
    std::map<std::string, uploaded_file> files; // used for file uploads (form data)
    const std::function<bool()> & should_stop;

>>>>>>> theirs (tools/server@upstream)
    std::string get_param(const std::string & key, const std::string & def = "") const {
        auto it = params.find(key);
        if (it != params.end()) return it->second;
        return def;
    }

    // PSR-7 helper: check if header exists
    bool has_header(const std::string & name) const {
        return headers.find(name) != headers.end();
    }

    // PSR-7 helper: get first header value
    std::string get_header(const std::string & name, const std::string & def = "") const {
        auto it = headers.find(name);
        if (it != headers.end() && !it->second.empty()) {
            return it->second[0];
        }
        return def;
    }

    // PSR-7 helper: get all header values
    std::vector<std::string> get_header_values(const std::string & name) const {
        auto it = headers.find(name);
        if (it != headers.end()) {
            return it->second;
        }
        return {};
    }

    // PSR-7 helper: get comma-joined header line
    std::string get_header_line(const std::string & name) const {
        auto it = headers.find(name);
        if (it != headers.end() && !it->second.empty()) {
            std::string result;
            for (size_t i = 0; i < it->second.size(); ++i) {
                if (i > 0) result += ", ";
                result += it->second[i];
            }
            return result;
        }
        return "";
    }
};

struct server_http_context {
    class Impl;
    std::unique_ptr<Impl> pimpl;

    std::thread thread; // server thread
    std::atomic<bool> is_ready = false;

    // note: the handler should never throw exceptions
    using handler_t = std::function<server_http_res_ptr(const server_http_req & req)>;
    mutable std::unordered_map<std::string, handler_t> handlers;

    std::string path_prefix;
    std::string hostname;
    int port;

    // Shutdown callback - called from signal handler in reactor context
    // Used to notify main thread (e.g., ctx_server.terminate())
    std::function<void()> on_shutdown;

    // Reference to server_context for slot state operations
    server_context * ctx_server = nullptr;
    void set_server_context(server_context * ctx) { ctx_server = ctx; }

    // Slot state operations (delegate to ctx_server)
    std::vector<uint8_t> get_slot_state(int slot_id);
    size_t set_slot_state(int slot_id, const uint8_t * data, size_t len);

    server_http_context();
    ~server_http_context();

    bool init(const common_params & params);
    bool start();
    void stop() const;

<<<<<<< ours (server-coro)
    using handler_t = std::function<server_http_res_ptr(const server_http_req & req)>;

||||||| base (tools/server@fork)
    // note: the handler should never throw exceptions
    using handler_t = std::function<server_http_res_ptr(const server_http_req & req)>;

=======
>>>>>>> theirs (tools/server@upstream)
    void get(const std::string & path, const handler_t & handler) const;
    void post(const std::string & path, const handler_t & handler) const;

    // Register the Google Cloud Platform (Vertex AI) compat (AIP_PREDICT_ROUTE env var, or /predict)
    // Must be called AFTER all other API routes are registered
    void register_gcp_compat();

    // for debugging
    std::string listening_address;
};
