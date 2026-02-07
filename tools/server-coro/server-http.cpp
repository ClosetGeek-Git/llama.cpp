#include "common.h"
#include "server-http.h"
#include "server-common.h"
#include "server-context.h"

// Swoole headers
#include "swoole.h"
#include "swoole_coroutine_api.h"
#include "swoole_signal.h"

#include "httplib_server.h"

#include <functional>
#include <string>
#include <thread>
#include <vector>
#include <unordered_set>
#include <atomic>
#include <regex>
#include <memory>

using json = nlohmann::ordered_json;

// Helper: convert path template like "/slots/:id_slot" to regex pattern and extract param names
static std::pair<std::string, std::vector<std::string>> path_to_regex(const std::string & path) {
    std::vector<std::string> param_names;
    std::string regex_pattern;
    
    size_t i = 0;
    while (i < path.size()) {
        if (path[i] == ':') {
            // Extract parameter name
            size_t j = i + 1;
            while (j < path.size() && path[j] != '/') {
                j++;
            }
            std::string param_name = path.substr(i + 1, j - i - 1);
            param_names.push_back(param_name);
            regex_pattern += "([^/]+)";
            i = j;
        } else {
            // Escape regex special characters
            char c = path[i];
            if (c == '.' || c == '+' || c == '*' || c == '?' || c == '^' || c == '$' ||
                c == '{' || c == '}' || c == '[' || c == ']' || c == '|' || c == '(' || c == ')' || c == '\\') {
                regex_pattern += '\\';
            }
            regex_pattern += c;
            i++;
        }
    }
    
    return {regex_pattern, param_names};
}

static json make_error_json(const std::string & message, int code, const std::string & type) {
    return json {
        {"error", {
            {"message", message},
            {"type", type},
            {"code", code}
        }}
    };
}

// Global pointer for signal handler access (set during start(), cleared on exit)
static server_http_context * g_http_context = nullptr;
static std::atomic_flag g_is_terminating = ATOMIC_FLAG_INIT;

class server_http_context::Impl {
public:
    std::unique_ptr<httplib_coro::Server> srv;
    std::atomic<bool> running{false};

    // config
    std::vector<std::string> api_keys;
    std::unordered_set<std::string> public_endpoints{"/health", "/v1/health", "/models", "/v1/models", "/api/tags"};

    // reference to parent for is_ready check
    const server_http_context * parent = nullptr;

    bool is_public_endpoint(const std::string & path) const {
        // Check with and without prefix
        if (public_endpoints.count(path) > 0) {
            return true;
        }
        // Try stripping api prefix
        if (parent && !parent->path_prefix.empty()) {
            if (path.rfind(parent->path_prefix, 0) == 0) {
                std::string stripped = path.substr(parent->path_prefix.size());
                if (stripped.empty()) stripped = "/";
                if (public_endpoints.count(stripped) > 0) {
                    return true;
                }
            }
        }
        return false;
    }

    bool validate_api_key(const httplib_coro::Request & req) const {
        if (api_keys.empty()) {
            return true;  // No API keys configured
        }

        std::string req_api_key = req.get_header_value("Authorization");
        if (req_api_key.empty()) {
            req_api_key = req.get_header_value("X-Api-Key");
        }

        // Remove "Bearer " prefix
        const std::string bearer = "Bearer ";
        if (req_api_key.rfind(bearer, 0) == 0) {
            req_api_key = req_api_key.substr(bearer.size());
        }

        for (const auto & key : api_keys) {
            if (key == req_api_key) {
                return true;
            }
        }
        return false;
    }
};

server_http_context::server_http_context()
    : pimpl(std::make_unique<server_http_context::Impl>())
{}

server_http_context::~server_http_context() = default;

// coro compatible
std::vector<uint8_t> server_http_context::get_slot_state(int slot_id) {
    if (!ctx_server) {
        return {};
    }
    return ctx_server->get_slot_state(slot_id);
}

// coro compatible
size_t server_http_context::set_slot_state(int slot_id, const uint8_t * data, size_t len) {
    if (!ctx_server) {
        return 0;
    }
    return ctx_server->set_slot_state(slot_id, data, len);
}

bool server_http_context::init(const common_params & params) {
    path_prefix = params.api_prefix;
    port = params.port;
    hostname = params.hostname;

    pimpl->api_keys = params.api_keys;
    pimpl->parent = this;

    pimpl->srv = std::make_unique<httplib_coro::Server>();
    auto & srv = pimpl->srv;

    // Configure logger
    srv->set_logger([](const httplib_coro::Request & req, const httplib_coro::Response & res) {
        if (req.path == "/v1/health") {
            return;  // Skip health check logging
        }
        SRV_INF("request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);
    });

    // Configure error handler
    srv->set_error_handler([](const httplib_coro::Request &, httplib_coro::Response & res) {
        if (res.status == 404) {
            res.set_content(
                safe_json_to_str(make_error_json("File Not Found", 404, "not_found_error")),
                "application/json; charset=utf-8"
            );
        }
    });

    // Configure timeouts
    srv->set_read_timeout(params.timeout_read);
    srv->set_write_timeout(params.timeout_write);

    return true;
}

// Signal handler that runs in reactor context (via signalfd/kqueue)
// Safe to call coroutine operations here
static void http_signal_handler(int signo) {
    if (g_is_terminating.test_and_set()) {
        // Second signal - force immediate exit
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    SRV_INF("%s: received signal %d, initiating shutdown...\n", __func__, signo);

    if (g_http_context) {
        // Stop the HTTP server (safe - we're in reactor context)
        g_http_context->stop();

        // Notify main thread to exit start_loop()
        if (g_http_context->on_shutdown) {
            g_http_context->on_shutdown();
        }
    }
}

bool server_http_context::start() {
    auto & srv = pimpl->srv;

    // Initialize Swoole runtime (must be called before any Swoole API)
    swoole_init();

    pimpl->running.store(true);

    // We need to track if bind succeeded from inside the coroutine
    std::atomic<int> bind_result{-1};  // -1 = pending, 0 = failed, >0 = success (port number)

    // Set global pointer for signal handler access
    g_http_context = this;
    g_is_terminating.clear();

    // Run HTTP server in a thread with Swoole event loop
    thread = std::thread([this, &bind_result]() {
        // Initialize Swoole event loop for this thread
        swoole_event_init(SW_EVENTLOOP_WAIT_EXIT);

        // Register signal handlers using Swoole's signal API
        // This integrates with signalfd/kqueue so callbacks run in reactor context
        // Note: swoole_signal_set will handle blocking signals and setting up signalfd
        swoole_signal_set(SIGINT, http_signal_handler);
        swoole_signal_set(SIGTERM, http_signal_handler);

        // Create coroutine for binding and the accept loop
        swoole::Coroutine::create([](void * arg) {
            auto * params = static_cast<std::pair<server_http_context*, std::atomic<int>*>*>(arg);
            auto * ctx = params->first;
            auto * result = params->second;
            
            auto & srv = ctx->pimpl->srv;
            
            // Bind socket inside coroutine context
            bool was_bound = false;
            int bound_port = ctx->port;
            if (ctx->port == 0) {
                bound_port = srv->bind_to_any_port(ctx->hostname.c_str());
                was_bound = (bound_port >= 0);
            } else {
                was_bound = srv->bind_to_port(ctx->hostname.c_str(), ctx->port);
            }

            if (!was_bound) {
                LOG_ERR("%s: couldn't bind HTTP server socket, hostname: %s, port: %d\n", __func__, ctx->hostname.c_str(), ctx->port);
                result->store(0);
                return;
            }
            
            result->store(bound_port);
            ctx->port = bound_port;
            
            // Now run the accept loop
            srv->listen_after_bind();
        }, new std::pair<server_http_context*, std::atomic<int>*>(this, &bind_result));

        // Block thread, drive all coroutines
        swoole_event_wait();

        // Clean up signal handlers
        swoole_signal_set(SIGINT, nullptr);
        swoole_signal_set(SIGTERM, nullptr);
    });

    // Wait for bind to complete
    while (bind_result.load() == -1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    if (bind_result.load() == 0) {
        pimpl->running.store(false);
        return false;
    }

    listening_address = string_format("http://%s:%d", hostname.c_str(), port);
    return true;
}

void server_http_context::stop() const {
    if (pimpl->srv && pimpl->running.exchange(false)) {
        pimpl->srv->stop();
    }
}

void server_http_context::get(const std::string & path, const server_http_context::handler_t & handler) const {
    auto [pattern, param_names] = path_to_regex(path_prefix + path);
    
    // Capture necessary data
    auto api_keys = pimpl->api_keys;
    auto public_endpoints = pimpl->public_endpoints;
    auto prefix = path_prefix;
    const server_http_context * parent = this;
    std::atomic<bool> * running_ptr = &pimpl->running;

    pimpl->srv->Get(pattern.c_str(), [handler, param_names, api_keys, public_endpoints, prefix, parent, running_ptr](const httplib_coro::Request & req, httplib_coro::Response & res) {
        // Check readiness (GET)
        if (!parent->is_ready.load()) {
            // Check if public endpoint
            bool is_public = public_endpoints.count(req.path) > 0;
            if (!is_public && !prefix.empty() && req.path.rfind(prefix, 0) == 0) {
                std::string stripped = req.path.substr(prefix.size());
                if (stripped.empty()) stripped = "/";
                is_public = public_endpoints.count(stripped) > 0;
            }
            if (!is_public) {
                res.status = 503;
                res.set_content(
                    safe_json_to_str(make_error_json("Loading model", 503, "unavailable_error")),
                    "application/json; charset=utf-8"
                );
                return;
            }
        }

        // Check API key
        if (!api_keys.empty()) {
            std::string req_api_key = req.get_header_value("Authorization");
            if (req_api_key.empty()) {
                req_api_key = req.get_header_value("X-Api-Key");
            }
            const std::string bearer = "Bearer ";
            if (req_api_key.rfind(bearer, 0) == 0) {
                req_api_key = req_api_key.substr(bearer.size());
            }

            bool valid = false;
            for (const auto & key : api_keys) {
                if (key == req_api_key) {
                    valid = true;
                    break;
                }
            }

            // Allow public endpoints without key
            bool is_public = public_endpoints.count(req.path) > 0;
            if (!is_public && !prefix.empty() && req.path.rfind(prefix, 0) == 0) {
                std::string stripped = req.path.substr(prefix.size());
                if (stripped.empty()) stripped = "/";
                is_public = public_endpoints.count(stripped) > 0;
            }

            if (!valid && !is_public) {
                res.status = 401;
                res.set_content(
                    safe_json_to_str(make_error_json("Invalid API Key", 401, "authentication_error")),
                    "application/json; charset=utf-8"
                );
                return;
            }
        }

        // Build server_http_req from httplib_coro::Request
        server_http_req request;

        // PSR-7 core fields
        request.method = req.method;
        request.request_target = req.target;
        request.path = req.path;
        request.body = req.body;

        // Extract protocol version from "HTTP/1.1" format
        if (req.version.size() > 5 && req.version.rfind("HTTP/", 0) == 0) {
            request.protocol_version = req.version.substr(5);
        }

        // Extract query string from target
        auto query_pos = req.target.find('?');
        if (query_pos != std::string::npos) {
            request.query_string = req.target.substr(query_pos + 1);
        }

        // PSR-7 server request fields
        request.remote_addr = req.remote_addr;
        request.remote_port = req.remote_port;

        // Parse host header for host/port
        std::string host_header = req.get_header_value("Host");
        if (!host_header.empty()) {
            auto colon_pos = host_header.find(':');
            if (colon_pos != std::string::npos) {
                request.host = host_header.substr(0, colon_pos);
                try {
                    request.port = std::stoi(host_header.substr(colon_pos + 1));
                } catch (...) {
                    request.port = 0;
                }
            } else {
                request.host = host_header;
            }
        }

        // Copy headers with multi-value support (PSR-7)
        for (const auto & [k, v] : req.headers) {
            request.headers[k].push_back(v);
        }

        // Copy params: path params first, then query params
        for (size_t i = 0; i < param_names.size() && i + 1 < req.matches.size(); i++) {
            request.params[param_names[i]] = req.matches[i + 1].str();
        }
        for (const auto & [k, v] : req.params) {
            request.params[k] = v;
        }

        auto should_stop_flag = std::make_shared<std::atomic<bool>>(false);
        request.should_stop = [should_stop_flag, running_ptr]() { return should_stop_flag->load() || !running_ptr->load(); };

        auto request_ptr = std::make_shared<server_http_req>(std::move(request));

        server_http_res_ptr response;
        try {
            response = handler(*request_ptr);
        } catch (const std::exception & e) {
            res.status = 500;
            res.set_content(
                safe_json_to_str(make_error_json(e.what(), 500, "internal_server_error")),
                "application/json; charset=utf-8"
            );
            return;
        }

        // Write multi-value headers (PSR-7)
        for (const auto & [k, values] : response->headers) {
            for (const auto & v : values) {
                res.set_header(k.c_str(), v.c_str());
            }
        }

        if (response->is_stream()) {
            res.status = response->status;
            auto resp_shared = std::shared_ptr<server_http_res>(std::move(response));
            res.set_chunked_content_provider(
                [resp_shared, request_ptr, should_stop_flag](size_t offset, httplib_coro::DataSink & sink) {
                    if (!sink.is_writable()) {
                        should_stop_flag->store(true);
                        return false;
                    }
                    std::string chunk;
                    bool has_next = resp_shared->next(chunk);
                    if (!chunk.empty()) {
                        sink.write(chunk.data(), chunk.size());
                        if (!sink.is_writable()) {
                            should_stop_flag->store(true);
                            return false;
                        }
                    }
                    if (!has_next) {
                        sink.done();
                    }
                    return has_next;
                }
            );
        } else {
            res.status = response->status;
            res.set_content(response->data, response->content_type.c_str());
        }
    });
}

void server_http_context::post(const std::string & path, const server_http_context::handler_t & handler) const {
    auto [pattern, param_names] = path_to_regex(path_prefix + path);
    
    // Capture necessary data
    auto api_keys = pimpl->api_keys;
    auto public_endpoints = pimpl->public_endpoints;
    auto prefix = path_prefix;
    const server_http_context * parent = this;
    std::atomic<bool> * running_ptr = &pimpl->running;

    pimpl->srv->Post(pattern.c_str(), [handler, param_names, api_keys, public_endpoints, prefix, parent, running_ptr](const httplib_coro::Request & req, httplib_coro::Response & res) {
        // Check readiness (POST)
        if (!parent->is_ready.load()) {
            bool is_public = public_endpoints.count(req.path) > 0;
            if (!is_public && !prefix.empty() && req.path.rfind(prefix, 0) == 0) {
                std::string stripped = req.path.substr(prefix.size());
                if (stripped.empty()) stripped = "/";
                is_public = public_endpoints.count(stripped) > 0;
            }
            if (!is_public) {
                res.status = 503;
                res.set_content(
                    safe_json_to_str(make_error_json("Loading model", 503, "unavailable_error")),
                    "application/json; charset=utf-8"
                );
                return;
            }
        }

        if (!api_keys.empty()) {
            std::string req_api_key = req.get_header_value("Authorization");
            if (req_api_key.empty()) {
                req_api_key = req.get_header_value("X-Api-Key");
            }
            const std::string bearer = "Bearer ";
            if (req_api_key.rfind(bearer, 0) == 0) {
                req_api_key = req_api_key.substr(bearer.size());
            }

            bool valid = false;
            for (const auto & key : api_keys) {
                if (key == req_api_key) {
                    valid = true;
                    break;
                }
            }

            bool is_public = public_endpoints.count(req.path) > 0;
            if (!is_public && !prefix.empty() && req.path.rfind(prefix, 0) == 0) {
                std::string stripped = req.path.substr(prefix.size());
                if (stripped.empty()) stripped = "/";
                is_public = public_endpoints.count(stripped) > 0;
            }

            if (!valid && !is_public) {
                res.status = 401;
                res.set_content(
                    safe_json_to_str(make_error_json("Invalid API Key", 401, "authentication_error")),
                    "application/json; charset=utf-8"
                );
                return;
            }
        }

        // Build server_http_req from httplib_coro::Request (PSR-7 compatible)
        server_http_req request;

        // PSR-7 core fields
        request.method = req.method;
        request.request_target = req.target;
        request.path = req.path;
        request.body = req.body;

        // Extract protocol version from "HTTP/1.1" format
        if (req.version.size() > 5 && req.version.rfind("HTTP/", 0) == 0) {
            request.protocol_version = req.version.substr(5);
        }

        // Extract query string from target
        auto query_pos = req.target.find('?');
        if (query_pos != std::string::npos) {
            request.query_string = req.target.substr(query_pos + 1);
        }

        // PSR-7 server request fields
        request.remote_addr = req.remote_addr;
        request.remote_port = req.remote_port;

        // Parse host header for host/port
        std::string host_header = req.get_header_value("Host");
        if (!host_header.empty()) {
            auto colon_pos = host_header.find(':');
            if (colon_pos != std::string::npos) {
                request.host = host_header.substr(0, colon_pos);
                try {
                    request.port = std::stoi(host_header.substr(colon_pos + 1));
                } catch (...) {
                    request.port = 0;
                }
            } else {
                request.host = host_header;
            }
        }

        // Copy headers with multi-value support (PSR-7)
        for (const auto & [k, v] : req.headers) {
            request.headers[k].push_back(v);
        }

        // Copy params: path params first, then query params
        for (size_t i = 0; i < param_names.size() && i + 1 < req.matches.size(); i++) {
            request.params[param_names[i]] = req.matches[i + 1].str();
        }
        for (const auto & [k, v] : req.params) {
            request.params[k] = v;
        }

        auto should_stop_flag = std::make_shared<std::atomic<bool>>(false);
        request.should_stop = [should_stop_flag, running_ptr]() { return should_stop_flag->load() || !running_ptr->load(); };

        auto request_ptr = std::make_shared<server_http_req>(std::move(request));

        server_http_res_ptr response;
        try {
            response = handler(*request_ptr);
        } catch (const std::exception & e) {
            res.status = 500;
            res.set_content(
                safe_json_to_str(make_error_json(e.what(), 500, "internal_server_error")),
                "application/json; charset=utf-8"
            );
            return;
        }

        // Write multi-value headers (PSR-7)
        for (const auto & [k, values] : response->headers) {
            for (const auto & v : values) {
                res.set_header(k.c_str(), v.c_str());
            }
        }

        if (response->is_stream()) {
            res.status = response->status;
            auto resp_shared = std::shared_ptr<server_http_res>(std::move(response));
            res.set_chunked_content_provider(
                [resp_shared, request_ptr, should_stop_flag](size_t offset, httplib_coro::DataSink & sink) {
                    if (!sink.is_writable()) {
                        should_stop_flag->store(true);
                        return false;
                    }
                    std::string chunk;
                    bool has_next = resp_shared->next(chunk);
                    if (!chunk.empty()) {
                        sink.write(chunk.data(), chunk.size());
                        if (!sink.is_writable()) {
                            should_stop_flag->store(true);
                            return false;
                        }
                    }
                    if (!has_next) {
                        sink.done();
                    }
                    return has_next;
                }
            );
        } else {
            res.status = response->status;
            res.set_content(response->data, response->content_type.c_str());
        }
    });
}
