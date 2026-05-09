#include "common.h"
#include "server-http.h"
#include "server-common.h"
#include "server-context.h"

// Swoole headers
#include "swoole.h"
#include "swoole_coroutine_api.h"
#include "swoole_signal.h"

#include "httplib_server.h"

#include <cstdlib>
#include <functional>
#include <future>
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

<<<<<<< ours (server-coro)
static json make_error_json(const std::string & message, int code, const std::string & type) {
    return json {
        {"error", {
            {"message", message},
            {"type", type},
            {"code", code}
        }}
    };
}
||||||| base (tools/server@fork)
// auto generated files (see README.md for details)
#include "index.html.gz.hpp"
#include "loading.html.hpp"
=======
#ifdef LLAMA_BUILD_WEBUI
// auto generated files (see README.md for details)
#include "index.html.hpp"
#include "bundle.js.hpp"
#include "bundle.css.hpp"
#include "loading.html.hpp"
#endif
>>>>>>> theirs (tools/server@upstream)

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

// For Google Cloud Platform deployment compatibility
struct gcp_params {
    bool enabled;
    std::string path_health;
    std::string path_predict;
    int port;

    // Ref: https://docs.cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables
    gcp_params() {
        enabled = getenv("AIP_MODE", "") == "PREDICTION";
        path_health = getenv("AIP_HEALTH_ROUTE", "", true); // default: using the route defined in server.cpp
        path_predict = getenv("AIP_PREDICT_ROUTE", "/predict", true);
        port = std::stoi(getenv("AIP_HTTP_PORT", "8080"));
    }

    static std::string getenv(const char * name, const std::string & default_value, bool ensure_leading_slash = false) {
        const char * value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return default_value;
        }
        std::string val = value;
        if (ensure_leading_slash && !val.empty() && val[0] != '/') {
            val.insert(val.begin(), '/');
        }
        return val;
    }
};

bool server_http_context::init(const common_params & params) {
    const gcp_params gcp;

    path_prefix = params.api_prefix;
    port = params.port;
    hostname = params.hostname;

<<<<<<< ours (server-coro)
    pimpl->api_keys = params.api_keys;
    pimpl->parent = this;
||||||| base (tools/server@fork)
    auto & srv = pimpl->srv;

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_INF("Running with SSL: key = %s, cert = %s\n", params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
        srv.reset(
            new httplib::SSLServer(params.ssl_file_cert.c_str(), params.ssl_file_key.c_str())
        );
    } else {
        LOG_INF("Running without SSL\n");
        srv.reset(new httplib::Server());
    }
#else
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_ERR("Server is built without SSL support\n");
        return false;
    }
    srv.reset(new httplib::Server());
#endif
=======
    if (gcp.enabled) {
        LOG_INF("%s: Google Cloud Platform compat: health route = %s, predict route = %s, port = %d\n", __func__, gcp.path_health.c_str(), gcp.path_predict.c_str(), gcp.port);

        if (port != gcp.port) {
            LOG_WRN("%s: Google Cloud Platform compat: overriding server port %d with AIP_HTTP_PORT %d\n", __func__, port, gcp.port);
        }

        port = gcp.port;
    }

    auto & srv = pimpl->srv;

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_INF("Running with SSL: key = %s, cert = %s\n", params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
        srv.reset(
            new httplib::SSLServer(params.ssl_file_cert.c_str(), params.ssl_file_key.c_str())
        );
    } else {
        LOG_INF("Running without SSL\n");
        srv.reset(new httplib::Server());
    }
#else
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_ERR("Server is built without SSL support\n");
        return false;
    }
    srv.reset(new httplib::Server());
#endif
>>>>>>> theirs (tools/server@upstream)

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
    srv->set_socket_options([reuse_port = params.reuse_port](socket_t sock) {
        httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1);
        if (reuse_port) {
#ifdef SO_REUSEPORT
            httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEPORT, 1);
#else
            LOG_WRN("%s: SO_REUSEPORT is not supported\n", __func__);
#endif
        }
    });

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

<<<<<<< ours (server-coro)
    if (g_http_context) {
        // Stop the HTTP server (safe - we're in reactor context)
        g_http_context->stop();
||||||| base (tools/server@fork)
    auto middleware_validate_api_key = [api_keys = params.api_keys](const httplib::Request & req, httplib::Response & res) {
        static const std::unordered_set<std::string> public_endpoints = {
            "/health",
            "/v1/health",
            "/models",
            "/v1/models",
            "/api/tags"
        };
=======
    auto middleware_validate_api_key = [api_keys = params.api_keys](const httplib::Request & req, httplib::Response & res) {
        static const std::unordered_set<std::string> public_endpoints = {
            "/health",
            "/v1/health",
            "/models",
            "/v1/models",
            "/",
            "/index.html",
            "/bundle.js",
            "/bundle.css",
        };
>>>>>>> theirs (tools/server@upstream)

        // Notify main thread to exit start_loop()
        if (g_http_context->on_shutdown) {
            g_http_context->on_shutdown();
        }
    }
}

<<<<<<< ours (server-coro)
bool server_http_context::start() {
    auto & srv = pimpl->srv;
||||||| base (tools/server@fork)
        // If path is public or is static file, skip validation
        if (public_endpoints.find(req.path) != public_endpoints.end() || req.path == "/") {
            return true;
        }
=======
        // If path is public or static file, skip validation
        if (public_endpoints.find(req.path) != public_endpoints.end()) {
            return true;
        }
>>>>>>> theirs (tools/server@upstream)

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

<<<<<<< ours (server-coro)
            if (!valid && !is_public) {
                res.status = 401;
||||||| base (tools/server@fork)
    auto middleware_server_state = [this](const httplib::Request & req, httplib::Response & res) {
        bool ready = is_ready.load();
        if (!ready) {
            auto tmp = string_split<std::string>(req.path, '.');
            if (req.path == "/" || tmp.back() == "html") {
                res.status = 503;
                res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len, "text/html; charset=utf-8");
            } else {
                // no endpoints is allowed to be accessed when the server is not ready
                // this is to prevent any data races or inconsistent states
                res.status = 503;
=======
    auto middleware_server_state = [this](const httplib::Request & req, httplib::Response & res) {
        bool ready = is_ready.load();
        if (!ready) {
#ifdef LLAMA_BUILD_WEBUI
            auto tmp = string_split<std::string>(req.path, '.');
            if (req.path == "/" || tmp.back() == "html") {
                res.status = 503;
                res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len, "text/html; charset=utf-8");
            } else
#endif
            {
                // no endpoints is allowed to be accessed when the server is not ready
                // this is to prevent any data races or inconsistent states
                res.status = 503;
>>>>>>> theirs (tools/server@upstream)
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

<<<<<<< ours (server-coro)
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
||||||| base (tools/server@fork)
    int n_threads_http = params.n_threads_http;
    if (n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    LOG_INF("%s: using %d threads for HTTP server\n", __func__, n_threads_http);
    srv->new_task_queue = [n_threads_http] { return new httplib::ThreadPool(n_threads_http); };

    //
    // Web UI setup
    //

    if (!params.webui) {
        LOG_INF("Web UI is disabled\n");
    } else {
        // register static assets routes
        if (!params.public_path.empty()) {
            // Set the base directory for serving static files
            bool is_found = srv->set_mount_point(params.api_prefix + "/", params.public_path);
            if (!is_found) {
                LOG_ERR("%s: static assets path not found: %s\n", __func__, params.public_path.c_str());
                return 1;
=======
    int n_threads_http = params.n_threads_http;
    if (n_threads_http < 1) {
        // +4 threads for monitoring, health and some threads reserved for MCP and other tasks in the future
        n_threads_http = std::max(params.n_parallel + 4, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    LOG_INF("%s: using %d threads for HTTP server\n", __func__, n_threads_http);
    srv->new_task_queue = [n_threads_http] {
        // spawn n_threads_http fixed thread (always alive), while allow up to 1024 max possible additional threads
        // when n_threads_http is used, server will create new "dynamic" threads that will be destroyed after processing each request
        // ref: https://github.com/yhirose/cpp-httplib/pull/2368
        size_t max_threads = (size_t)n_threads_http + 1024;
        return new httplib::ThreadPool(n_threads_http, max_threads);
    };

    //
    // Web UI setup
    //

    if (!params.webui) {
        LOG_INF("Web UI is disabled\n");
    } else {
        // register static assets routes
        if (!params.public_path.empty()) {
            // Set the base directory for serving static files
            bool is_found = srv->set_mount_point(params.api_prefix + "/", params.public_path);
            if (!is_found) {
                LOG_ERR("%s: static assets path not found: %s\n", __func__, params.public_path.c_str());
                return 1;
>>>>>>> theirs (tools/server@upstream)
            }
<<<<<<< ours (server-coro)
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
||||||| base (tools/server@fork)
        } else {
            // using embedded static index.html
            srv->Get(params.api_prefix + "/", [](const httplib::Request & req, httplib::Response & res) {
                if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                    res.set_content("Error: gzip is not supported by this browser", "text/plain");
                } else {
                    res.set_header("Content-Encoding", "gzip");
                    // COEP and COOP headers, required by pyodide (python interpreter)
                    res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
                    res.set_header("Cross-Origin-Opener-Policy", "same-origin");
                    res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
                }
                return false;
            });
=======
        } else {
#ifdef LLAMA_BUILD_WEBUI
            // using embedded static index.html
            srv->Get(params.api_prefix + "/", [](const httplib::Request & /*req*/, httplib::Response & res) {
                // COEP and COOP headers, required by pyodide (python interpreter)
                res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
                res.set_header("Cross-Origin-Opener-Policy", "same-origin");
                res.set_content(reinterpret_cast<const char*>(index_html), index_html_len, "text/html; charset=utf-8");
                return false;
            });
            srv->Get(params.api_prefix + "/bundle.js", [](const httplib::Request & /*req*/, httplib::Response & res) {
                res.set_content(reinterpret_cast<const char*>(bundle_js), bundle_js_len, "application/javascript; charset=utf-8");
                return false;
            });
            srv->Get(params.api_prefix + "/bundle.css", [](const httplib::Request & /*req*/, httplib::Response & res) {
                res.set_content(reinterpret_cast<const char*>(bundle_css), bundle_css_len, "text/css; charset=utf-8");
                return false;
            });
#endif
>>>>>>> theirs (tools/server@upstream)
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

<<<<<<< ours (server-coro)
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
||||||| base (tools/server@fork)
// using unique_ptr for request to allow safe capturing in lambdas
using server_http_req_ptr = std::unique_ptr<server_http_req>;

static void process_handler_response(server_http_req_ptr && request, server_http_res_ptr & response, httplib::Response & res) {
    if (response->is_stream()) {
        res.status = response->status;
        set_headers(res, response->headers);
        std::string content_type = response->content_type;
        // convert to shared_ptr as both chunked_content_provider() and on_complete() need to use it
        std::shared_ptr<server_http_req> q_ptr = std::move(request);
        std::shared_ptr<server_http_res> r_ptr = std::move(response);
        const auto chunked_content_provider = [response = r_ptr](size_t, httplib::DataSink & sink) -> bool {
            std::string chunk;
            bool has_next = response->next(chunk);
            if (!chunk.empty()) {
                // TODO: maybe handle sink.write unsuccessful? for now, we rely on is_connection_closed()
                sink.write(chunk.data(), chunk.size());
                SRV_DBG("http: streamed chunk: %s\n", chunk.c_str());
            }
            if (!has_next) {
                sink.done();
                SRV_DBG("%s", "http: stream ended\n");
=======
static std::string build_query_string(const httplib::Request & req) {
    std::string qs;
    for (const auto & [key, value] : req.params) {
        if (!qs.empty()) {
            qs += '&';
        }
        qs += httplib::encode_query_component(key) + "=" + httplib::encode_query_component(value);
    }
    return qs;
}

// using unique_ptr for request to allow safe capturing in lambdas
using server_http_req_ptr = std::unique_ptr<server_http_req>;

static void process_handler_response(server_http_req_ptr && request, server_http_res_ptr & response, httplib::Response & res) {
    if (response->is_stream()) {
        res.status = response->status;
        set_headers(res, response->headers);
        std::string content_type = response->content_type;
        // convert to shared_ptr as both chunked_content_provider() and on_complete() need to use it
        std::shared_ptr<server_http_req> q_ptr = std::move(request);
        std::shared_ptr<server_http_res> r_ptr = std::move(response);
        const auto chunked_content_provider = [response = r_ptr](size_t, httplib::DataSink & sink) -> bool {
            std::string chunk;
            bool has_next = response->next(chunk);
            if (!chunk.empty()) {
                if (!sink.write(chunk.data(), chunk.size())) {
                    return false;
                }
                SRV_DBG("http: streamed chunk: %s\n", chunk.c_str());
            }
            if (!has_next) {
                sink.done();
                SRV_DBG("%s", "http: stream ended\n");
>>>>>>> theirs (tools/server@upstream)
            }
        }

<<<<<<< ours (server-coro)
        // Copy headers with multi-value support (PSR-7)
        for (const auto & [k, v] : req.headers) {
            request.headers[k].push_back(v);
        }
||||||| base (tools/server@fork)
void server_http_context::get(const std::string & path, const server_http_context::handler_t & handler) const {
    pimpl->srv->Get(path_prefix + path, [handler](const httplib::Request & req, httplib::Response & res) {
        server_http_req_ptr request = std::make_unique<server_http_req>(server_http_req{
            get_params(req),
            get_headers(req),
            req.path,
            req.body,
            req.is_connection_closed
        });
        server_http_res_ptr response = handler(*request);
        process_handler_response(std::move(request), response, res);
    });
}
=======
void server_http_context::get(const std::string & path, const server_http_context::handler_t & handler) const {
    handlers.emplace(path, handler);
    pimpl->srv->Get(path_prefix + path, [handler](const httplib::Request & req, httplib::Response & res) {
        server_http_req_ptr request = std::make_unique<server_http_req>(server_http_req{
            get_params(req),
            get_headers(req),
            req.path,
            build_query_string(req),
            req.body,
            {},
            req.is_connection_closed
        });
        server_http_res_ptr response = handler(*request);
        process_handler_response(std::move(request), response, res);
    });
}
>>>>>>> theirs (tools/server@upstream)

<<<<<<< ours (server-coro)
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
||||||| base (tools/server@fork)
void server_http_context::post(const std::string & path, const server_http_context::handler_t & handler) const {
    pimpl->srv->Post(path_prefix + path, [handler](const httplib::Request & req, httplib::Response & res) {
        server_http_req_ptr request = std::make_unique<server_http_req>(server_http_req{
            get_params(req),
            get_headers(req),
            req.path,
            req.body,
            req.is_connection_closed
        });
        server_http_res_ptr response = handler(*request);
        process_handler_response(std::move(request), response, res);
=======
void server_http_context::post(const std::string & path, const server_http_context::handler_t & handler) const {
    handlers.emplace(path, handler);
    pimpl->srv->Post(path_prefix + path, [handler](const httplib::Request & req, httplib::Response & res) {
        std::string body = req.body;
        std::map<std::string, uploaded_file> files;

        if (req.is_multipart_form_data()) {
            // translate text fields to a JSON object and use it as the body
            json form_json = json::object();
            for (const auto & [key, field] : req.form.fields) {
                if (form_json.contains(key)) {
                    // if the key already exists, convert it to an array
                    if (!form_json[key].is_array()) {
                        json existing_value = form_json[key];
                        form_json[key] = json::array({existing_value});
                    }
                    form_json[key].push_back(field.content);
                } else {
                    form_json[key] = field.content;
                }
            }
            body = form_json.dump();

            // populate files from multipart form
            for (const auto & [key, file] : req.form.files) {
                files[key] = uploaded_file{
                    raw_buffer(file.content.begin(), file.content.end()),
                    file.filename,
                    file.content_type,
                };
            }
        }

        server_http_req_ptr request = std::make_unique<server_http_req>(server_http_req{
            get_params(req),
            get_headers(req),
            req.path,
            build_query_string(req),
            body,
            std::move(files),
            req.is_connection_closed
        });
        server_http_res_ptr response = handler(*request);
        process_handler_response(std::move(request), response, res);
>>>>>>> theirs (tools/server@upstream)
    });
}
<<<<<<< ours (server-coro)
||||||| base (tools/server@fork)

=======

//
// Vertex AI Prediction protocol (AIP_PREDICT_ROUTE)
// https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements
//

// Derives the camelCase @requestFormat alias for a registered path.
// e.g. "/v1/chat/completions" -> "chatCompletions", "/apply-template" -> "applyTemplate"
static std::string path_to_gcp_format(const std::string & path) {
    std::string s = path;
    if (s.size() > 3 && s[0] == '/' && s[1] == 'v' && s[2] == '1') {
        s = s.substr(3);
    }
    if (!s.empty() && s[0] == '/') {
        s = s.substr(1);
    }
    std::string result;
    bool cap = false;
    for (unsigned char c : s) {
        if (c == ':') break; // stop before path parameters
        if (c == '/' || c == '-' || c == '_') {
            cap = true;
        } else {
            result += cap ? (char)std::toupper(c) : (char)c;
            cap = false;
        }
    }
    return result;
}

static json parse_gcp_predict_response(const server_http_res_ptr & res) {
    if (res == nullptr) {
        throw std::runtime_error("empty response from internal handler");
    }
    if (res->is_stream()) {
        throw std::invalid_argument("predict route does not support streaming responses");
    }
    if (res->data.empty()) {
        return nullptr;
    }
    try {
        return json::parse(res->data);
    } catch (...) {
        return res->data;
    }
}

void server_http_context::register_gcp_compat() {
    const gcp_params gcp;

    if (!gcp.enabled) {
        // do nothing
        return;
    }

    if (handlers.count(gcp.path_predict)) {
        LOG_ERR("%s: AIP_PREDICT_ROUTE=%s conflicts with an existing llama-server route\n", __func__, gcp.path_predict.c_str());
        exit(1);
    }

    // camelCase alias -> canonical path (first registration wins on collision)
    // e.g. "chatCompletions" -> "/v1/chat/completions"
    std::unordered_map<std::string, std::string> alias_to_path;
    for (const auto & [path, _] : handlers) {
        alias_to_path.emplace(path_to_gcp_format(path), path);
    }

    if (!gcp.path_health.empty()) {
        auto health_handler = handlers.find("/health");
        GGML_ASSERT(health_handler != handlers.end());
        get(gcp.path_health, health_handler->second);
    }

    post(gcp.path_predict, [this, alias_to_path = std::move(alias_to_path)](const server_http_req & req) -> server_http_res_ptr {
        static const auto build_error = [](const std::string & message, error_type type) -> json {
            return json {{"error", format_error_response(message, type)}};
        };

        json data;
        try {
            data = json::parse(req.body);
        } catch (const std::exception & e) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = safe_json_to_str({{"error", format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST)}});
            return res;
        }
        if (!data.is_object()) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = safe_json_to_str({{"error", format_error_response("request body must be a JSON object", ERROR_TYPE_INVALID_REQUEST)}});
            return res;
        }
        if (!data.contains("instances") || !data.at("instances").is_array()) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = safe_json_to_str({{"error", format_error_response("request body must include an array field named instances", ERROR_TYPE_INVALID_REQUEST)}});
            return res;
        }

        const json & instances = data.at("instances");
        static const size_t MAX_INSTANCES = 128;
        if (instances.size() > MAX_INSTANCES) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = safe_json_to_str({{"error", format_error_response("instances array exceeds maximum size of " + std::to_string(MAX_INSTANCES), ERROR_TYPE_INVALID_REQUEST)}});
            return res;
        }

        std::vector<std::future<json>> futures;
        futures.reserve(instances.size());

        for (const auto & instance : instances) {
            futures.push_back(std::async(std::launch::async, [this, &req, &alias_to_path, instance]() -> json {
                if (!instance.is_object()) {
                    return build_error("each instance must be a JSON object", ERROR_TYPE_INVALID_REQUEST);
                }
                if (!instance.contains("@requestFormat") || !instance.at("@requestFormat").is_string()) {
                    return build_error("each instance must include a string @requestFormat", ERROR_TYPE_INVALID_REQUEST);
                }

                try {
                    json payload = instance;
                    const std::string format = payload.at("@requestFormat").get<std::string>();
                    payload.erase("@requestFormat");

                    if (payload.contains("stream")) {
                        LOG_WRN("%s: ignoring client-provided stream field in instance, streaming is not supported in predict route\n", __func__);
                        payload["stream"] = false;
                    }

                    // accept both camelCase aliases (e.g. "chatCompletions") and direct paths
                    std::string dispatch_path;
                    auto it_alias = alias_to_path.find(format);
                    if (it_alias != alias_to_path.end()) {
                        dispatch_path = it_alias->second;
                    } else if (handlers.count(format)) {
                        dispatch_path = format;
                    } else {
                        return build_error("no handler registered for @requestFormat: " + format, ERROR_TYPE_INVALID_REQUEST);
                    }

                    const server_http_req internal_req {
                        req.params,
                        req.headers,
                        path_prefix + dispatch_path,
                        req.query_string,
                        payload.dump(),
                        {},
                        req.should_stop,
                    };

                    server_http_res_ptr internal_res = handlers.at(dispatch_path)(internal_req);
                    return parse_gcp_predict_response(internal_res);
                } catch (const std::invalid_argument & e) {
                    return build_error(e.what(), ERROR_TYPE_INVALID_REQUEST);
                } catch (const std::exception & e) {
                    return build_error(e.what(), ERROR_TYPE_SERVER);
                } catch (...) {
                    return build_error("unknown error", ERROR_TYPE_SERVER);
                }
            }));
        }

        json predictions = json::array();
        for (auto & future : futures) {
            predictions.push_back(future.get());
        }

        auto res = std::make_unique<server_http_res>();
        res->data = safe_json_to_str({{"predictions", predictions}});
        return res;
    });
}
>>>>>>> theirs (tools/server@upstream)
