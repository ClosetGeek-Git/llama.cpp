#include "common.h"
#include "server-http.h"
#include "server-common.h"

#include <zmq.h>

#include <functional>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <mutex>
#include <sstream>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

static std::vector<std::string> split_path(const std::string & p) {
    std::vector<std::string> out;
    size_t i = 0, n = p.size();
    while (i < n) {
        while (i < n && p[i] == '/') ++i;
        if (i >= n) break;
        size_t j = i;
        while (j < n && p[j] != '/') ++j;
        out.emplace_back(p.substr(i, j - i));
        i = j;
    }
    return out;
}

struct route_entry {
    std::string tmpl;                      // e.g. "/slots/:id_slot"
    server_http_context::handler_t handler;
    std::vector<std::string> parts;        // split tmpl parts
};

static bool match_route(const route_entry & r, const std::string & path, std::map<std::string, std::string> & out_params, const std::string & prefix) {
    // apply api prefix if any
    std::string p = path;
    if (!prefix.empty()) {
        if (p.rfind(prefix, 0) == 0) {
            p = p.substr(prefix.size());
            if (p.empty()) p = "/";
        } else {
            return false;
        }
    }
    auto path_parts = split_path(p);
    if (path_parts.size() != r.parts.size()) return false;

    for (size_t i = 0; i < r.parts.size(); ++i) {
        const auto & a = r.parts[i];
        const auto & b = path_parts[i];
        if (!a.empty() && a[0] == ':') {
            out_params[a.substr(1)] = b;
        } else if (a != b) {
            return false;
        }
    }
    return true;
}

class server_http_context::Impl {
public:
    void * zmq_ctx = nullptr;
    void * router  = nullptr;

    std::atomic<bool> running{false};

    // config copied from common_params
    std::string api_prefix;
    std::string bind_endpoint;
    std::vector<std::string> api_keys;

    // routes
    std::vector<route_entry> get_routes;
    std::vector<route_entry> post_routes;

    // middleware-like options
    std::unordered_set<std::string> public_endpoints{"/health", "/v1/health", "/models", "/v1/models", "/api/tags"};

    // register a route
    void add_route(std::vector<route_entry> & vec, const std::string & path, const handler_t & handler) {
        route_entry e;
        e.tmpl = path;
        e.handler = handler;
        e.parts = split_path(path);
        vec.emplace_back(std::move(e));
    }

    // find best matching route
    const handler_t * find_handler(const std::string & method, const std::string & path, std::map<std::string, std::string> & out_params) {
        const auto & vec = (method == "GET") ? get_routes : post_routes;
        for (const auto & r : vec) {
            out_params.clear();
            if (match_route(r, path, out_params, api_prefix)) {
                return &r.handler;
            }
        }
        return nullptr;
    }

    static std::string zmq_recv_last_frame(void * sock, std::vector<std::string> & ident) {
        ident.clear();
        printf("Waiting to receive ZMQ message...\n");
        std::string payload;
        while (true) {
            zmq_msg_t msg;
            zmq_msg_init(&msg);
            int rc = zmq_msg_recv(&msg, sock, 0);
            if (rc < 0) {
                zmq_msg_close(&msg);
                return {};
            }
            bool more = zmq_msg_more(&msg);

            if (more) {
                ident.emplace_back(static_cast<const char *>(zmq_msg_data(&msg)), static_cast<size_t>(rc));
            } else {
                payload.assign(static_cast<const char *>(zmq_msg_data(&msg)), static_cast<size_t>(rc));

                std::vector<char> inMsg(zmq_msg_size(&msg));
                std::memcpy(inMsg.data(), zmq_msg_data(&msg), zmq_msg_size(&msg));

                printf("ZMQ Payload: ");
                for (char c : inMsg) {
                    printf("%c", c); // Print each element as a character
                }
                printf("\n");
            }
            zmq_msg_close(&msg);
            if (!more) break;
        }
        // frames are: [identity frames ...] [payload]
        return payload;
    }

    static bool zmq_send_reply(void * sock, const std::vector<std::string> & ident, const std::string & payload) {
        // send back identity frames
        for (const auto & id : ident) {
            int flags = ZMQ_SNDMORE;
            if (&id == &ident.back()) {
                // last ident frame carries SNDMORE too (payload is next)
                flags = ZMQ_SNDMORE;
            }
            if (zmq_send(sock, id.data(), (int) id.size(), flags) < 0) return false;
        }
        // payload
        if (zmq_send(sock, payload.data(), (int) payload.size(), 0) < 0) return false;
        return true;
    }
};

server_http_context::server_http_context()
    : pimpl(std::make_unique<server_http_context::Impl>())
{}

server_http_context::~server_http_context() = default;

static json make_error_json(const std::string & message, int code, const std::string & type) {
    return json {
        {"error", {
            {"message", message},
            {"type", type},
            {"code", code}
        }}
    };
}

bool server_http_context::init(const common_params & params) {
    path_prefix = params.api_prefix;
    port = 8001;
    hostname = "127.0.0.1";

    pimpl->api_prefix = path_prefix;
    pimpl->api_keys   = params.api_keys;

    // build ZeroMQ endpoint
    bool is_sock = false;
    if (string_ends_with(std::string(hostname), ".sock")) {
        is_sock = true;
        pimpl->bind_endpoint = string_format("ipc://%s", hostname.c_str());
    } else {
        if (port == 0) {
            // choose a port automatically is non-trivial in ZMQ; keep provided port or default
            port = params.port == 0 ? 8080 : params.port;
        }
        pimpl->bind_endpoint = string_format("tcp://%s:%d", hostname.c_str(), port);
    }

    listening_address = pimpl->bind_endpoint;
    return true;
}

bool server_http_context::start() {
    pimpl->zmq_ctx = zmq_ctx_new();
    if (!pimpl->zmq_ctx) {
        LOG_ERR("%s: couldn't create ZMQ context\n", __func__);
        return false;
    }

    pimpl->router = zmq_socket(pimpl->zmq_ctx, ZMQ_ROUTER);
    if (!pimpl->router) {
        LOG_ERR("%s: couldn't create ZMQ ROUTER socket\n", __func__);
        return false;
    }

    int rc = zmq_bind(pimpl->router, pimpl->bind_endpoint.c_str());
    if (rc != 0) {
        LOG_ERR("%s: couldn't bind ZMQ endpoint: %s\n", __func__, pimpl->bind_endpoint.c_str());
        return false;
    }

    pimpl->running.store(true);
    thread = std::thread([this]() {
        SRV_INF("%s: ZMQ server listening on %s\n", __func__, pimpl->bind_endpoint.c_str());

        while (pimpl->running.load()) {
            std::vector<std::string> ident;
            std::string payload = Impl::zmq_recv_last_frame(pimpl->router, ident);
            if (payload.empty()) {
                if (!pimpl->running.load()) break;
                continue;
            }

            // parse request JSON
            json req_json;
            try {
                req_json = json::parse(payload);
            } catch (const std::exception & e) {
                auto err = make_error_json(std::string("Invalid JSON: ") + e.what(), 400, "invalid_request_error");
                Impl::zmq_send_reply(pimpl->router, ident, safe_json_to_str(err));
                continue;
            }

            // Expected request envelope:
            // { "id": "...", "method": "GET"|"POST", "path": "/...", "headers": {...}, "params": {...}, "body": "..." }
            const std::string method = req_json.value("method", "GET");
            const std::string path   = req_json.value("path", "/");
            const json & headers_j   = req_json.contains("headers") ? req_json["headers"] : json::object();
            const json & params_j    = req_json.contains("params")  ? req_json["params"]  : json::object();
            const std::string body   = req_json.value("body", std::string());

            // basic server readiness gating
            if (!is_ready.load()) {
                // allow public endpoints while loading
                auto no_prefix_path = path_prefix.empty() ? path : path.substr(path_prefix.size());
                if (pimpl->public_endpoints.count(path) == 0 && pimpl->public_endpoints.count(no_prefix_path) == 0) {
                    auto err = make_error_json("Loading model", 503, "unavailable_error");
                    Impl::zmq_send_reply(pimpl->router, ident, safe_json_to_str(err));
                    continue;
                }
            }

            // API key validation (optional)
            if (!pimpl->api_keys.empty()) {
                std::string req_api_key;
                if (headers_j.contains("Authorization")) {
                    req_api_key = headers_j["Authorization"].get<std::string>();
                    const std::string bearer = "Bearer ";
                    if (req_api_key.rfind(bearer, 0) == 0) {
                        req_api_key = req_api_key.substr(bearer.size());
                    }
                } else if (headers_j.contains("X-Api-Key")) {
                    req_api_key = headers_j["X-Api-Key"].get<std::string>();
                }
                bool ok = false;
                for (auto & k : pimpl->api_keys) if (k == req_api_key) { ok = true; break; }

                // allow public endpoints w/o key
                auto no_prefix_path = path_prefix.empty() ? path : path.substr(path_prefix.size());
                if (!ok && pimpl->public_endpoints.count(path) == 0 && pimpl->public_endpoints.count(no_prefix_path) == 0) {
                    auto err = make_error_json("Invalid API Key", 401, "authentication_error");
                    Impl::zmq_send_reply(pimpl->router, ident, safe_json_to_str(err));
                    continue;
                }
            }

            // find handler
            std::map<std::string, std::string> path_params;
            const handler_t * handler = pimpl->find_handler(method, path, path_params);

            if (!handler) {
                auto err = make_error_json("File Not Found", 404, "not_found_error");
                Impl::zmq_send_reply(pimpl->router, ident, safe_json_to_str(err));
                continue;
            }

            // build request object
            std::map<std::string, std::string> headers;
            for (const auto & kv : headers_j.items()) {
                headers[kv.key()] = kv.value().get<std::string>();
            }
            std::map<std::string, std::string> params = path_params;
            for (const auto & kv : params_j.items()) {
                params[kv.key()] = kv.value().is_string() ? kv.value().get<std::string>() : kv.value().dump();
            }

            std::atomic<bool> stop_flag{false};
            auto should_stop = [&stop_flag]() { return stop_flag.load(); };

            server_http_res_ptr response;
            try {
                response = (*handler)(server_http_req{
                    params,
                    headers,
                    path,
                    body,
                    should_stop
                });
            } catch (const std::exception & e) {
                auto err = make_error_json(e.what(), 500, "internal_server_error");
                Impl::zmq_send_reply(pimpl->router, ident, safe_json_to_str(err));
                continue;
            }

            // convert response to JSON envelope
            json res_j;
            res_j["status"]       = response->status;
            res_j["content_type"] = response->content_type;
            res_j["headers"]      = response->headers;

            if (response->is_stream()) {
                // consume streaming chunks into one combined payload
                std::string all;
                std::string chunk;
                while (response->next(chunk)) {
                    if (!chunk.empty()) {
                        all.append(chunk);
                    }
                }
                res_j["data"]   = all;
                res_j["stream"] = true;
            } else {
                res_j["data"]   = response->data;
                res_j["stream"] = false;
            }

            Impl::zmq_send_reply(pimpl->router, ident, safe_json_to_str(res_j));
        }
    });

    return true;
}

void server_http_context::stop() const {
    if (pimpl->running.exchange(false)) {
        if (pimpl->router) {
            zmq_close(pimpl->router);
            pimpl->router = nullptr;
        }
        if (pimpl->zmq_ctx) {
            zmq_ctx_term(pimpl->zmq_ctx);
            pimpl->zmq_ctx = nullptr;
        }
    }
}

void server_http_context::get(const std::string & path, const server_http_context::handler_t & handler) const {
    pimpl->add_route(pimpl->get_routes, path, handler);
}

void server_http_context::post(const std::string & path, const server_http_context::handler_t & handler) const {
    pimpl->add_route(pimpl->post_routes, path, handler);
}

