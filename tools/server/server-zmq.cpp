#include "server-zmq.h"
#include "server-common.h"
#include "server-context.h"

#include "common.h"
#include "log.h"

#include <zmq.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

//
// Helpers
//

// Pre-extract the "id" field from a raw payload via a tiny regex, BEFORE
// json::parse runs. This is what lets us echo `rid` in 400 error envelopes
// when parse itself fails — otherwise the client cannot correlate the error
// with its in-flight request.
static std::string extract_rid_from_raw(const std::string & raw) {
    static const std::regex rid_re(R"RX("id"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)")RX");
    std::smatch m;
    if (std::regex_search(raw, m, rid_re)) {
        return m[1].str();
    }
    return std::string{};
}

static json make_error_json(const std::string & message,
                            int code,
                            const std::string & type,
                            const std::string & rid) {
    return json{
        {"status", code},
        {"stream", false},
        {"rid",    rid},
        {"data",   safe_json_to_str(json{{
            "error", {
                {"message", message},
                {"type",    type},
                {"code",    code},
            }
        }})},
        {"content_type", "application/json; charset=utf-8"},
    };
}

// ROUTER envelope shape: [client_id, payload]. Returns false on terminal recv
// failure (e.g. ETERM during shutdown). Empty client_id with success means
// 0-byte routing prefix — unusual but tolerated.
static bool recv_envelope(void * sock, std::string & client_id, std::string & payload) {
    client_id.clear();
    payload.clear();

    zmq_msg_t id;
    zmq_msg_init(&id);
    int rc = zmq_msg_recv(&id, sock, 0);
    if (rc < 0) {
        zmq_msg_close(&id);
        return false;
    }
    client_id.assign(static_cast<const char *>(zmq_msg_data(&id)), static_cast<size_t>(rc));
    zmq_msg_close(&id);

    zmq_msg_t pl;
    zmq_msg_init(&pl);
    rc = zmq_msg_recv(&pl, sock, 0);
    if (rc < 0) {
        zmq_msg_close(&pl);
        return false;
    }
    payload.assign(static_cast<const char *>(zmq_msg_data(&pl)), static_cast<size_t>(rc));
    zmq_msg_close(&pl);
    return true;
}

static bool send_envelope(void * sock, const std::string & client_id, const std::string & payload) {
    if (zmq_send(sock, client_id.data(), (int)client_id.size(), ZMQ_SNDMORE) < 0) return false;
    if (zmq_send(sock, payload.data(), (int)payload.size(), 0) < 0) return false;
    return true;
}

// Send one multipart message comprising: client_id + header + each chunk + empty terminator.
// All frames except the last carry ZMQ_SNDMORE.
//
// We thread each piece through zmq_send individually but in a single
// "message" semantically (no other thread can interleave on this DEALER socket
// — the worker owns it). The receiver iterates by checking RCVMORE.
static bool send_stream_open(void * sock, const std::string & client_id, const std::string & header_json) {
    if (zmq_send(sock, client_id.data(), (int)client_id.size(), ZMQ_SNDMORE) < 0) return false;
    if (zmq_send(sock, header_json.data(), (int)header_json.size(), ZMQ_SNDMORE) < 0) return false;
    return true;
}

static bool send_stream_chunk(void * sock, const std::string & chunk) {
    if (zmq_send(sock, chunk.data(), (int)chunk.size(), ZMQ_SNDMORE) < 0) return false;
    return true;
}

static bool send_stream_close(void * sock) {
    // empty terminator, no SNDMORE
    if (zmq_send(sock, "", 0, 0) < 0) return false;
    return true;
}

//
// Impl
//

class server_zmq_context::Impl {
public:
    void * zmq_ctx  = nullptr;
    void * frontend = nullptr; // ROUTER (one bound per --zmq-bind)
    void * backend  = nullptr; // DEALER (inproc://llama_backend)

    std::atomic<bool> running{false};

    // config
    std::string              api_prefix;
    std::vector<std::string> api_keys;
    std::unordered_set<std::string> public_endpoints{
        "/health", "/v1/health", "/models", "/v1/models", "/api/tags"
    };

    // route table — registered before start()
    struct route {
        std::string method;
        std::string path;
        server_transport_handler_t handler;
    };
    std::vector<route> routes;

    // workers
    std::vector<std::thread> workers;
    std::thread proxy_thread;
    int n_workers = 0;
    int hwm       = 64;

    // in-flight cancellation registry. Each running request registers its
    // shared_ptr<atomic<bool>> under its rid; a CANCEL envelope flips the bool.
    std::mutex in_flight_mu;
    std::unordered_map<std::string, std::shared_ptr<std::atomic<bool>>> in_flight;

    const route * find_handler(const std::string & method,
                               const std::string & path,
                               std::map<std::string, std::string> & out_params) {
        for (const auto & r : routes) {
            if (r.method != method) continue;
            std::map<std::string, std::string> caps;
            if (match_route_template(r.path, path, caps, api_prefix)) {
                out_params.insert(caps.begin(), caps.end());
                return &r;
            }
        }
        return nullptr;
    }

    bool api_key_ok(const std::map<std::string, std::string> & headers, const std::string & path) {
        if (api_keys.empty()) return true;
        auto stripped = api_prefix.empty() ? path : (path.rfind(api_prefix, 0) == 0 ? path.substr(api_prefix.size()) : path);
        if (public_endpoints.count(path) || public_endpoints.count(stripped) || path == "/") return true;
        std::string key;
        if (auto it = headers.find("Authorization"); it != headers.end()) {
            key = it->second;
            static const std::string bearer = "Bearer ";
            if (key.rfind(bearer, 0) == 0) key = key.substr(bearer.size());
        } else if (auto it = headers.find("X-Api-Key"); it != headers.end()) {
            key = it->second;
        }
        for (const auto & k : api_keys) {
            if (k == key) return true;
        }
        return false;
    }

    void run_worker(server_zmq_context & owner, int idx) {
        void * worker = zmq_socket(zmq_ctx, ZMQ_DEALER);
        if (!worker) {
            LOG_ERR("zmq worker[%d]: socket() failed\n", idx);
            return;
        }
        // worker identity for logs (libzmq will use it if we ever switch from
        // inproc to direct routing)
        std::string wid = "worker-" + std::to_string(idx);
        zmq_setsockopt(worker, ZMQ_IDENTITY, wid.data(), (int)wid.size());

        // HWM and linger tuning — zmq_ctx_term must not block at shutdown
        int linger = 0;
        zmq_setsockopt(worker, ZMQ_LINGER, &linger, sizeof(linger));
        zmq_setsockopt(worker, ZMQ_SNDHWM, &hwm,    sizeof(hwm));
        zmq_setsockopt(worker, ZMQ_RCVHWM, &hwm,    sizeof(hwm));

        if (zmq_connect(worker, "inproc://llama_backend") != 0) {
            LOG_ERR("zmq worker[%d]: connect(inproc) failed: %s\n", idx, zmq_strerror(errno));
            zmq_close(worker);
            return;
        }
        LOG_INF("zmq worker[%d]: started\n", idx);

        while (running.load()) {
            std::string client_id;
            std::string payload;
            if (!recv_envelope(worker, client_id, payload)) {
                if (!running.load()) break;
                // recv failure during normal operation — continue rather than
                // burn the worker
                continue;
            }

            // Pre-extract rid from raw payload so error envelopes for parse
            // failures still echo the client's correlation id.
            const std::string early_rid = extract_rid_from_raw(payload);

            json req_json;
            try {
                req_json = json::parse(payload);
            } catch (const std::exception & e) {
                std::string err_msg = std::string("Invalid JSON: ") + e.what();
                json env = make_error_json(err_msg, 400, "invalid_request_error", early_rid);
                send_envelope(worker, client_id, safe_json_to_str(env));
                continue;
            }

            const std::string method = req_json.value("method", "GET");
            const std::string path   = req_json.value("path", "/");
            const std::string rid    = req_json.value("id", early_rid);

            // CANCEL is a one-way control envelope: flip the in-flight bool
            // for the named rid and return no response.
            if (method == "CANCEL") {
                std::lock_guard<std::mutex> lk(in_flight_mu);
                auto it = in_flight.find(rid);
                if (it != in_flight.end()) {
                    it->second->store(true, std::memory_order_relaxed);
                    LOG_INF("zmq worker[%d]: cancel rid=%s\n", idx, rid.c_str());
                }
                continue;
            }

            // Readiness gate — refuse non-public traffic until is_ready flips
            if (!owner.is_ready.load(std::memory_order_acquire)) {
                auto stripped = api_prefix.empty() ? path : (path.rfind(api_prefix, 0) == 0 ? path.substr(api_prefix.size()) : path);
                if (!public_endpoints.count(path) && !public_endpoints.count(stripped)) {
                    json env = make_error_json("Loading model", 503, "unavailable_error", rid);
                    send_envelope(worker, client_id, safe_json_to_str(env));
                    continue;
                }
            }

            // Headers / params from JSON
            std::map<std::string, std::string> headers;
            if (req_json.contains("headers") && req_json["headers"].is_object()) {
                for (const auto & kv : req_json["headers"].items()) {
                    headers[kv.key()] = kv.value().is_string() ? kv.value().get<std::string>() : kv.value().dump();
                }
            }
            std::map<std::string, std::string> params;
            if (req_json.contains("params") && req_json["params"].is_object()) {
                for (const auto & kv : req_json["params"].items()) {
                    params[kv.key()] = kv.value().is_string() ? kv.value().get<std::string>() : kv.value().dump();
                }
            }
            const std::string body = req_json.value("body", std::string{});

            // API key check
            if (!api_key_ok(headers, path)) {
                json env = make_error_json("Invalid API Key", 401, "authentication_error", rid);
                send_envelope(worker, client_id, safe_json_to_str(env));
                continue;
            }

            // Route lookup
            std::map<std::string, std::string> path_params;
            const auto * matched = find_handler(method, path, path_params);
            if (!matched) {
                json env = make_error_json("File Not Found", 404, "not_found_error", rid);
                send_envelope(worker, client_id, safe_json_to_str(env));
                continue;
            }
            for (auto & kv : path_params) params[kv.first] = kv.second;

            // Per-request cancel registry + should_stop closure
            auto cancel = std::make_shared<std::atomic<bool>>(false);
            if (!rid.empty()) {
                std::lock_guard<std::mutex> lk(in_flight_mu);
                in_flight[rid] = cancel;
            }
            // lifetime: must outlive the handler call and any streaming next()
            std::function<bool()> should_stop = [this, cancel]() {
                return !running.load(std::memory_order_relaxed) || cancel->load(std::memory_order_relaxed);
            };

            server_http_req req{params, headers, path, /*query_string*/ "", body, /*files*/ {}, should_stop};

            server_http_res_ptr response;
            try {
                response = matched->handler(req);
            } catch (const std::exception & e) {
                json env = make_error_json(e.what(), 500, "internal_server_error", rid);
                send_envelope(worker, client_id, safe_json_to_str(env));
                // unregister cancel before continuing
                if (!rid.empty()) {
                    std::lock_guard<std::mutex> lk(in_flight_mu);
                    in_flight.erase(rid);
                }
                continue;
            }

            // Send response
            if (response->is_stream()) {
                // Streaming: single multipart message [client_id, header, chunk*, empty_terminator].
                json header_env{
                    {"status",       response->status},
                    {"content_type", response->content_type},
                    {"headers",      response->headers},
                    {"rid",          rid},
                    {"stream",       true},
                };
                if (!send_stream_open(worker, client_id, safe_json_to_str(header_env))) {
                    LOG_WRN("zmq worker[%d]: stream open send failed for rid=%s\n", idx, rid.c_str());
                } else {
                    std::string chunk;
                    size_t chunk_idx = 0;
                    while (response->next(chunk)) {
                        if (chunk.empty()) {
                            // No data to send for this iteration — skip the
                            // frame entirely; client uses RCVMORE to know when
                            // the stream is done, so empty intermediate frames
                            // would be a wire-level discriminator we don't need.
                            continue;
                        }
                        if (!send_stream_chunk(worker, chunk)) {
                            LOG_WRN("zmq worker[%d]: chunk send failed for rid=%s at chunk=%zu\n", idx, rid.c_str(), chunk_idx);
                            break;
                        }
                        chunk_idx++;
                    }
                    if (!send_stream_close(worker)) {
                        LOG_WRN("zmq worker[%d]: terminator send failed for rid=%s\n", idx, rid.c_str());
                    }
                    LOG_DBG("zmq worker[%d]: stream done rid=%s chunks=%zu\n", idx, rid.c_str(), chunk_idx);
                }
            } else {
                json env{
                    {"status",       response->status},
                    {"content_type", response->content_type},
                    {"headers",      response->headers},
                    {"rid",          rid},
                    {"stream",       false},
                    {"data",         response->data},
                };
                send_envelope(worker, client_id, safe_json_to_str(env));
            }

            // on_end hook (used by Phase 3 session-aware completion to fire
            // post-completion save). Transport-agnostic; we call it after the
            // final wire byte for both stream and non-stream responses.
            if (response->on_end) {
                try {
                    response->on_end();
                } catch (const std::exception & e) {
                    LOG_ERR("zmq worker[%d]: on_end threw for rid=%s: %s\n", idx, rid.c_str(), e.what());
                }
            }

            // Unregister cancel — done with this request.
            if (!rid.empty()) {
                std::lock_guard<std::mutex> lk(in_flight_mu);
                in_flight.erase(rid);
            }
        }

        LOG_INF("zmq worker[%d]: stopping\n", idx);
        zmq_close(worker);
    }
};

//
// server_zmq_context
//

server_zmq_context::server_zmq_context()
    : pimpl(std::make_unique<server_zmq_context::Impl>())
{}

server_zmq_context::~server_zmq_context() = default;

bool server_zmq_context::init(const common_params & params) {
    path_prefix       = params.api_prefix;
    pimpl->api_prefix = path_prefix;
    pimpl->api_keys   = params.api_keys;

    if (bind_endpoints.empty()) {
        // PID-derived default IPC path so two coexisting llama-server processes
        // don't fight over the same socket file.
        bind_endpoints.push_back("ipc:///tmp/llama-server-" + std::to_string(getpid()) + ".sock");
    }

    if (n_workers <= 0) {
        n_workers = std::max(2, params.n_parallel + 2);
    }
    pimpl->n_workers = n_workers;
    pimpl->hwm       = hwm;

    {
        std::string s;
        for (size_t i = 0; i < bind_endpoints.size(); ++i) {
            if (i) s += ", ";
            s += bind_endpoints[i];
        }
        listening_address = s;
    }
    LOG_INF("zmq init: prefix='%s' endpoints=[%s] workers=%d hwm=%d\n",
            path_prefix.c_str(), listening_address.c_str(), n_workers, hwm);
    return true;
}

bool server_zmq_context::start() {
    pimpl->zmq_ctx = zmq_ctx_new();
    if (!pimpl->zmq_ctx) {
        LOG_ERR("zmq: ctx_new failed\n");
        return false;
    }

    pimpl->frontend = zmq_socket(pimpl->zmq_ctx, ZMQ_ROUTER);
    if (!pimpl->frontend) {
        LOG_ERR("zmq: ROUTER socket() failed\n");
        return false;
    }
    int linger = 0;
    zmq_setsockopt(pimpl->frontend, ZMQ_LINGER, &linger, sizeof(linger));
    zmq_setsockopt(pimpl->frontend, ZMQ_SNDHWM, &hwm,    sizeof(hwm));
    zmq_setsockopt(pimpl->frontend, ZMQ_RCVHWM, &hwm,    sizeof(hwm));

    for (const auto & ep : bind_endpoints) {
        if (zmq_bind(pimpl->frontend, ep.c_str()) != 0) {
            LOG_ERR("zmq: bind(%s) failed: %s\n", ep.c_str(), zmq_strerror(errno));
            return false;
        }
        LOG_INF("zmq: bound %s\n", ep.c_str());
    }

    pimpl->backend = zmq_socket(pimpl->zmq_ctx, ZMQ_DEALER);
    if (!pimpl->backend) {
        LOG_ERR("zmq: DEALER socket() failed\n");
        return false;
    }
    zmq_setsockopt(pimpl->backend, ZMQ_LINGER, &linger, sizeof(linger));
    zmq_setsockopt(pimpl->backend, ZMQ_SNDHWM, &hwm,    sizeof(hwm));
    zmq_setsockopt(pimpl->backend, ZMQ_RCVHWM, &hwm,    sizeof(hwm));

    if (zmq_bind(pimpl->backend, "inproc://llama_backend") != 0) {
        LOG_ERR("zmq: backend inproc bind failed: %s\n", zmq_strerror(errno));
        return false;
    }

    pimpl->running.store(true);

    pimpl->proxy_thread = std::thread([this]() {
        // zmq_proxy returns when one of the sockets is closed/terminated;
        // stop() drives that.
        (void) zmq_proxy(pimpl->frontend, pimpl->backend, nullptr);
    });

    for (int i = 0; i < pimpl->n_workers; ++i) {
        pimpl->workers.emplace_back([this, i]() { pimpl->run_worker(*this, i); });
    }
    LOG_INF("zmq: %d workers started\n", pimpl->n_workers);
    return true;
}

void server_zmq_context::stop() {
    if (!pimpl->running.exchange(false)) return;

    LOG_INF("%s", "zmq: stopping...\n");

    if (pimpl->frontend) {
        zmq_close(pimpl->frontend);
        pimpl->frontend = nullptr;
    }
    if (pimpl->backend) {
        zmq_close(pimpl->backend);
        pimpl->backend = nullptr;
    }

    if (pimpl->proxy_thread.joinable()) {
        pimpl->proxy_thread.join();
    }
    for (auto & t : pimpl->workers) {
        if (t.joinable()) t.join();
    }
    pimpl->workers.clear();

    if (pimpl->zmq_ctx) {
        zmq_ctx_term(pimpl->zmq_ctx);
        pimpl->zmq_ctx = nullptr;
    }
    LOG_INF("%s", "zmq: stopped\n");
}

void server_zmq_context::get(const std::string & path, const handler_t & handler) {
    pimpl->routes.push_back({"GET", path, handler});
}

void server_zmq_context::post(const std::string & path, const handler_t & handler) {
    pimpl->routes.push_back({"POST", path, handler});
}
