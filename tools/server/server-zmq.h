#pragma once

#include "server-transport.h"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

struct common_params;

// ZMQ-backed transport for the same handler surface that cpp-httplib serves.
// The on-wire shape per request is:
//   request  (DEALER -> ROUTER, multipart): [ <client_id implicit>, json_envelope ]
//   response (ROUTER -> DEALER, multipart): [ <client_id implicit>, json_envelope ]
//   streaming response: [ <client_id>, header_json (SNDMORE), chunk1 (SNDMORE), ...,
//                         chunkN (SNDMORE), <empty terminator> ]
//
// json_envelope (request):
//   { "method": "GET"|"POST",
//     "path":   "/v1/...",
//     "id":     "<rid string for correlation>",   // optional
//     "headers": {...},                            // optional
//     "params":  {...},                            // optional (query)
//     "body":   "<raw body string>" }              // optional
//
// json_envelope (non-streaming response):
//   { "status": 200, "content_type": "...", "headers": {...},
//     "rid": "<echo>", "stream": false, "data": "<body>" }
//
// header_json (streaming response, first frame):
//   { "status": 200, "content_type": "...", "headers": {...},
//     "rid": "<echo>", "stream": true }
//
// A cancel envelope is { "method": "CANCEL", "id": "<rid>" }; the server flips
// the in-flight should_stop atomic for that rid (returns no reply).
struct server_zmq_context {
    class Impl;
    std::unique_ptr<Impl> pimpl;

    std::atomic<bool> is_ready = false;

    std::string path_prefix;
    std::vector<std::string> bind_endpoints;   // e.g. {"ipc:///tmp/x.sock", "tcp://0.0.0.0:5555"}
    int n_workers = 0;                         // 0 = auto (n_parallel + 2)
    int hwm       = 64;                        // ZMQ SNDHWM / RCVHWM cap

    server_zmq_context();
    ~server_zmq_context();

    bool init(const common_params & params);
    bool start();
    void stop();                               // NOT const; mutates internal state

    using handler_t = server_transport_handler_t;

    void get(const std::string & path, const handler_t & handler);
    void post(const std::string & path, const handler_t & handler);

    // for debugging / log lines
    std::string listening_address; // human-readable summary of bound endpoints
};
