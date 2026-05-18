#pragma once

// Transport-agnostic request/response types and handler signature shared by
// every transport implementation (cpp-httplib, ZMQ, etc). This is the contract
// between server_routes (which produces handlers) and the transport layer
// (which dispatches requests to them).
//
// The names retain the "http" prefix for source compatibility — they describe
// HTTP-shaped requests regardless of the underlying wire format.

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

// generator-like API for response generation
// this object responds with one of two modes:
// 1) normal response: `data` contains the full response body
// 2) streaming response: each call to next(output) generates the next chunk
//    when next(output) returns false, no more data after the current chunk
//    note: some chunks can be empty, in which case no data is sent for that chunk
struct server_http_res {
    std::string content_type = "application/json; charset=utf-8";
    int status = 200;
    std::string data;
    std::map<std::string, std::string> headers;

    // TODO: move this to a virtual function once we have proper polymorphism support
    std::function<bool(std::string &)> next = nullptr;

    // Optional callback invoked by the transport once the response has been
    // fully delivered (after the final chunk for streaming responses; after
    // the response is written for non-streaming). Used by S3's one-shot
    // session block to fire the post-completion save. Transport-agnostic;
    // cpp-httplib calls it from on_complete, ZMQ calls it after the terminator
    // frame is sent. May be null.
    std::function<void()> on_end = nullptr;

    bool is_stream() const {
        return next != nullptr;
    }

    virtual ~server_http_res() = default;
};

// unique pointer, used by set_chunked_content_provider
// httplib requires the stream provider to be stored in heap
using server_http_res_ptr = std::unique_ptr<server_http_res>;
using raw_buffer = std::vector<uint8_t>;

struct uploaded_file {
    raw_buffer data;
    std::string filename;
    std::string content_type;
};

struct server_http_req {
    std::map<std::string, std::string> params; // path_params + query_params
    std::map<std::string, std::string> headers; // used by MCP proxy
    std::string path;
    std::string query_string; // query parameters string (e.g. "action=save")
    std::string body;
    std::map<std::string, uploaded_file> files; // used for file uploads (form data)
    const std::function<bool()> & should_stop;

    std::string get_param(const std::string & key, const std::string & def = "") const {
        auto it = params.find(key);
        if (it != params.end()) {
            return it->second;
        }
        return def;
    }
};

// Transport-agnostic handler signature. The handler should never throw —
// transports rely on ex_wrapper to translate exceptions into structured error
// responses.
using server_transport_handler_t = std::function<server_http_res_ptr(const server_http_req & req)>;
