#pragma once

#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"

#include <nlohmann/json_fwd.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

struct server_context_impl; // private implementation

struct server_context_meta {
    std::string build_info;
    std::string model_name;
    std::set<std::string> model_aliases;
    std::set<std::string> model_tags;
    std::string model_path;
    bool has_mtmd;
    bool has_inp_image;
    bool has_inp_audio;
    json json_ui_settings;            // Primary: new name
    json json_webui_settings;            // Deprecated: use json_ui_settings instead (kept for backward compat)
    int slot_n_ctx;
    enum llama_pooling_type pooling_type;

    // chat params
    server_chat_params & chat_params;
    std::map<std::string, bool> chat_template_caps;

    // tokens
    std::string bos_token_str;
    std::string eos_token_str;
    llama_token fim_pre_token;
    llama_token fim_sub_token;
    llama_token fim_mid_token;
    llama_token fim_pad_token;
    llama_token fim_rep_token;
    llama_token fim_sep_token;

    // sampling
    std::vector<llama_logit_bias> logit_bias_eog;

    // model meta
    enum llama_vocab_type model_vocab_type;
    int32_t model_vocab_n_tokens;
    int32_t model_n_ctx_train;
    int32_t model_n_embd_inp;
    uint64_t model_n_params;
    uint64_t model_size;
};

struct server_context {
    std::unique_ptr<server_context_impl> impl;

    server_context();
    ~server_context();

    // load the model and initialize llama_context
    // returns true on success
    bool load_model(common_params & params);

    // this function will block main thread until termination
    void start_loop();

    // terminate main loop (will unblock start_loop)
    void terminate();

    // get the underlaying llama_context, can return nullptr if sleeping
    // not thread-safe, should only be used from the main thread
    llama_context * get_llama_context() const;

    // get a new response reader, used by CLI application
    server_response_reader get_response_reader();

    // get server metadata (read-only), can only be called after load_model()
    // not thread-safe, should only be used from the main thread
    server_context_meta get_meta() const;

    // register a callback to be called when sleeping state changes
    // must be set before load_model() is called
    void on_sleeping_changed(std::function<void(bool)> callback);
};


// forward declarations
struct server_res_generator;

// A single route entry: HTTP method, path template ("/v1/slots/:id_slot/info"),
// and the handler that serves it. Data-driven route registration lets every
// transport (cpp-httplib, ZMQ, future) iterate the same list, with route
// mutations (e.g. router-mode handler swaps in server.cpp) applied once on
// server_routes BEFORE the list is materialized.
struct server_route {
    std::string method;   // "GET" or "POST"
    std::string path;     // template using :name syntax
    server_transport_handler_t handler;
};

// Server-side session storage entry.
//
// `data` is held as a shared_ptr<const vector<uint8_t>> so the restore path
// can take an O(1) refcount bump under `sessions_mutex` and release the lock
// before the (potentially 100MB) blob read. With a raw vector the lock-held
// copy serialized all session-restore traffic; with shared_ptr the only
// lock-held work is finding the entry and incrementing the refcount.
//
// The blob is immutable once stored (replace = whole-blob swap), so const-ptr
// sharing is safe — readers see a snapshot even if a concurrent save bumps in
// a new entry.
struct session_state {
    std::shared_ptr<const std::vector<uint8_t>> data;  // SES1-format blob
    int64_t created_at = 0;
    int64_t updated_at = 0;
    bool    pinned     = false;  // skipped during LRU eviction (see S2)

    session_state() = default;
    session_state(std::shared_ptr<const std::vector<uint8_t>> d, int64_t now)
        : data(std::move(d)), created_at(now), updated_at(now) {}

    // Movable AND copyable: copy is O(1) (refcount bump only). The previous
    // move-only contract existed to prevent accidental O(N) vector copies;
    // that concern goes away under shared_ptr.
    session_state(session_state &&) = default;
    session_state & operator=(session_state &&) = default;
    session_state(const session_state &) = default;
    session_state & operator=(const session_state &) = default;

    size_t size() const { return data ? data->size() : 0; }
};

struct server_routes {
    server_routes(const common_params & params, server_context & ctx_server);

    void init_routes();

    // Materialize the route table. Call ONCE per process, after any handler
    // mutations (e.g. router-mode proxy swaps in server.cpp) are in place but
    // BEFORE any transport's get()/post() registration. Returns by value so
    // the transport can move it into its own internal map.
    std::vector<server_route> routes() const;

    // note: this is not thread-safe and can only when ctx_http.is_ready is false
    void update_meta(const server_context & ctx_server) {
        this->meta = std::make_unique<server_context_meta>(ctx_server.get_meta());
    }

    // handlers using lambda function, so that they can capture `this` without `std::bind`
    // they won't be called until ctx_http.is_ready is set to true
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_metrics;
    server_http_context::handler_t get_slots;
    server_http_context::handler_t post_slots;
    server_http_context::handler_t get_props;
    server_http_context::handler_t post_props;
    server_http_context::handler_t post_infill;
    server_http_context::handler_t post_completions;
    server_http_context::handler_t post_completions_oai;
    server_http_context::handler_t post_chat_completions;
    server_http_context::handler_t post_responses_oai;
    server_http_context::handler_t post_transcriptions_oai;
    server_http_context::handler_t post_anthropic_messages;
    server_http_context::handler_t post_anthropic_count_tokens;
    server_http_context::handler_t post_apply_template;
    server_http_context::handler_t get_models;
    server_http_context::handler_t post_tokenize;
    server_http_context::handler_t post_detokenize;
    server_http_context::handler_t post_embeddings;
    server_http_context::handler_t post_embeddings_oai;
    server_http_context::handler_t post_rerank;
    server_http_context::handler_t post_classify;
    server_http_context::handler_t get_slot_info;
    server_http_context::handler_t get_lora_adapters;
    server_http_context::handler_t post_lora_adapters;
    server_http_context::handler_t post_sessions;
    server_http_context::handler_t get_sessions;
    server_http_context::handler_t post_session_pin;
    server_http_context::handler_t post_session_unpin;

    // to be used in router mode (upstream addition)
    json get_model_info() const;

private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type);

    // S3: session block carried alongside a completion request. The PHP/Caelus
    // frontend uses this to migrate a user from one host's warm tier to
    // another: include restore_key on the first turn after a host switch
    // (server inlines the Redis-fetched blob), and save_key_after to write the
    // post-completion KV state back to the warm tier (PHP optionally cools to
    // Redis via the separate /sessions endpoints).
    struct session_action {
        std::string restore_key;     // session id to restore into id_slot before dispatch
        std::string save_key_after;  // session id to save id_slot's state into post-completion
        bool        evict_after = false;  // drop the warm copy after save (PHP holds canonical)
        bool        present     = false;  // true if the body had a "session" field at all
    };
    // Read and strip the "session" object from body. Idempotent on inputs
    // without the field — returns an action with present=false.
    static session_action extract_session_action(json & body);

    // Wraps handle_completions_impl with the pre-restore and post-save hooks.
    // Returns an error response if the restore fails. Stream save failures
    // log only; the response is already on the wire.
    std::unique_ptr<server_res_generator> handle_completions_with_session(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const session_action & sa);
    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_save_state(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore_state(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_tokens(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_context_shift(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type);

    // using unique_ptr to allow late initialization of const
    std::unique_ptr<const server_context_meta> meta;

    const common_params & params;
    const server_context_impl & ctx_server;

    server_queue & queue_tasks;
    server_response & queue_results;
    std::unique_ptr<server_res_generator> create_response(bool bypass_sleep = false);

    // Server-side session storage (in-process "warm" tier; PHP/Caelus drives
    // the "cool" Redis tier on top of this surface). Keyed by string so the
    // PHP frontend can index directly by user_id (Colyseus presence pattern)
    // without an int translation layer.
    std::map<std::string, session_state> sessions;
    mutable std::mutex sessions_mutex;
    // Running total of bytes held in `sessions`. Updated alongside insertions
    // and removals while sessions_mutex is held; Phase 3 (S2) consumes this to
    // enforce --sessions-max-bytes. Atomic so external observers can read it
    // without acquiring sessions_mutex.
    std::atomic<size_t> sessions_total_bytes_{0};
    std::unique_ptr<server_res_generator> handle_sessions_action(const server_http_req & req, const std::string & id_session);
    std::unique_ptr<server_res_generator> handle_sessions_list(const server_http_req & req);

    // S2: enforce params.sessions_max_bytes by LRU-evicting unpinned entries.
    // Caller must hold sessions_mutex. Returns the number of entries evicted.
    // No-op when sessions_max_bytes == 0 (unbounded).
    size_t evict_sessions_until_under_budget();

    // S3: synchronous one-shot helpers used by completion handlers consuming
    // the optional "session" body block. Each posts a single task via a
    // one-off response_reader and waits for completion. On failure, returns
    // false; the caller decides whether to surface the error (pre-completion
    // restore failures abort the request; post-completion save failures only
    // log, since the response is already on the wire).
    bool perform_session_restore(const std::string & key, int id_slot,
                                 const std::function<bool()> & should_stop,
                                 std::string & err_out);
    bool perform_session_save  (const std::string & key, int id_slot,
                                 bool evict_after,
                                 const std::function<bool()> & should_stop,
                                 std::string & err_out);
};
