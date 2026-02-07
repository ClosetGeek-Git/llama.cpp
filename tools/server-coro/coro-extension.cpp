// PHP Extension implementation for llama-server-coro
// Exposes all server-coro handlers to PHP/Swoole coroutines

#include "coro-extension.h"

#include "common.h"
#include "arg.h"
#include "llama.h"
#include "log.h"
#include "server-common.h"
#include "server-queue.h"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <string>
#include <map>
#include <cstdlib>
#include <mutex>
#include <chrono>

using json = nlohmann::ordered_json;

// Debug logging for PHP extension - enabled by LLAMA_PHP_DEBUG_JSON env var
static bool php_debug_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = getenv("LLAMA_PHP_DEBUG_JSON");
        enabled = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
    }
    return enabled == 1;
}

#define PHP_DBG(req_id, fmt, ...) \
    do { if (php_debug_enabled()) { fprintf(stderr, "[PHP_DBG req=%d] %s: " fmt "\n", req_id, __func__, ##__VA_ARGS__); fflush(stderr); } } while(0)

// Request ID counter for tracking
static std::atomic<int> g_request_id_counter{0};

// Model status enum
enum class ModelStatus {
    LOADING,
    LOADED,
    FAILED,
    UNLOADING
};

// Session state blob stored in the session map
// Uses zend_string* for zero-copy interop with PHP and direct buffer access for llama APIs
struct SessionState {
    zend_string *data = nullptr;
    int64_t created_at = 0;
    int64_t updated_at = 0;

    SessionState() = default;

    ~SessionState() {
        if (data) {
            zend_string_release(data);
        }
    }

    // Move constructor
    SessionState(SessionState &&other) noexcept
        : data(other.data), created_at(other.created_at), updated_at(other.updated_at) {
        other.data = nullptr;
    }

    // Move assignment
    SessionState &operator=(SessionState &&other) noexcept {
        if (this != &other) {
            if (data) {
                zend_string_release(data);
            }
            data = other.data;
            created_at = other.created_at;
            updated_at = other.updated_at;
            other.data = nullptr;
        }
        return *this;
    }

    // Delete copy to prevent accidental 800MB copies
    SessionState(const SessionState &) = delete;
    SessionState &operator=(const SessionState &) = delete;
};

// Per-model instance
struct ModelInstance {
    std::string name;
    common_params params;
    std::unique_ptr<server_context> ctx_server;
    std::unique_ptr<server_routes> routes;
    std::thread inference_thread;
    std::atomic<ModelStatus> status{ModelStatus::LOADING};
    std::atomic<int64_t> last_used{0};
    std::string error_message;

    // Session map: session_id -> KV cache state blob
    std::map<int, SessionState> sessions;
    std::mutex sessions_mutex;
    
    void update_last_used() {
        last_used.store(std::chrono::steady_clock::now().time_since_epoch().count());
    }
};

// Model registry
static std::map<std::string, std::unique_ptr<ModelInstance>> g_models;
static std::mutex g_models_mutex;
static std::string g_legacy_model_name;
static size_t g_models_max = 0;
static std::atomic<bool> g_backend_initialized{false};

// Extract model name from -m or --model argument (filename without .gguf)
static std::string extract_model_name_from_argv(const std::vector<char*> &argv) {
    for (size_t i = 0; i < argv.size(); i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argv.size()) {
            std::string path = argv[i + 1];
            size_t slash = path.find_last_of("/\\");
            std::string filename = (slash != std::string::npos) ? path.substr(slash + 1) : path;
            if (filename.size() > 5 && filename.substr(filename.size() - 5) == ".gguf") {
                filename = filename.substr(0, filename.size() - 5);
            }
            return filename;
        }
    }
    return "";
}

// Extract model name from JSON body
static std::string extract_model_name(const std::string &body) {
    if (body.empty()) return "";
    try {
        json j = json::parse(body);
        if (j.contains("model") && j["model"].is_string()) {
            return j["model"].get<std::string>();
        }
    } catch (...) {}
    return "";
}

// LRU eviction helper (must hold g_models_mutex when calling)
static void evict_lru_if_needed() {
    if (g_models_max == 0 || g_models.size() < g_models_max) return;
    
    std::string oldest_name;
    int64_t oldest_time = INT64_MAX;
    for (const auto &[name, inst] : g_models) {
        if (inst->status.load() == ModelStatus::LOADED && inst->last_used.load() < oldest_time) {
            oldest_time = inst->last_used.load();
            oldest_name = name;
        }
    }
    if (!oldest_name.empty()) {
        auto it = g_models.find(oldest_name);
        if (it != g_models.end()) {
            it->second->ctx_server->terminate();
            if (it->second->inference_thread.joinable()) {
                it->second->inference_thread.join();
            }
            g_models.erase(it);
        }
    }
}

// Llama\Request class entry and handlers
zend_class_entry *llama_request_ce = nullptr;
static zend_object_handlers llama_request_handlers;

// Forward declarations for class methods
static PHP_METHOD(LlamaRequest, __construct);
static PHP_METHOD(LlamaRequest, isStream);
static PHP_METHOD(LlamaRequest, getStatusCode);
static PHP_METHOD(LlamaRequest, getData);
static PHP_METHOD(LlamaRequest, next);
static PHP_METHOD(LlamaRequest, cancel);

// Create object handler
static zend_object *llama_request_create_object(zend_class_entry *ce)
{
    LlamaRequestObject *intern = static_cast<LlamaRequestObject *>(
        ecalloc(1, sizeof(LlamaRequestObject) + zend_object_properties_size(ce))
    );

    // Placement new for C++ objects
    new (&intern->request) server_http_req();
    intern->response = nullptr;
    intern->cancelled = std::make_shared<std::atomic<bool>>(false);
    intern->is_stream = false;
    intern->request_id = -1;  // Will be assigned in __construct

    // Session fields
    intern->session_model_inst = nullptr;
    intern->session_id = -1;
    intern->session_slot_id = -1;
    intern->session_update = false;
    intern->session_remove = false;
    intern->session_saved = false;

    zend_object_std_init(&intern->std, ce);
    object_properties_init(&intern->std, ce);

    intern->std.handlers = &llama_request_handlers;

    return &intern->std;
}

// Free object handler
static void llama_request_free_object(zend_object *obj)
{
    LlamaRequestObject *intern = llama_request_from_obj(obj);

    // Cancel if streaming
    if (intern->cancelled) {
        intern->cancelled->store(true);
    }

    // Destructor for C++ objects
    intern->request.~server_http_req();
    intern->response.reset();
    intern->cancelled.reset();

    zend_object_std_dtor(&intern->std);
}

// Argument info for Llama\Request methods
ZEND_BEGIN_ARG_INFO_EX(arginfo_llama_request_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, params, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_request_isStream, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_request_getStatusCode, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_request_getData, 0, 0, IS_ARRAY, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_request_next, 0, 0, IS_ARRAY, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_request_cancel, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

// Method entries for Llama\Request
static const zend_function_entry llama_request_methods[] = {
    PHP_ME(LlamaRequest, __construct, arginfo_llama_request_construct, ZEND_ACC_PUBLIC)
    PHP_ME(LlamaRequest, isStream, arginfo_llama_request_isStream, ZEND_ACC_PUBLIC)
    PHP_ME(LlamaRequest, getStatusCode, arginfo_llama_request_getStatusCode, ZEND_ACC_PUBLIC)
    PHP_ME(LlamaRequest, getData, arginfo_llama_request_getData, ZEND_ACC_PUBLIC)
    PHP_ME(LlamaRequest, next, arginfo_llama_request_next, ZEND_ACC_PUBLIC)
    PHP_ME(LlamaRequest, cancel, arginfo_llama_request_cancel, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

// Module initialization
PHP_MINIT_FUNCTION(swoole_llama)
{
    // Register Llama\Request class
    zend_class_entry ce;
    INIT_NS_CLASS_ENTRY(ce, "Llama", "Request", llama_request_methods);
    llama_request_ce = zend_register_internal_class(&ce);
    llama_request_ce->create_object = llama_request_create_object;

    // Initialize object handlers
    memcpy(&llama_request_handlers, &std_object_handlers, sizeof(zend_object_handlers));
    llama_request_handlers.offset = XtOffsetOf(LlamaRequestObject, std);
    llama_request_handlers.free_obj = llama_request_free_object;

    return SUCCESS;
}

// Module shutdown
PHP_MSHUTDOWN_FUNCTION(swoole_llama)
{
    // Unload all models
    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        for (auto &[name, inst] : g_models) {
            if (inst->ctx_server) {
                inst->ctx_server->terminate();
            }
            if (inst->inference_thread.joinable()) {
                inst->inference_thread.join();
            }
        }
        g_models.clear();
    }
    
    if (g_backend_initialized.load()) {
        llama_backend_free();
        g_backend_initialized.store(false);
    }
    
    return SUCCESS;
}

// Module info
PHP_MINFO_FUNCTION(swoole_llama)
{
    php_info_print_table_start();
    php_info_print_table_header(2, "swoole_llama support", "enabled");
    php_info_print_table_row(2, "Version", PHP_SWOOLE_LLAMA_VERSION);
    php_info_print_table_end();
}

// swoole_llama_init(array $argv): bool
PHP_FUNCTION(swoole_llama_init)
{
    zval *z_argv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(z_argv)
    ZEND_PARSE_PARAMETERS_END();

    // Convert PHP array to argc/argv
    HashTable *ht = Z_ARRVAL_P(z_argv);
    int argc = zend_hash_num_elements(ht);

    std::vector<std::string> arg_storage;
    std::vector<char*> argv;
    arg_storage.reserve(argc);
    argv.reserve(argc);

    zval *val;
    ZEND_HASH_FOREACH_VAL(ht, val) {
        convert_to_string(val);
        arg_storage.push_back(std::string(Z_STRVAL_P(val), Z_STRLEN_P(val)));
        argv.push_back(const_cast<char*>(arg_storage.back().c_str()));
    } ZEND_HASH_FOREACH_END();

    // Extract model name
    std::string model_name = extract_model_name_from_argv(argv);
    if (model_name.empty()) {
        php_error_docref(nullptr, E_WARNING, "No model specified (-m or --model)");
        RETURN_FALSE;
    }

    // Check if already loaded
    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        if (g_models.find(model_name) != g_models.end()) {
            php_error_docref(nullptr, E_WARNING, "Model '%s' already loaded", model_name.c_str());
            RETURN_FALSE;
        }
    }

    // Parse params
    auto inst = std::make_unique<ModelInstance>();
    inst->name = model_name;
    if (!common_params_parse(argc, argv.data(), inst->params, LLAMA_EXAMPLE_SERVER)) {
        php_error_docref(nullptr, E_WARNING, "Failed to parse arguments");
        RETURN_FALSE;
    }

    // Normalize params
    if (inst->params.embedding && inst->params.n_batch > inst->params.n_ubatch) {
        inst->params.n_batch = inst->params.n_ubatch;
    }
    if (inst->params.n_parallel < 0) {
        inst->params.n_parallel = 4;
        inst->params.kv_unified = true;
    }
    if (inst->params.model_alias.empty() && !inst->params.model.name.empty()) {
        inst->params.model_alias = inst->params.model.name;
    }

    // Initialize backend once
    if (!g_backend_initialized.load()) {
        common_init();
        llama_backend_init();
        llama_numa_init(inst->params.numa);
        g_backend_initialized.store(true);
        
        // Read LLAMA_MODELS_MAX env
        const char *max_env = getenv("LLAMA_MODELS_MAX");
        if (max_env) g_models_max = std::max(0, atoi(max_env));
    }

    // Create context and routes
    inst->ctx_server = std::make_unique<server_context>();
    inst->routes = std::make_unique<server_routes>(inst->params, *inst->ctx_server);
    inst->update_last_used();

    // Capture raw pointer for thread
    ModelInstance *inst_ptr = inst.get();

    // Start inference thread
    inst->inference_thread = std::thread([inst_ptr]() {
        try {
            if (!inst_ptr->ctx_server->load_model(inst_ptr->params)) {
                inst_ptr->status.store(ModelStatus::FAILED);
                inst_ptr->error_message = "Failed to load model";
                return;
            }
            inst_ptr->routes->update_meta(*inst_ptr->ctx_server);
            inst_ptr->status.store(ModelStatus::LOADED);
            inst_ptr->ctx_server->start_loop();
        } catch (const std::exception &e) {
            inst_ptr->status.store(ModelStatus::FAILED);
            inst_ptr->error_message = e.what();
        }
    });

    // Register in map
    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        evict_lru_if_needed();
        g_models[model_name] = std::move(inst);
        g_legacy_model_name = model_name;
    }

    RETURN_TRUE;
}

// swoole_llama_ready(): int  (0 = not ready, 1 = ready, -1 = failed)
PHP_FUNCTION(swoole_llama_ready)
{
    std::lock_guard<std::mutex> lock(g_models_mutex);
    if (g_legacy_model_name.empty()) {
        RETURN_LONG(-1);
    }
    auto it = g_models.find(g_legacy_model_name);
    if (it == g_models.end()) {
        RETURN_LONG(-1);
    }
    ModelStatus status = it->second->status.load();
    if (status == ModelStatus::FAILED) {
        RETURN_LONG(-1);
    }
    if (status == ModelStatus::LOADED) {
        RETURN_LONG(1);
    }
    RETURN_LONG(0);
}

// swoole_llama_shutdown(): bool
PHP_FUNCTION(swoole_llama_shutdown)
{
    std::unique_lock<std::mutex> lock(g_models_mutex);
    if (g_legacy_model_name.empty()) {
        RETURN_TRUE;
    }
    auto it = g_models.find(g_legacy_model_name);
    if (it == g_models.end()) {
        RETURN_TRUE;
    }
    
    it->second->ctx_server->terminate();
    
    // Release lock while joining thread
    std::thread thread_to_join = std::move(it->second->inference_thread);
    g_models.erase(it);
    g_legacy_model_name.clear();
    lock.unlock();
    
    if (thread_to_join.joinable()) {
        thread_to_join.join();
    }
    
    RETURN_TRUE;
}

// Helper: Convert PHP array to server_http_req
static bool php_array_to_request(zval *z_request, server_http_req &req, std::shared_ptr<std::atomic<bool>> &cancelled) {
    if (Z_TYPE_P(z_request) != IS_ARRAY) {
        return false;
    }

    HashTable *ht = Z_ARRVAL_P(z_request);
    zval *val;

    // Required: method
    val = zend_hash_str_find(ht, "method", sizeof("method") - 1);
    if (!val || Z_TYPE_P(val) != IS_STRING) {
        return false;
    }
    req.method = std::string(Z_STRVAL_P(val), Z_STRLEN_P(val));

    // Required: path
    val = zend_hash_str_find(ht, "path", sizeof("path") - 1);
    if (!val || Z_TYPE_P(val) != IS_STRING) {
        return false;
    }
    req.path = std::string(Z_STRVAL_P(val), Z_STRLEN_P(val));

    // Optional: body
    val = zend_hash_str_find(ht, "body", sizeof("body") - 1);
    if (val && Z_TYPE_P(val) == IS_STRING) {
        req.body = std::string(Z_STRVAL_P(val), Z_STRLEN_P(val));
    }

    // Optional: headers (array of string => string[])
    val = zend_hash_str_find(ht, "headers", sizeof("headers") - 1);
    if (val && Z_TYPE_P(val) == IS_ARRAY) {
        HashTable *headers_ht = Z_ARRVAL_P(val);
        zend_string *key;
        zval *header_val;
        ZEND_HASH_FOREACH_STR_KEY_VAL(headers_ht, key, header_val) {
            if (key && header_val) {
                std::string header_name(ZSTR_VAL(key), ZSTR_LEN(key));
                if (Z_TYPE_P(header_val) == IS_ARRAY) {
                    // Multi-value header
                    zval *v;
                    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(header_val), v) {
                        if (Z_TYPE_P(v) == IS_STRING) {
                            req.headers[header_name].push_back(std::string(Z_STRVAL_P(v), Z_STRLEN_P(v)));
                        }
                    } ZEND_HASH_FOREACH_END();
                } else if (Z_TYPE_P(header_val) == IS_STRING) {
                    // Single value header
                    req.headers[header_name].push_back(std::string(Z_STRVAL_P(header_val), Z_STRLEN_P(header_val)));
                }
            }
        } ZEND_HASH_FOREACH_END();
    }

    // Optional: params (array of string => string) - merged query + path params
    val = zend_hash_str_find(ht, "params", sizeof("params") - 1);
    if (val && Z_TYPE_P(val) == IS_ARRAY) {
        HashTable *params_ht = Z_ARRVAL_P(val);
        zend_string *key;
        zval *param_val;
        ZEND_HASH_FOREACH_STR_KEY_VAL(params_ht, key, param_val) {
            if (key && param_val && Z_TYPE_P(param_val) == IS_STRING) {
                req.params[std::string(ZSTR_VAL(key), ZSTR_LEN(key))] =
                    std::string(Z_STRVAL_P(param_val), Z_STRLEN_P(param_val));
            }
        } ZEND_HASH_FOREACH_END();
    }

    // Optional: query_string
    val = zend_hash_str_find(ht, "query_string", sizeof("query_string") - 1);
    if (val && Z_TYPE_P(val) == IS_STRING) {
        req.query_string = std::string(Z_STRVAL_P(val), Z_STRLEN_P(val));
    }

    // Optional: scheme
    val = zend_hash_str_find(ht, "scheme", sizeof("scheme") - 1);
    if (val && Z_TYPE_P(val) == IS_STRING) {
        req.scheme = std::string(Z_STRVAL_P(val), Z_STRLEN_P(val));
    }

    // Optional: host
    val = zend_hash_str_find(ht, "host", sizeof("host") - 1);
    if (val && Z_TYPE_P(val) == IS_STRING) {
        req.host = std::string(Z_STRVAL_P(val), Z_STRLEN_P(val));
    }

    // Optional: port
    val = zend_hash_str_find(ht, "port", sizeof("port") - 1);
    if (val && Z_TYPE_P(val) == IS_LONG) {
        req.port = static_cast<int>(Z_LVAL_P(val));
    }

    // Optional: remote_addr
    val = zend_hash_str_find(ht, "remote_addr", sizeof("remote_addr") - 1);
    if (val && Z_TYPE_P(val) == IS_STRING) {
        req.remote_addr = std::string(Z_STRVAL_P(val), Z_STRLEN_P(val));
    }

    // Set up should_stop callback
    cancelled = std::make_shared<std::atomic<bool>>(false);
    req.should_stop = [cancelled]() { return cancelled->load(); };

    // Request raw JSON format (no SSE wrapper)
    if (req.headers.find("X-Response-Type") == req.headers.end()) {
        req.headers["X-Response-Type"].push_back("raw");
    }

    return true;
}

// Helper: perform session save/remove after completion finishes
static void session_post_completion(LlamaRequestObject *intern) {
    if (!intern->session_model_inst || intern->session_saved) {
        return;
    }
    intern->session_saved = true;

    ModelInstance *inst = intern->session_model_inst;
    int req_id = intern->request_id;

    // Session remove
    if (intern->session_remove && intern->session_id >= 0) {
        PHP_DBG(req_id, "removing session %d", intern->session_id);
        std::lock_guard<std::mutex> slock(inst->sessions_mutex);
        inst->sessions.erase(intern->session_id);
    }

    // Session save: capture KV cache state from slot after completion
    if (intern->session_update && intern->session_id >= 0 && intern->session_slot_id >= 0) {
        PHP_DBG(req_id, "saving session %d from slot %d", intern->session_id, intern->session_slot_id);
        std::vector<uint8_t> state_data = inst->ctx_server->get_slot_state(intern->session_slot_id);
        if (state_data.empty()) {
            PHP_DBG(req_id, "WARNING: session save failed for session %d from slot %d",
                    intern->session_id, intern->session_slot_id);
        } else {
            PHP_DBG(req_id, "session save OK: %zu bytes from slot %d", state_data.size(), intern->session_slot_id);
            // Wrap vector into non-persistent zend_string for zero-copy storage
            zend_string *zstr = zend_string_init(reinterpret_cast<const char *>(state_data.data()), state_data.size(), 0);
            int64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
            std::lock_guard<std::mutex> slock(inst->sessions_mutex);
            auto sit = inst->sessions.find(intern->session_id);
            if (sit != inst->sessions.end()) {
                if (sit->second.data) {
                    zend_string_release(sit->second.data);
                }
                sit->second.data = zstr;
                sit->second.updated_at = now;
            } else {
                SessionState state;
                state.data = zstr;
                state.created_at = now;
                state.updated_at = now;
                inst->sessions.emplace(intern->session_id, std::move(state));
            }
        }
    }
}

// Llama\Request::__construct(array $params)
static PHP_METHOD(LlamaRequest, __construct)
{
    zval *z_params;
    int req_id = g_request_id_counter.fetch_add(1);

    PHP_DBG(req_id, "ENTER");

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(z_params)
    ZEND_PARSE_PARAMETERS_END();

    LlamaRequestObject *intern = Z_LLAMA_REQUEST_P(ZEND_THIS);
    intern->request_id = req_id;

    // Convert PHP array to request
    PHP_DBG(req_id, "parsing request array...");
    if (!php_array_to_request(z_params, intern->request, intern->cancelled)) {
        PHP_DBG(req_id, "ERROR: Invalid request format");
        // Return OAI error response instead of throwing
        json error_json = format_error_response("Invalid request format", ERROR_TYPE_INVALID_REQUEST);
        intern->response = std::make_unique<server_http_res>();
        intern->response->status = 400;
        intern->response->data = json{{"error", error_json}}.dump();
        intern->is_stream = false;
        return;
    }
    PHP_DBG(req_id, "parsed: method=%s path=%s body_len=%zu", 
            intern->request.method.c_str(), intern->request.path.c_str(), intern->request.body.size());

    // Extract model name from request body (REQUIRED)
    std::string model_name = extract_model_name(intern->request.body);
    if (model_name.empty()) {
        PHP_DBG(req_id, "ERROR: no 'model' field in request body");
        // Return OAI error response instead of throwing
        json error_json = format_error_response("No 'model' field in request body", ERROR_TYPE_INVALID_REQUEST);
        intern->response = std::make_unique<server_http_res>();
        intern->response->status = 400;
        intern->response->data = json{{"error", error_json}}.dump();
        intern->is_stream = false;
        return;
    }

    // Look up model instance
    ModelInstance *inst = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        auto it = g_models.find(model_name);
        if (it == g_models.end()) {
            PHP_DBG(req_id, "ERROR: model '%s' not found", model_name.c_str());
            // Return OAI error response instead of throwing
            json error_json = format_error_response("Model '" + model_name + "' not found", ERROR_TYPE_NOT_FOUND);
            intern->response = std::make_unique<server_http_res>();
            intern->response->status = 404;
            intern->response->data = json{{"error", error_json}}.dump();
            intern->is_stream = false;
            return;
        }
        inst = it->second.get();
        inst->update_last_used();
    }

    if (inst->status.load() != ModelStatus::LOADED) {
        PHP_DBG(req_id, "ERROR: model '%s' not ready", model_name.c_str());
        // Return OAI error response instead of throwing
        json error_json = format_error_response("Model '" + model_name + "' is not ready", ERROR_TYPE_UNAVAILABLE);
        intern->response = std::make_unique<server_http_res>();
        intern->response->status = 503;
        intern->response->data = json{{"error", error_json}}.dump();
        intern->is_stream = false;
        return;
    }

    // Parse session fields from request body
    {
        try {
            json body = json::parse(intern->request.body);
            if (body.contains("session_id") && body["session_id"].is_number_integer()) {
                intern->session_id = body["session_id"].get<int>();
                intern->session_model_inst = inst;
                intern->session_slot_id = json_value(body, "id_slot", -1);
                intern->session_update = json_value(body, "session_update", false);
                intern->session_remove = json_value(body, "session_remove", false);
                PHP_DBG(req_id, "session: id=%d slot=%d update=%d remove=%d",
                        intern->session_id, intern->session_slot_id,
                        intern->session_update, intern->session_remove);
            }
        } catch (...) {
            // body parse error is handled later by the handler
        }
    }

    // Session restore: if session_id and id_slot are both set, restore KV cache before processing
    if (intern->session_model_inst && intern->session_id >= 0 && intern->session_slot_id >= 0) {
        const uint8_t *restore_ptr = nullptr;
        size_t restore_len = 0;
        {
            std::lock_guard<std::mutex> slock(inst->sessions_mutex);
            auto sit = inst->sessions.find(intern->session_id);
            if (sit != inst->sessions.end() && sit->second.data) {
                restore_ptr = reinterpret_cast<const uint8_t *>(ZSTR_VAL(sit->second.data));
                restore_len = ZSTR_LEN(sit->second.data);
            }
        }
        if (restore_ptr && restore_len > 0) {
            PHP_DBG(req_id, "restoring session %d (%zu bytes) to slot %d",
                    intern->session_id, restore_len, intern->session_slot_id);
            size_t n_read = inst->ctx_server->set_slot_state(intern->session_slot_id, restore_ptr, restore_len);
            if (n_read == 0) {
                PHP_DBG(req_id, "WARNING: session restore failed for session %d to slot %d",
                        intern->session_id, intern->session_slot_id);
            } else {
                PHP_DBG(req_id, "session restore OK: %zu bytes read", n_read);
            }
        } else {
            PHP_DBG(req_id, "no session data found for session %d, proceeding without restore",
                    intern->session_id);
        }
    }

    // Find handler using path matching
    std::map<std::string, std::string> path_params;
    PHP_DBG(req_id, "finding handler...");
    const server_http_context::handler_t *handler = inst->routes->find_handler(intern->request.method, intern->request.path, path_params);

    if (!handler) {
        PHP_DBG(req_id, "ERROR: No handler found");
        // Return OAI error response instead of throwing
        std::string msg = "No handler for " + intern->request.method + " " + intern->request.path;
        json error_json = format_error_response(msg, ERROR_TYPE_NOT_FOUND);
        intern->response = std::make_unique<server_http_res>();
        intern->response->status = 404;
        intern->response->data = json{{"error", error_json}}.dump();
        intern->is_stream = false;
        return;
    }
    PHP_DBG(req_id, "handler found, path_params=%zu", path_params.size());

    // Merge path params into request params (path params take precedence)
    for (const auto &[k, v] : path_params) {
        intern->request.params[k] = v;
    }

    // Invoke handler with the persistent request reference
    int64_t handler_start_us = ggml_time_us();
    timing_log("HANDLER_INVOKE_ENTER", req_id, {{"path", intern->request.path}});
    PHP_DBG(req_id, "invoking handler... timestamp_us=%lld", (long long)handler_start_us);
    try {
        intern->response = (*handler)(intern->request);
    } catch (const std::exception &e) {
        PHP_DBG(req_id, "ERROR: handler threw exception: %s", e.what());
        // Return OAI error response instead of throwing
        json error_json = format_error_response(std::string("Handler error: ") + e.what(), ERROR_TYPE_SERVER);
        intern->response = std::make_unique<server_http_res>();
        intern->response->status = 500;
        intern->response->data = json{{"error", error_json}}.dump();
        intern->is_stream = false;
        return;
    }
    int64_t handler_end_us = ggml_time_us();
    timing_log("HANDLER_INVOKE_EXIT", req_id, {{"duration_us", handler_end_us - handler_start_us}});
    PHP_DBG(req_id, "handler returned: elapsed_us=%lld timestamp_us=%lld", (long long)(handler_end_us - handler_start_us), (long long)handler_end_us);

    if (!intern->response) {
        PHP_DBG(req_id, "ERROR: handler returned null response");
        // Return OAI error response instead of throwing
        json error_json = format_error_response("Handler returned null response", ERROR_TYPE_SERVER);
        intern->response = std::make_unique<server_http_res>();
        intern->response->status = 500;
        intern->response->data = json{{"error", error_json}}.dump();
        intern->is_stream = false;
        return;
    }

    intern->is_stream = intern->response->is_stream();
    PHP_DBG(req_id, "EXIT: is_stream=%d response_data_len=%zu", intern->is_stream, intern->response->data.size());

    // For non-streaming: completion is already done at this point, perform session save/remove
    if (!intern->is_stream && intern->session_model_inst) {
        session_post_completion(intern);
    }
    // For streaming: session save/remove happens when stream ends (in next() returning null)
}

// Static helper: Convert nlohmann::json to PHP zval (recursive)
static void json_to_zval(const json &j, zval *z) {
    if (j.is_null()) {
        ZVAL_NULL(z);
    } else if (j.is_boolean()) {
        ZVAL_BOOL(z, j.get<bool>());
    } else if (j.is_number_integer()) {
        ZVAL_LONG(z, j.get<int64_t>());
    } else if (j.is_number_float()) {
        ZVAL_DOUBLE(z, j.get<double>());
    } else if (j.is_string()) {
        const std::string &s = j.get_ref<const std::string &>();
        ZVAL_STRINGL(z, s.c_str(), s.length());
    } else if (j.is_array()) {
        array_init(z);
        for (size_t i = 0; i < j.size(); i++) {
            zval elem;
            json_to_zval(j[i], &elem);
            add_next_index_zval(z, &elem);
        }
    } else if (j.is_object()) {
        array_init(z);
        for (auto it = j.begin(); it != j.end(); ++it) {
            zval elem;
            json_to_zval(it.value(), &elem);
            add_assoc_zval(z, it.key().c_str(), &elem);
        }
    }
}

// Llama\Request::isStream(): bool
static PHP_METHOD(LlamaRequest, isStream)
{
    ZEND_PARSE_PARAMETERS_NONE();

    LlamaRequestObject *intern = Z_LLAMA_REQUEST_P(ZEND_THIS);
    PHP_DBG(intern->request_id, "is_stream=%d", intern->is_stream);
    RETURN_BOOL(intern->is_stream);
}

// Llama\Request::getStatusCode(): int
static PHP_METHOD(LlamaRequest, getStatusCode)
{
    ZEND_PARSE_PARAMETERS_NONE();

    LlamaRequestObject *intern = Z_LLAMA_REQUEST_P(ZEND_THIS);
    int status = intern->response ? intern->response->status : 200;
    PHP_DBG(intern->request_id, "status=%d", status);
    RETURN_LONG(status);
}

// Llama\Request::getData(): ?array
static PHP_METHOD(LlamaRequest, getData)
{
    ZEND_PARSE_PARAMETERS_NONE();

    LlamaRequestObject *intern = Z_LLAMA_REQUEST_P(ZEND_THIS);
    PHP_DBG(intern->request_id, "ENTER: is_stream=%d", intern->is_stream);

    if (!intern->response) {
        PHP_DBG(intern->request_id, "EXIT: no response, returning null");
        RETURN_NULL();
    }

    std::string chunk;
    if (intern->is_stream) {
        // For streaming: get first chunk from response->data
        if (intern->response->data.empty()) {
            PHP_DBG(intern->request_id, "EXIT (stream): no data, returning null");
            RETURN_NULL();
        }
        chunk = std::move(intern->response->data);
        intern->response->data.clear();
    } else {
        // For non-streaming: get full response from response->data
        chunk = intern->response->data;
    }

    // Strip trailing newline
    if (!chunk.empty() && chunk.back() == '\n') {
        chunk.pop_back();
    }
    if (chunk.empty()) {
        RETURN_NULL();
    }

    // Parse JSON and convert to PHP array
    try {
        json parsed = json::parse(chunk);
        zval result;
        json_to_zval(parsed, &result);
        PHP_DBG(intern->request_id, "EXIT: returning parsed JSON as array, len=%zu", chunk.length());
        RETURN_ZVAL(&result, 0, 0);
    } catch (const std::exception &e) {
        PHP_DBG(intern->request_id, "EXIT: JSON parse error: %s", e.what());
        RETURN_NULL();
    }
}

// Llama\Request::next(): ?array
static PHP_METHOD(LlamaRequest, next)
{
    ZEND_PARSE_PARAMETERS_NONE();

    LlamaRequestObject *intern = Z_LLAMA_REQUEST_P(ZEND_THIS);
    PHP_DBG(intern->request_id, "ENTER: is_stream=%d", intern->is_stream);

    if (!intern->response || !intern->is_stream) {
        PHP_DBG(intern->request_id, "EXIT: not streaming, returning null");
        RETURN_NULL();
    }

    std::string chunk;

    // Loop until we get actual content or stream ends
    PHP_DBG(intern->request_id, "calling response->next()...");
    while (true) {
        bool has_more = intern->response->next(chunk);

        if (!has_more) {
            PHP_DBG(intern->request_id, "EXIT: has_more=false, stream ended");
            // Stream ended: perform session save/remove
            if (intern->session_model_inst) {
                session_post_completion(intern);
            }
            RETURN_NULL();
        }

        if (!chunk.empty()) {
            PHP_DBG(intern->request_id, "got chunk, len=%zu, first100=[%.*s]", chunk.size(), (int)std::min(chunk.size(), (size_t)100), chunk.c_str());
            break;
        }
        // Empty chunk but has_more=true means flush/continue, keep looping
        PHP_DBG(intern->request_id, "empty chunk, continuing loop...");
    }

    // Parse JSON and return as array
    // The chunk is raw JSON + newline from TASK_RESPONSE_TYPE_RAW
    try {
        // Remove trailing newline if present
        if (!chunk.empty() && chunk.back() == '\n') {
            chunk.pop_back();
        }
        if (chunk.empty()) {
            RETURN_NULL();
        }

        json parsed = json::parse(chunk);
        zval result;
        json_to_zval(parsed, &result);
        PHP_DBG(intern->request_id, "EXIT: returning parsed JSON as array");
        RETURN_ZVAL(&result, 0, 0);

    } catch (const std::exception &e) {
        // JSON parse error - return null
        PHP_DBG(intern->request_id, "EXIT: JSON parse error: %s", e.what());
        RETURN_NULL();
    }
}

// Llama\Request::cancel(): void
static PHP_METHOD(LlamaRequest, cancel)
{
    ZEND_PARSE_PARAMETERS_NONE();

    LlamaRequestObject *intern = Z_LLAMA_REQUEST_P(ZEND_THIS);
    PHP_DBG(intern->request_id, "ENTER");

    if (intern->cancelled) {
        intern->cancelled->store(true);
        PHP_DBG(intern->request_id, "EXIT: cancelled set to true");
    } else {
        PHP_DBG(intern->request_id, "EXIT: no cancelled ptr");
    }
}

// Argument info
ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_init, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, argv, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_void, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_load_model, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, argv, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_model_ready, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_unload_model, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_list_models, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_session_get, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, model_name, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, session_id, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_session_set, 0, 0, 3)
    ZEND_ARG_TYPE_INFO(0, model_name, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, session_id, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, data, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_session_remove, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, model_name, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, session_id, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_session_list, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, model_name, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_session_save, 0, 0, 3)
    ZEND_ARG_TYPE_INFO(0, model_name, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, session_id, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, slot_id, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_session_restore, 0, 0, 3)
    ZEND_ARG_TYPE_INFO(0, model_name, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, session_id, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, slot_id, IS_LONG, 0)
ZEND_END_ARG_INFO()

// swoole_llama_load_model(array $argv): bool
PHP_FUNCTION(swoole_llama_load_model)
{
    zval *z_argv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(z_argv)
    ZEND_PARSE_PARAMETERS_END();

    // Convert PHP array to argc/argv
    HashTable *ht = Z_ARRVAL_P(z_argv);
    int argc = zend_hash_num_elements(ht);

    std::vector<std::string> arg_storage;
    std::vector<char*> argv;
    arg_storage.reserve(argc);
    argv.reserve(argc);

    zval *val;
    ZEND_HASH_FOREACH_VAL(ht, val) {
        convert_to_string(val);
        arg_storage.push_back(std::string(Z_STRVAL_P(val), Z_STRLEN_P(val)));
        argv.push_back(const_cast<char*>(arg_storage.back().c_str()));
    } ZEND_HASH_FOREACH_END();

    // Extract model name
    std::string model_name = extract_model_name_from_argv(argv);
    if (model_name.empty()) {
        php_error_docref(nullptr, E_WARNING, "No model specified (-m or --model)");
        RETURN_FALSE;
    }

    // Check if already loaded
    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        if (g_models.find(model_name) != g_models.end()) {
            php_error_docref(nullptr, E_WARNING, "Model '%s' already loaded", model_name.c_str());
            RETURN_FALSE;
        }
    }

    // Parse params
    auto inst = std::make_unique<ModelInstance>();
    inst->name = model_name;
    if (!common_params_parse(argc, argv.data(), inst->params, LLAMA_EXAMPLE_SERVER)) {
        php_error_docref(nullptr, E_WARNING, "Failed to parse arguments for model '%s'", model_name.c_str());
        RETURN_FALSE;
    }

    // Normalize params
    if (inst->params.embedding && inst->params.n_batch > inst->params.n_ubatch) {
        inst->params.n_batch = inst->params.n_ubatch;
    }
    if (inst->params.n_parallel < 0) {
        inst->params.n_parallel = 4;
        inst->params.kv_unified = true;
    }
    if (inst->params.model_alias.empty() && !inst->params.model.name.empty()) {
        inst->params.model_alias = inst->params.model.name;
    }

    // Initialize backend once
    if (!g_backend_initialized.load()) {
        common_init();
        llama_backend_init();
        llama_numa_init(inst->params.numa);
        g_backend_initialized.store(true);
        
        // Read LLAMA_MODELS_MAX env
        const char *max_env = getenv("LLAMA_MODELS_MAX");
        if (max_env) g_models_max = std::max(0, atoi(max_env));
    }

    // Create context and routes
    inst->ctx_server = std::make_unique<server_context>();
    inst->routes = std::make_unique<server_routes>(inst->params, *inst->ctx_server);
    inst->update_last_used();

    // Capture raw pointer for thread
    ModelInstance *inst_ptr = inst.get();

    // Start inference thread
    inst->inference_thread = std::thread([inst_ptr]() {
        try {
            if (!inst_ptr->ctx_server->load_model(inst_ptr->params)) {
                inst_ptr->status.store(ModelStatus::FAILED);
                inst_ptr->error_message = "Failed to load model";
                return;
            }
            inst_ptr->routes->update_meta(*inst_ptr->ctx_server);
            inst_ptr->status.store(ModelStatus::LOADED);
            inst_ptr->ctx_server->start_loop();
        } catch (const std::exception &e) {
            inst_ptr->status.store(ModelStatus::FAILED);
            inst_ptr->error_message = e.what();
        }
    });

    // Register in map
    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        evict_lru_if_needed();
        g_models[model_name] = std::move(inst);
    }

    RETURN_TRUE;
}

// swoole_llama_model_ready(string $name): int
PHP_FUNCTION(swoole_llama_model_ready)
{
    zend_string *z_name;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(z_name)
    ZEND_PARSE_PARAMETERS_END();

    std::string name(ZSTR_VAL(z_name), ZSTR_LEN(z_name));
    
    std::lock_guard<std::mutex> lock(g_models_mutex);
    auto it = g_models.find(name);
    if (it == g_models.end()) {
        RETURN_LONG(-1);
    }
    ModelStatus status = it->second->status.load();
    if (status == ModelStatus::FAILED) RETURN_LONG(-1);
    if (status == ModelStatus::LOADED) RETURN_LONG(1);
    RETURN_LONG(0);
}

// swoole_llama_unload_model(string $name): bool
PHP_FUNCTION(swoole_llama_unload_model)
{
    zend_string *z_name;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(z_name)
    ZEND_PARSE_PARAMETERS_END();

    std::string name(ZSTR_VAL(z_name), ZSTR_LEN(z_name));
    
    std::unique_lock<std::mutex> lock(g_models_mutex);
    auto it = g_models.find(name);
    if (it == g_models.end()) {
        RETURN_FALSE;
    }
    
    it->second->ctx_server->terminate();
    
    // Release lock while joining thread
    std::thread thread_to_join = std::move(it->second->inference_thread);
    g_models.erase(it);
    lock.unlock();
    
    if (thread_to_join.joinable()) {
        thread_to_join.join();
    }
    RETURN_TRUE;
}

// swoole_llama_list_models(): array
PHP_FUNCTION(swoole_llama_list_models)
{
    ZEND_PARSE_PARAMETERS_NONE();
    
    array_init(return_value);
    
    std::lock_guard<std::mutex> lock(g_models_mutex);
    for (const auto &[name, inst] : g_models) {
        zval model_info;
        array_init(&model_info);
        add_assoc_string(&model_info, "name", name.c_str());
        const char *status_str = "unknown";
        switch (inst->status.load()) {
            case ModelStatus::LOADING: status_str = "loading"; break;
            case ModelStatus::LOADED: status_str = "loaded"; break;
            case ModelStatus::FAILED: status_str = "failed"; break;
            case ModelStatus::UNLOADING: status_str = "unloading"; break;
        }
        add_assoc_string(&model_info, "status", status_str);
        add_next_index_zval(return_value, &model_info);
    }
}

// swoole_llama_session_get(string $model_name, int $session_id): ?string
// Returns the raw binary blob for the given session, or null if not found
PHP_FUNCTION(swoole_llama_session_get)
{
    zend_string *z_model;
    zend_long session_id;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_STR(z_model)
        Z_PARAM_LONG(session_id)
    ZEND_PARSE_PARAMETERS_END();

    std::string model_name(ZSTR_VAL(z_model), ZSTR_LEN(z_model));

    // Look up model under global lock, release before heavy copy
    ModelInstance *inst = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        auto it = g_models.find(model_name);
        if (it == g_models.end()) {
            php_error_docref(nullptr, E_WARNING, "Model '%s' not found", model_name.c_str());
            RETURN_NULL();
        }
        inst = it->second.get();
    }

    std::lock_guard<std::mutex> slock(inst->sessions_mutex);
    auto sit = inst->sessions.find(static_cast<int>(session_id));
    if (sit == inst->sessions.end() || !sit->second.data) {
        RETURN_NULL();
    }

    // Zero-copy: bump refcount and return the same zend_string
    RETURN_STR(zend_string_copy(sit->second.data));
}

// swoole_llama_session_set(string $model_name, int $session_id, string $data): bool
// Sets the raw binary blob for the given session (import from external storage)
PHP_FUNCTION(swoole_llama_session_set)
{
    zend_string *z_model;
    zend_long session_id;
    zend_string *z_data;

    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_STR(z_model)
        Z_PARAM_LONG(session_id)
        Z_PARAM_STR(z_data)
    ZEND_PARSE_PARAMETERS_END();

    std::string model_name(ZSTR_VAL(z_model), ZSTR_LEN(z_model));

    // Look up model under global lock, release before heavy copy
    ModelInstance *inst = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        auto it = g_models.find(model_name);
        if (it == g_models.end()) {
            php_error_docref(nullptr, E_WARNING, "Model '%s' not found", model_name.c_str());
            RETURN_FALSE;
        }
        inst = it->second.get();
    }

    // Zero-copy: bump refcount on the PHP string — no 800MB memcpy
    zend_string *stored = zend_string_copy(z_data);
    int64_t now = std::chrono::steady_clock::now().time_since_epoch().count();

    {
        std::lock_guard<std::mutex> slock(inst->sessions_mutex);
        auto sit = inst->sessions.find(static_cast<int>(session_id));
        if (sit != inst->sessions.end()) {
            if (sit->second.data) {
                zend_string_release(sit->second.data);
            }
            sit->second.data = stored;
            sit->second.updated_at = now;
        } else {
            SessionState state;
            state.data = stored;
            state.created_at = now;
            state.updated_at = now;
            inst->sessions.emplace(static_cast<int>(session_id), std::move(state));
        }
    }

    RETURN_TRUE;
}

// swoole_llama_session_remove(string $model_name, int $session_id): bool
PHP_FUNCTION(swoole_llama_session_remove)
{
    zend_string *z_model;
    zend_long session_id;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_STR(z_model)
        Z_PARAM_LONG(session_id)
    ZEND_PARSE_PARAMETERS_END();

    std::string model_name(ZSTR_VAL(z_model), ZSTR_LEN(z_model));

    ModelInstance *inst = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        auto it = g_models.find(model_name);
        if (it == g_models.end()) {
            php_error_docref(nullptr, E_WARNING, "Model '%s' not found", model_name.c_str());
            RETURN_FALSE;
        }
        inst = it->second.get();
    }

    // Erase under session lock; move blob out so destructor runs outside the lock
    SessionState removed_state;
    bool erased = false;
    {
        std::lock_guard<std::mutex> slock(inst->sessions_mutex);
        auto sit = inst->sessions.find(static_cast<int>(session_id));
        if (sit != inst->sessions.end()) {
            removed_state = std::move(sit->second);
            inst->sessions.erase(sit);
            erased = true;
        }
    }
    // removed_state destructor (potentially large free) runs here, outside all locks

    RETURN_BOOL(erased);
}

// swoole_llama_session_list(string $model_name): array
// Returns array of session info: [['id' => int, 'size' => int, 'created_at' => int, 'updated_at' => int], ...]
PHP_FUNCTION(swoole_llama_session_list)
{
    zend_string *z_model;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(z_model)
    ZEND_PARSE_PARAMETERS_END();

    std::string model_name(ZSTR_VAL(z_model), ZSTR_LEN(z_model));

    ModelInstance *inst = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        auto it = g_models.find(model_name);
        if (it == g_models.end()) {
            php_error_docref(nullptr, E_WARNING, "Model '%s' not found", model_name.c_str());
            array_init(return_value);
            return;
        }
        inst = it->second.get();
    }

    std::lock_guard<std::mutex> slock(inst->sessions_mutex);

    array_init(return_value);
    for (const auto &[sid, state] : inst->sessions) {
        zval entry;
        array_init(&entry);
        add_assoc_long(&entry, "id", sid);
        add_assoc_long(&entry, "size", static_cast<zend_long>(state.data ? ZSTR_LEN(state.data) : 0));
        add_assoc_long(&entry, "created_at", state.created_at);
        add_assoc_long(&entry, "updated_at", state.updated_at);
        add_next_index_zval(return_value, &entry);
    }
}

// swoole_llama_session_save(string $model_name, int $session_id, int $slot_id): bool
// Captures the current KV cache state of the given slot and stores it in the session map
PHP_FUNCTION(swoole_llama_session_save)
{
    zend_string *z_model;
    zend_long session_id;
    zend_long slot_id;

    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_STR(z_model)
        Z_PARAM_LONG(session_id)
        Z_PARAM_LONG(slot_id)
    ZEND_PARSE_PARAMETERS_END();

    std::string model_name(ZSTR_VAL(z_model), ZSTR_LEN(z_model));

    ModelInstance *inst = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        auto it = g_models.find(model_name);
        if (it == g_models.end()) {
            php_error_docref(nullptr, E_WARNING, "Model '%s' not found", model_name.c_str());
            RETURN_FALSE;
        }
        inst = it->second.get();
    }

    if (inst->status.load() != ModelStatus::LOADED) {
        php_error_docref(nullptr, E_WARNING, "Model '%s' is not ready", model_name.c_str());
        RETURN_FALSE;
    }

    // Use existing thread-safe get_slot_state which posts SEQ_STATE_GET to inference thread
    std::vector<uint8_t> state_data = inst->ctx_server->get_slot_state(static_cast<int>(slot_id));
    if (state_data.empty()) {
        php_error_docref(nullptr, E_WARNING, "Failed to get slot state for slot %d", static_cast<int>(slot_id));
        RETURN_FALSE;
    }

    // Wrap vector into non-persistent zend_string for zero-copy storage
    zend_string *zstr = zend_string_init(reinterpret_cast<const char *>(state_data.data()), state_data.size(), 0);
    int64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
    {
        std::lock_guard<std::mutex> slock(inst->sessions_mutex);
        auto sit = inst->sessions.find(static_cast<int>(session_id));
        if (sit != inst->sessions.end()) {
            if (sit->second.data) {
                zend_string_release(sit->second.data);
            }
            sit->second.data = zstr;
            sit->second.updated_at = now;
        } else {
            SessionState state;
            state.data = zstr;
            state.created_at = now;
            state.updated_at = now;
            inst->sessions.emplace(static_cast<int>(session_id), std::move(state));
        }
    }

    RETURN_TRUE;
}

// swoole_llama_session_restore(string $model_name, int $session_id, int $slot_id): bool
// Restores the KV cache state from the session map into the given slot
PHP_FUNCTION(swoole_llama_session_restore)
{
    zend_string *z_model;
    zend_long session_id;
    zend_long slot_id;

    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_STR(z_model)
        Z_PARAM_LONG(session_id)
        Z_PARAM_LONG(slot_id)
    ZEND_PARSE_PARAMETERS_END();

    std::string model_name(ZSTR_VAL(z_model), ZSTR_LEN(z_model));

    ModelInstance *inst = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        auto it = g_models.find(model_name);
        if (it == g_models.end()) {
            php_error_docref(nullptr, E_WARNING, "Model '%s' not found", model_name.c_str());
            RETURN_FALSE;
        }
        inst = it->second.get();
    }

    if (inst->status.load() != ModelStatus::LOADED) {
        php_error_docref(nullptr, E_WARNING, "Model '%s' is not ready", model_name.c_str());
        RETURN_FALSE;
    }

    // Get session state pointer — safe to read directly since coroutine blocks until set_slot_state returns
    const uint8_t *restore_ptr = nullptr;
    size_t restore_len = 0;
    {
        std::lock_guard<std::mutex> slock(inst->sessions_mutex);
        auto sit = inst->sessions.find(static_cast<int>(session_id));
        if (sit == inst->sessions.end() || !sit->second.data) {
            php_error_docref(nullptr, E_WARNING, "Session %d not found for model '%s'", static_cast<int>(session_id), model_name.c_str());
            RETURN_FALSE;
        }
        restore_ptr = reinterpret_cast<const uint8_t *>(ZSTR_VAL(sit->second.data));
        restore_len = ZSTR_LEN(sit->second.data);
    }

    // Use existing thread-safe set_slot_state which posts SEQ_STATE_SET to inference thread
    size_t n_read = inst->ctx_server->set_slot_state(static_cast<int>(slot_id), restore_ptr, restore_len);
    if (n_read == 0) {
        php_error_docref(nullptr, E_WARNING, "Failed to restore session %d to slot %d", static_cast<int>(session_id), static_cast<int>(slot_id));
        RETURN_FALSE;
    }

    RETURN_TRUE;
}

// Function entries
static const zend_function_entry swoole_llama_functions[] = {
    PHP_FE(swoole_llama_init, arginfo_swoole_llama_init)
    PHP_FE(swoole_llama_ready, arginfo_swoole_llama_void)
    PHP_FE(swoole_llama_shutdown, arginfo_swoole_llama_void)
    PHP_FE(swoole_llama_load_model, arginfo_swoole_llama_load_model)
    PHP_FE(swoole_llama_model_ready, arginfo_swoole_llama_model_ready)
    PHP_FE(swoole_llama_unload_model, arginfo_swoole_llama_unload_model)
    PHP_FE(swoole_llama_list_models, arginfo_swoole_llama_list_models)
    PHP_FE(swoole_llama_session_get, arginfo_swoole_llama_session_get)
    PHP_FE(swoole_llama_session_set, arginfo_swoole_llama_session_set)
    PHP_FE(swoole_llama_session_remove, arginfo_swoole_llama_session_remove)
    PHP_FE(swoole_llama_session_list, arginfo_swoole_llama_session_list)
    PHP_FE(swoole_llama_session_save, arginfo_swoole_llama_session_save)
    PHP_FE(swoole_llama_session_restore, arginfo_swoole_llama_session_restore)
    PHP_FE_END
};

// Module dependency - Swoole must be loaded first
static const zend_module_dep swoole_llama_deps[] = {
    ZEND_MOD_REQUIRED("swoole")
    ZEND_MOD_END
};

// Module entry
zend_module_entry swoole_llama_module_entry = {
    STANDARD_MODULE_HEADER_EX,
    nullptr,                      // ini_entry
    swoole_llama_deps,            // deps - Swoole must load first
    PHP_SWOOLE_LLAMA_EXTNAME,
    swoole_llama_functions,
    PHP_MINIT(swoole_llama),
    PHP_MSHUTDOWN(swoole_llama),
    nullptr, // RINIT
    nullptr, // RSHUTDOWN
    PHP_MINFO(swoole_llama),
    PHP_SWOOLE_LLAMA_VERSION,
    STANDARD_MODULE_PROPERTIES
};

#ifdef COMPILE_DL_SWOOLE_LLAMA
extern "C" {
    ZEND_GET_MODULE(swoole_llama)
}
#endif
