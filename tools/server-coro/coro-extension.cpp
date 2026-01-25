// PHP Extension implementation for llama-server-coro
// Exposes all server-coro handlers to PHP/Swoole coroutines

#include "coro-extension.h"

#include "common.h"
#include "arg.h"
#include "llama.h"
#include "log.h"
#include "server-common.h"

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
    PHP_DBG(req_id, "invoking handler...");
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
    PHP_DBG(req_id, "handler returned");

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

// Function entries
static const zend_function_entry swoole_llama_functions[] = {
    PHP_FE(swoole_llama_init, arginfo_swoole_llama_init)
    PHP_FE(swoole_llama_ready, arginfo_swoole_llama_void)
    PHP_FE(swoole_llama_shutdown, arginfo_swoole_llama_void)
    PHP_FE(swoole_llama_load_model, arginfo_swoole_llama_load_model)
    PHP_FE(swoole_llama_model_ready, arginfo_swoole_llama_model_ready)
    PHP_FE(swoole_llama_unload_model, arginfo_swoole_llama_unload_model)
    PHP_FE(swoole_llama_list_models, arginfo_swoole_llama_list_models)
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
