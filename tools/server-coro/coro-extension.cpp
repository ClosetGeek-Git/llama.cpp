// PHP Extension implementation for llama-server-coro
// Exposes all server-coro handlers to PHP/Swoole coroutines

#include "coro-extension.h"

#include "common.h"
#include "arg.h"
#include "llama.h"
#include "log.h"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <string>
#include <map>

using json = nlohmann::ordered_json;

// Global state (equivalent to server.cpp locals)
static common_params *g_params = nullptr;
static server_context *g_ctx_server = nullptr;
static server_routes *g_routes = nullptr;
static std::thread g_inference_thread;
static std::atomic<bool> g_initialized{false};
static std::atomic<bool> g_model_loaded{false};
static std::atomic<bool> g_model_load_failed{false};

// Llama\Request class entry and handlers
zend_class_entry *llama_request_ce = nullptr;
static zend_object_handlers llama_request_handlers;

// Forward declarations for class methods
static PHP_METHOD(LlamaRequest, __construct);
static PHP_METHOD(LlamaRequest, isStream);
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

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_request_getData, 0, 0, IS_STRING, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_request_next, 0, 0, IS_ARRAY, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_request_cancel, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

// Method entries for Llama\Request
static const zend_function_entry llama_request_methods[] = {
    PHP_ME(LlamaRequest, __construct, arginfo_llama_request_construct, ZEND_ACC_PUBLIC)
    PHP_ME(LlamaRequest, isStream, arginfo_llama_request_isStream, ZEND_ACC_PUBLIC)
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
    // Ensure cleanup if not already done
    if (g_initialized.load()) {
        if (g_ctx_server) {
            g_ctx_server->terminate();
        }
        if (g_inference_thread.joinable()) {
            g_inference_thread.join();
        }
        delete g_routes;
        g_routes = nullptr;
        delete g_ctx_server;
        g_ctx_server = nullptr;
        delete g_params;
        g_params = nullptr;
        llama_backend_free();
        g_initialized.store(false);
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

    if (g_initialized.load()) {
        php_error_docref(nullptr, E_WARNING, "llama context already initialized");
        RETURN_FALSE;
    }

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

    // Parse command-line arguments
    g_params = new common_params();
    if (!common_params_parse(argc, argv.data(), *g_params, LLAMA_EXAMPLE_SERVER)) {
        delete g_params;
        g_params = nullptr;
        php_error_docref(nullptr, E_WARNING, "Failed to parse arguments");
        RETURN_FALSE;
    }

    // Validate batch size for embeddings
    if (g_params->embedding && g_params->n_batch > g_params->n_ubatch) {
        g_params->n_batch = g_params->n_ubatch;
    }

    // Auto n_parallel
    if (g_params->n_parallel < 0) {
        g_params->n_parallel = 4;
        g_params->kv_unified = true;
    }

    // Model alias
    if (g_params->model_alias.empty() && !g_params->model.name.empty()) {
        g_params->model_alias = g_params->model.name;
    }

    // Initialize common
    common_init();

    // Create server context
    g_ctx_server = new server_context();

    // Initialize llama backend
    llama_backend_init();
    llama_numa_init(g_params->numa);

    // Create routes (this initializes all handlers and registers route mappings)
    g_routes = new server_routes(*g_params, *g_ctx_server);

    // Start inference thread - model loading happens here, not on PHP thread
    // This allows swoole_llama_init() to return immediately
    g_inference_thread = std::thread([]() {
        // Load model on inference thread
        if (!g_ctx_server->load_model(*g_params)) {
            g_model_load_failed.store(true);
            return;
        }

        // Update metadata after model load
        g_routes->update_meta(*g_ctx_server);

        g_model_loaded.store(true);

        // Run inference loop (blocks until terminate() called)
        g_ctx_server->start_loop();
    });

    g_initialized.store(true);

    RETURN_TRUE;
}

// swoole_llama_ready(): int  (0 = not ready, 1 = ready, -1 = failed)
PHP_FUNCTION(swoole_llama_ready)
{
    if (!g_initialized.load()) {
        RETURN_LONG(-1);
    }

    if (g_model_load_failed.load()) {
        RETURN_LONG(-1);
    }

    if (g_model_loaded.load()) {
        RETURN_LONG(1);
    }

    RETURN_LONG(0);
}

// swoole_llama_shutdown(): bool
PHP_FUNCTION(swoole_llama_shutdown)
{
    if (!g_initialized.load()) {
        RETURN_TRUE;
    }

    // Terminate inference loop (unblocks start_loop)
    g_ctx_server->terminate();

    // Wait for inference thread to finish
    if (g_inference_thread.joinable()) {
        g_inference_thread.join();
    }

    // Cleanup in reverse order
    delete g_routes;
    g_routes = nullptr;

    delete g_ctx_server;
    g_ctx_server = nullptr;

    delete g_params;
    g_params = nullptr;

    llama_backend_free();

    g_initialized.store(false);
    g_model_loaded.store(false);
    g_model_load_failed.store(false);

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
    req.headers["X-Response-Type"].push_back("raw");

    return true;
}

// Llama\Request::__construct(array $params)
static PHP_METHOD(LlamaRequest, __construct)
{
    zval *z_params;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(z_params)
    ZEND_PARSE_PARAMETERS_END();

    if (!g_routes) {
        zend_throw_exception(zend_ce_exception, "llama context not initialized", 0);
        RETURN_THROWS();
    }

    LlamaRequestObject *intern = Z_LLAMA_REQUEST_P(ZEND_THIS);

    // Convert PHP array to request
    if (!php_array_to_request(z_params, intern->request, intern->cancelled)) {
        zend_throw_exception(zend_ce_exception, "Invalid request format", 0);
        RETURN_THROWS();
    }

    // Find handler using path matching
    std::map<std::string, std::string> path_params;
    const server_http_context::handler_t *handler = g_routes->find_handler(
        intern->request.method, intern->request.path, path_params);

    if (!handler) {
        zend_throw_exception_ex(zend_ce_exception, 0,
            "No handler for %s %s", intern->request.method.c_str(), intern->request.path.c_str());
        RETURN_THROWS();
    }

    // Merge path params into request params (path params take precedence)
    for (const auto &[k, v] : path_params) {
        intern->request.params[k] = v;
    }

    // Invoke handler with the persistent request reference
    try {
        intern->response = (*handler)(intern->request);
    } catch (const std::exception &e) {
        zend_throw_exception_ex(zend_ce_exception, 0, "Handler error: %s", e.what());
        RETURN_THROWS();
    }

    if (!intern->response) {
        zend_throw_exception(zend_ce_exception, "Handler returned null response", 0);
        RETURN_THROWS();
    }

    intern->is_stream = intern->response->is_stream();
}

// Llama\Request::isStream(): bool
static PHP_METHOD(LlamaRequest, isStream)
{
    ZEND_PARSE_PARAMETERS_NONE();

    LlamaRequestObject *intern = Z_LLAMA_REQUEST_P(ZEND_THIS);
    RETURN_BOOL(intern->is_stream);
}

// Llama\Request::getData(): ?string
static PHP_METHOD(LlamaRequest, getData)
{
    ZEND_PARSE_PARAMETERS_NONE();

    LlamaRequestObject *intern = Z_LLAMA_REQUEST_P(ZEND_THIS);

    if (!intern->response) {
        RETURN_NULL();
    }

    if (intern->is_stream) {
        // For streaming: return first chunk from response->data
        if (!intern->response->data.empty()) {
            std::string first_chunk = std::move(intern->response->data);
            intern->response->data.clear();
            RETURN_STRINGL(first_chunk.c_str(), first_chunk.length());
        }
        RETURN_NULL();
    } else {
        // For non-streaming: return response->data
        RETURN_STRINGL(intern->response->data.c_str(), intern->response->data.length());
    }
}

// Llama\Request::next(): ?array
static PHP_METHOD(LlamaRequest, next)
{
    ZEND_PARSE_PARAMETERS_NONE();

    LlamaRequestObject *intern = Z_LLAMA_REQUEST_P(ZEND_THIS);

    if (!intern->response || !intern->is_stream) {
        RETURN_NULL();
    }

    std::string chunk;

    // Loop until we get actual content or stream ends
    while (true) {
        bool has_more = intern->response->next(chunk);

        if (!has_more) {
            RETURN_NULL();
        }

        if (!chunk.empty()) {
            break;
        }
        // Empty chunk but has_more=true means flush/continue, keep looping
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

        // Convert JSON to PHP array
        // Simple recursive JSON to PHP array conversion
        std::function<void(const json &, zval *)> json_to_zval;
        json_to_zval = [&json_to_zval](const json &j, zval *z) {
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
        };

        zval result;
        json_to_zval(parsed, &result);
        RETURN_ZVAL(&result, 0, 0);

    } catch (const std::exception &e) {
        // JSON parse error - return null
        RETURN_NULL();
    }
}

// Llama\Request::cancel(): void
static PHP_METHOD(LlamaRequest, cancel)
{
    ZEND_PARSE_PARAMETERS_NONE();

    LlamaRequestObject *intern = Z_LLAMA_REQUEST_P(ZEND_THIS);

    if (intern->cancelled) {
        intern->cancelled->store(true);
    }
}

// Argument info
ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_init, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, argv, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_swoole_llama_void, 0, 0, 0)
ZEND_END_ARG_INFO()

// Function entries
static const zend_function_entry swoole_llama_functions[] = {
    PHP_FE(swoole_llama_init, arginfo_swoole_llama_init)
    PHP_FE(swoole_llama_ready, arginfo_swoole_llama_void)
    PHP_FE(swoole_llama_shutdown, arginfo_swoole_llama_void)
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
