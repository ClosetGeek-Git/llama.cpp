#pragma once

// PHP Extension header for llama-server-coro
// This file provides the interface between PHP/Swoole and server-coro handlers

#ifdef __cplusplus
extern "C" {
#endif

// PHP headers
#include "php.h"
#include "php_ini.h"
#include "ext/standard/info.h"
#include "zend_exceptions.h"

#define PHP_SWOOLE_LLAMA_VERSION "1.0.0"
#define PHP_SWOOLE_LLAMA_EXTNAME "swoole_llama"

// Module entry
extern zend_module_entry swoole_llama_module_entry;
#define phpext_swoole_llama_ptr &swoole_llama_module_entry

// Module lifecycle functions
PHP_MINIT_FUNCTION(swoole_llama);
PHP_MSHUTDOWN_FUNCTION(swoole_llama);
PHP_MINFO_FUNCTION(swoole_llama);

// Extension functions
PHP_FUNCTION(swoole_llama_init);
PHP_FUNCTION(swoole_llama_ready);
PHP_FUNCTION(swoole_llama_shutdown);

// Llama\Request class entry
extern zend_class_entry *llama_request_ce;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

// Server-coro headers - Swoole C API is included by server-queue.cpp
#include "server-http.h"
#include "server-context.h"

#include <atomic>
#include <memory>
#include <thread>

// Object storage for Llama\Request class
struct LlamaRequestObject {
    server_http_req request;
    server_http_res_ptr response;
    std::shared_ptr<std::atomic<bool>> cancelled;
    bool is_stream;
    zend_object std;
};

// Helper to get LlamaRequestObject from zend_object
static inline LlamaRequestObject *llama_request_from_obj(zend_object *obj) {
    return reinterpret_cast<LlamaRequestObject *>(
        reinterpret_cast<char *>(obj) - XtOffsetOf(LlamaRequestObject, std)
    );
}

#define Z_LLAMA_REQUEST_P(zv) llama_request_from_obj(Z_OBJ_P(zv))

#endif // __cplusplus
