#include "server-context.h"
#include "server-http.h"
#include "server-zmq.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-tools.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <atomic>
#include <clocale>
#include <exception>
#include <signal.h>
#include <thread> // for std::thread::hardware_concurrency

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#endif

// Shutdown is mediated by a self-pipe so the OS signal handler stays
// async-signal-safe: it only calls write() and _exit() (both ASS).
// The actual shutdown_handler (std::function, may allocate, takes locks) runs
// on a normal thread that drains the pipe.
//
// On Windows the console handler already runs on its own (non-signal) thread,
// so calling std::function from it is safe; we keep the direct-call path there.

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

#if defined(_WIN32)
static inline void signal_handler(int signal) {
    // Windows console handler thread context: std::function call is safe here.
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        _exit(1);
    }
    if (shutdown_handler) {
        shutdown_handler(signal);
    }
}
#else
// POSIX: self-pipe + listener thread.
static int shutdown_pipe_fd[2] = {-1, -1};

static void signal_handler_posix(int signo) {
    if (is_terminating.test_and_set()) {
        // second signal: force exit. write+_exit are async-signal-safe; printf/exit are not.
        static const char msg[] = "Received second interrupt, terminating immediately.\n";
        ssize_t n = write(STDERR_FILENO, msg, sizeof(msg) - 1);
        (void) n;
        _exit(1);
    }
    // Write one byte to the pipe; the listener thread (running in normal context)
    // will read it and invoke shutdown_handler. write() to a pipe of <= PIPE_BUF
    // bytes is atomic and async-signal-safe.
    unsigned char byte = (unsigned char)(signo & 0xFF);
    ssize_t n = write(shutdown_pipe_fd[1], &byte, 1);
    (void) n; // best effort; pipe full just drops the redundant signal
}

static void install_posix_signal_handlers() {
    // create the self-pipe before installing sigaction so the handler can never
    // race against an uninitialized pipe fd
    if (pipe(shutdown_pipe_fd) != 0) {
        LOG_ERR("%s: failed to create shutdown pipe: %s\n", __func__, strerror(errno));
        return;
    }
    // close-on-exec so children don't inherit; non-blocking on the write end so
    // multiple rapid signals don't deadlock the kernel buffer.
    for (int i = 0; i < 2; i++) {
        int flags = fcntl(shutdown_pipe_fd[i], F_GETFD);
        if (flags >= 0) fcntl(shutdown_pipe_fd[i], F_SETFD, flags | FD_CLOEXEC);
    }
    int wflags = fcntl(shutdown_pipe_fd[1], F_GETFL);
    if (wflags >= 0) fcntl(shutdown_pipe_fd[1], F_SETFL, wflags | O_NONBLOCK);

    struct sigaction sa{};
    sa.sa_handler = signal_handler_posix;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; // do NOT set SA_RESTART; we want read() in the listener to be EINTR-able if needed
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

// Blocks the calling thread until a signal byte arrives on the pipe, then runs
// shutdown_handler in normal (non-signal) context. Returns when the pipe is
// closed or read() fails terminally.
static void shutdown_pipe_listener_loop() {
    while (true) {
        unsigned char byte = 0;
        ssize_t n = read(shutdown_pipe_fd[0], &byte, 1);
        if (n == 1) {
            if (shutdown_handler) {
                shutdown_handler((int)byte);
            }
            return; // shutdown_handler should drive the rest of teardown
        }
        if (n == 0) {
            return; // EOF: pipe closed
        }
        if (errno == EINTR) {
            continue;
        }
        LOG_ERR("%s: read from shutdown pipe failed: %s\n", __func__, strerror(errno));
        return;
    }
}
#endif

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // own arguments required by this example
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // router server never loads a model and must not touch the GPU
    // skip device enumeration so the CUDA primary context stays uncreated
    const bool is_router_server = params.model.path.empty();
    common_params_print_info(params, !is_router_server);

    // validate batch size for embeddings
    // embeddings require all tokens to be processed in a single ubatch
    // see https://github.com/ggml-org/llama.cpp/issues/12836
    if (params.embedding && params.n_batch > params.n_ubatch) {
        SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
        SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
        params.n_batch = params.n_ubatch;
    }

    if (params.n_parallel < 0) {
        SRV_INF("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

        params.n_parallel = 4;
        params.kv_unified = true;
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    if (params.model_alias.empty() && !params.model.name.empty()) {
        params.model_alias.insert(params.model.name);
    }

    // struct that contains llama context and inference
    server_context ctx_server;

    // Validate that at least one transport is enabled. Both are allowed
    // simultaneously and serve the same handler table.
    if (!params.enable_http && !params.enable_zmq) {
        LOG_ERR("%s: at least one transport must be enabled (--enable-http or --enable-zmq)\n", __func__);
        return 1;
    }

    // Declare both transports unconditionally so they're in scope for shutdown
    // logic; only init/start them if requested.
    server_http_context ctx_http;
    server_zmq_context  ctx_zmq;

    if (params.enable_http && !ctx_http.init(params)) {
        LOG_ERR("%s: failed to initialize HTTP transport\n", __func__);
        return 1;
    }
    if (params.enable_zmq) {
        ctx_zmq.bind_endpoints = params.zmq_bind_endpoints;
        ctx_zmq.n_workers      = params.zmq_workers;
        ctx_zmq.hwm            = params.zmq_hwm;
        if (!ctx_zmq.init(params)) {
            LOG_ERR("%s: failed to initialize ZMQ transport\n", __func__);
            return 1;
        }
    }

    //
    // Router
    //

    // register API routes
    server_routes routes(params, ctx_server);
    server_tools tools;

    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_classify               = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;
    }

    // Materialize the route table AFTER any router-mode handler swaps so the
    // canonical list reflects the current bindings. Then register on every
    // ENABLED transport — both can coexist.
    const auto route_list = routes.routes();
    for (const auto & r : route_list) {
        const auto wrapped = ex_wrapper(r.handler);
        if (params.enable_http) {
            if (r.method == "GET") ctx_http.get(r.path, wrapped);
            else                    ctx_http.post(r.path, wrapped);
        }
        if (params.enable_zmq) {
            if (r.method == "GET") ctx_zmq.get(r.path, wrapped);
            else                    ctx_zmq.post(r.path, wrapped);
        }
    }
    // Router-only endpoints: bypass server_routes since the load/unload
    // handlers belong to the optionally-constructed models_routes.
    if (is_router_server) {
        const auto load_wrap   = ex_wrapper(models_routes->post_router_models_load);
        const auto unload_wrap = ex_wrapper(models_routes->post_router_models_unload);
        if (params.enable_http) {
            ctx_http.post("/models/load",   load_wrap);
            ctx_http.post("/models/unload", unload_wrap);
        }
        if (params.enable_zmq) {
            ctx_zmq.post("/models/load",   load_wrap);
            ctx_zmq.post("/models/unload", unload_wrap);
        }
    }

    // HTTP-only post-route hooks: GCP-compat, CORS proxy, and built-in tools.
    // These are HTTP-shaped features (regex routing, browser/CORS, multipart
    // form uploads) and don't map onto the ZMQ envelope wire format.
    if (params.enable_http) {
        // Google Cloud Platform (Vertex AI) compat
        // Must be called AFTER all other API routes are registered
        ctx_http.register_gcp_compat();

        // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
        // Supports both new ui_mcp_proxy and deprecated webui_mcp_proxy fields
        if (params.ui_mcp_proxy || params.webui_mcp_proxy) {
            SRV_WRN("%s", "-----------------\n");
            SRV_WRN("%s", "CORS proxy is enabled, do not expose server to untrusted environments\n");
            SRV_WRN("%s", "This feature is EXPERIMENTAL and may be removed or changed in future versions\n");
            SRV_WRN("%s", "-----------------\n");
            ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
            ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
        }
        // EXPERIMENTAL built-in tools
        if (!params.server_tools.empty()) {
            try {
                tools.setup(params.server_tools);
            } catch (const std::exception & e) {
                SRV_ERR("tools setup failed: %s\n", e.what());
                return 1;
            }
            SRV_WRN("%s", "-----------------\n");
            SRV_WRN("%s", "Built-in tools are enabled, do not expose server to untrusted environments\n");
            SRV_WRN("%s", "This feature is EXPERIMENTAL and may be changed in the future\n");
            SRV_WRN("%s", "-----------------\n");
            ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
            ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
        }
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting router server, no model will be loaded in this process\n");

        clean_up = [&models_routes, &ctx_http, &ctx_zmq, &params]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // Stop transports first (idempotent) so no new requests arrive,
            // then drain the model manager.
            if (params.enable_http) ctx_http.stop();
            if (params.enable_zmq)  ctx_zmq.stop();
            if (models_routes.has_value()) {
                models_routes->models.unload_all();
            }
            llama_backend_free();
        };

        if (params.enable_http && !ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }
        if (params.enable_zmq && !ctx_zmq.start()) {
            clean_up();
            LOG_ERR("%s: exiting due to ZMQ server error\n", __func__);
            return 1;
        }
        if (params.enable_http) ctx_http.is_ready.store(true);
        if (params.enable_zmq)  ctx_zmq.is_ready.store(true);

        shutdown_handler = [&](int) {
            if (params.enable_http) ctx_http.stop();
            if (params.enable_zmq)  ctx_zmq.stop();
        };

    } else {
        // setup clean up function, to be called before exit. Order is
        // load-bearing: stop transports first (no new requests), then terminate
        // the inference loop (drains queue_results so blocked recv() unwinds).
        clean_up = [&ctx_http, &ctx_zmq, &ctx_server, &params]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            if (params.enable_http) ctx_http.stop();
            if (params.enable_zmq)  ctx_zmq.stop();
            ctx_server.terminate();
            llama_backend_free();
        };

        // start the transports before loading the model to be able to serve /health requests
        if (params.enable_http && !ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }
        if (params.enable_zmq && !ctx_zmq.start()) {
            clean_up();
            LOG_ERR("%s: exiting due to ZMQ server error\n", __func__);
            return 1;
        }

        // load the model
        SRV_INF("%s", "loading model\n");

        if (server_models::is_child_server()) {
            ctx_server.on_sleeping_changed([&](bool sleeping) {
                server_models::notify_router_sleeping_state(sleeping);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            SRV_ERR("%s", "exiting due to model loading error\n");
            return 1;
        }

        routes.update_meta(ctx_server);
        if (params.enable_http) ctx_http.is_ready.store(true);
        if (params.enable_zmq)  ctx_zmq.is_ready.store(true);

        SRV_INF("%s", "model loaded\n");

        shutdown_handler = [&](int) {
            // Stop transports first (refuses new requests immediately), then
            // terminate the inference loop. ctx_server.terminate() also
            // terminates queue_results, so any in-flight handler thread
            // blocked in recv() unwinds with nullptr instead of hanging.
            if (params.enable_http) ctx_http.stop();
            if (params.enable_zmq)  ctx_zmq.stop();
            ctx_server.terminate();
        };
    }

    // Install signal handlers. On POSIX this creates the self-pipe and starts
    // a listener thread; the OS handler itself only writes to the pipe.
    std::thread shutdown_listener_thread;
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    install_posix_signal_handlers();
    shutdown_listener_thread = std::thread(shutdown_pipe_listener_loop);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    if (is_router_server) {
        const std::string addr = params.enable_http ? ctx_http.listening_address : ctx_zmq.listening_address;
        SRV_INF("router server is listening on %s\n", addr.c_str());
        SRV_WRN("%s", "NOTE: router mode is experimental\n");
        SRV_WRN("%s", "      it is not recommended to use this mode in untrusted environments\n");

        // Block main until shutdown is requested. We rebuild shutdown_handler
        // here to additionally signal a condvar; main wakes, then runs
        // clean_up to finalize teardown. This works uniformly whether HTTP,
        // ZMQ, or both are enabled (no dependency on which thread is joinable).
        std::mutex              shutdown_mu;
        std::condition_variable shutdown_cv;
        bool                    shutdown_requested = false;
        shutdown_handler = [&](int) {
            if (params.enable_http) ctx_http.stop();
            if (params.enable_zmq)  ctx_zmq.stop();
            {
                std::lock_guard<std::mutex> lk(shutdown_mu);
                shutdown_requested = true;
            }
            shutdown_cv.notify_all();
        };

        {
            std::unique_lock<std::mutex> lk(shutdown_mu);
            shutdown_cv.wait(lk, [&]{ return shutdown_requested; });
        }
        // wait for httplib listener to drain (no-op if HTTP disabled)
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }

        clean_up();
    } else {
        SRV_INF("server is listening on %s\n", ctx_http.listening_address.c_str());

        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (server_models::is_child_server()) {
            json model_info = routes.get_model_info();
            monitor_thread = server_models::setup_child_server(shutdown_handler, model_info);
        }

        // this call blocks the main thread until queue_tasks.terminate() is called
        ctx_server.start_loop();

        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    // unblock the shutdown listener so it can exit cleanly. closing the write
    // end gives read() a 0-return EOF.
    if (shutdown_pipe_fd[1] >= 0) {
        close(shutdown_pipe_fd[1]);
        shutdown_pipe_fd[1] = -1;
    }
    if (shutdown_listener_thread.joinable()) {
        shutdown_listener_thread.join();
    }
    if (shutdown_pipe_fd[0] >= 0) {
        close(shutdown_pipe_fd[0]);
        shutdown_pipe_fd[0] = -1;
    }
#endif

    return 0;
}
