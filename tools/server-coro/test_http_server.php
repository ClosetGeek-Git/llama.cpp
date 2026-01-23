<?php
/**
 * PHP Swoole HTTP Server for Llama API
 * 
 * This server wraps the server-coro PHP extension to expose llama inference
 * via HTTP endpoints compatible with OpenAI API format.
 * 
 * Features:
 * - Concurrent request batching through llama inference loop
 * - Streaming (SSE) and non-streaming responses
 * - Client disconnect detection with request cancellation
 * - Health endpoint with loading state
 * 
 * Usage:
 *   php test_http_server.php -m /path/to/model.gguf [--n-gpu-layers 99] ...
 */

declare(strict_types=1);

use Swoole\Coroutine as Co;
use Swoole\Coroutine\Http\Server;
use Swoole\Http\Request;
use Swoole\Http\Response;

Co::set(['hook_flags' => SWOOLE_HOOK_ALL]);

// Server configuration
const SERVER_HOST = '0.0.0.0';
const SERVER_PORT = 9501;
const PUBLIC_DIR = __DIR__ . '/public';

/**
 * Add CORS headers for browser access
 */
function add_cors_headers(Response $response): void
{
    $response->header('Access-Control-Allow-Origin', '*');
    $response->header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    $response->header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
}

/**
 * Convert Swoole HTTP Request to \Llama\Request params array
 */
function swoole_request_to_llama_params(Request $request): array
{
    $server = $request->server ?? [];
    $headers = $request->header ?? [];
    
    // Build PSR-7 compatible headers (multi-value format)
    $llamaHeaders = [];
    foreach ($headers as $key => $value) {
        $normalizedKey = str_replace(' ', '-', ucwords(str_replace('-', ' ', $key)));
        $llamaHeaders[$normalizedKey] = is_array($value) ? $value : [$value];
    }
    
    // Request raw JSON format - we handle SSE framing here
    $llamaHeaders['X-Response-Type'] = ['raw'];
    
    return [
        'method' => $server['request_method'] ?? 'GET',
        'path' => $server['request_uri'] ?? '/',
        'body' => $request->rawContent() ?: '',
        'headers' => $llamaHeaders,
        'query_string' => $server['query_string'] ?? '',
    ];
}

/**
 * Handle streaming response with disconnect detection
 */
function handle_streaming_response(\Llama\Request $llamaReq, Response $response): void
{
    add_cors_headers($response);
    $response->header('Content-Type', 'text/event-stream');
    $response->header('Cache-Control', 'no-cache');
    $response->header('Connection', 'keep-alive');
    $response->header('X-Accel-Buffering', 'no');
    
    // next() returns parsed array, we encode for SSE
    while (($chunk = $llamaReq->next()) !== null) {
        $json = json_encode($chunk, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
        if ($response->write("data: {$json}\n\n") === false) {
            $llamaReq->cancel();
            return;
        }
    }
    
    $response->write("data: [DONE]\n\n");
    $response->end();
}

/**
 * Handle non-streaming response
 */
function handle_non_streaming_response(\Llama\Request $llamaReq, Response $response): void
{
    add_cors_headers($response);
    $response->header('Content-Type', 'application/json; charset=utf-8');
    $response->end($llamaReq->getData() ?? '{}');
}

/**
 * Generic API request handler
 */
function handle_api_request(Request $request, Response $response): void
{
    add_cors_headers($response);
    
    // Handle OPTIONS preflight
    if (($request->server['request_method'] ?? 'GET') === 'OPTIONS') {
        $response->status(204);
        $response->end();
        return;
    }
    
    if (swoole_llama_ready() !== 1) {
        $response->header('Content-Type', 'application/json');
        $response->status(503);
        $response->end('{"error":{"message":"Model loading","code":503}}');
        return;
    }
    
    try {
        $llamaReq = new \Llama\Request(swoole_request_to_llama_params($request));
        
        if ($llamaReq->isStream()) {
            handle_streaming_response($llamaReq, $response);
        } else {
            handle_non_streaming_response($llamaReq, $response);
        }
    } catch (Throwable $e) {
        $response->header('Content-Type', 'application/json');
        $response->status(500);
        $response->end(json_encode(['error' => ['message' => $e->getMessage(), 'code' => 500]]));
    }
}

/**
 * Health check handler
 */
function handle_health(Request $request, Response $response): void
{
    add_cors_headers($response);
    $response->header('Content-Type', 'application/json');
    $ready = swoole_llama_ready();
    
    if ($ready === 0) {
        $response->status(503);
        $response->end('{"status":"loading"}');
    } elseif ($ready === -1) {
        $response->status(500);
        $response->end('{"status":"error"}');
    } else {
        $response->end('{"status":"ok"}');
    }
}

/**
 * Create and configure the HTTP server
 */
function create_server(): Server
{
    $server = new Server(SERVER_HOST, SERVER_PORT, false);
    
    // Health endpoints
    $server->handle('/health', 'handle_health');
    $server->handle('/v1/health', 'handle_health');
    
    // Props endpoint for webui
    $server->handle('/props', function (Request $request, Response $response) {
        add_cors_headers($response);
        $response->header('Content-Type', 'application/json');
        $response->end(json_encode([
            'default_generation_settings' => [
                'n_ctx' => 4096,
                'model' => basename($GLOBALS['argv'][array_search('-m', $GLOBALS['argv']) + 1] ?? 'model'),
            ],
            'total_slots' => 4,
            'chat_template' => 'chatml',
        ]));
    });
    
    // API endpoints
    $server->handle('/v1/models', 'handle_api_request');
    $server->handle('/v1/chat/completions', 'handle_api_request');
    $server->handle('/v1/completions', 'handle_api_request');
    $server->handle('/v1/embeddings', 'handle_api_request');
    $server->handle('/test/stream', 'handle_api_request');
    
    // WebUI and catch-all
    $server->handle('/', function (Request $request, Response $response) {
        $path = $request->server['request_uri'] ?? '/';
        
        // Serve webui only for root path
        if ($path === '/' || $path === '/index.html') {
            // CORS headers for pyodide (WASM)
            $response->header('Cross-Origin-Embedder-Policy', 'require-corp');
            $response->header('Cross-Origin-Opener-Policy', 'same-origin');
            
            // Serve loading.html if model not ready
            if (swoole_llama_ready() !== 1) {
                $response->header('Content-Type', 'text/html; charset=utf-8');
                $response->sendfile(PUBLIC_DIR . '/loading.html');
                return;
            }
            
            // Serve main webui
            $response->header('Content-Type', 'text/html; charset=utf-8');
            $response->sendfile(PUBLIC_DIR . '/index.html');
            return;
        }
        
        // 404 for all other paths
        $response->header('Content-Type', 'application/json');
        $response->status(404);
        $response->end('{"error":{"message":"Not Found","code":404}}');
    });
    
    return $server;
}

// Main entry point
Co\run(function () use ($argv) {
    echo "=== Llama PHP HTTP Server ===\n";
    
    if (!swoole_llama_init($argv)) {
        echo "ERROR: Failed to initialize llama\n";
        return;
    }
    
    echo "Loading model...\n";
    $start = microtime(true);
    while (swoole_llama_ready() === 0) {
        Co::sleep(0.1);
    }
    
    if (swoole_llama_ready() === -1) {
        echo "ERROR: Model failed to load\n";
        swoole_llama_shutdown();
        return;
    }
    
    printf("Model loaded in %.2f seconds\n", microtime(true) - $start);
    
    $server = create_server();
    
    // Graceful shutdown
    $shuttingDown = false;
    $shutdown = function () use ($server, &$shuttingDown) {
        if ($shuttingDown) return;
        $shuttingDown = true;
        echo "\nShutting down...\n";
        $server->shutdown();
        swoole_llama_shutdown();
        echo "Done.\n";
    };
    
    pcntl_async_signals(true);
    pcntl_signal(SIGINT, $shutdown);
    pcntl_signal(SIGTERM, $shutdown);
    
    echo "Listening on http://" . SERVER_HOST . ":" . SERVER_PORT . "\n";
    echo "WebUI: http://localhost:" . SERVER_PORT . "/\n";
    echo "API: /v1/models, /v1/chat/completions, /v1/completions, /v1/embeddings\n";
    echo "Press Ctrl+C to stop\n\n";
    
    $server->start();
});
