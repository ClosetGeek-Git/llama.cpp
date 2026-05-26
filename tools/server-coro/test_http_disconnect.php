<?php
/**
 * Client Disconnect Cancellation Test for Llama PHP Server
 * 
 * Tests that when a client disconnects mid-stream, the server properly
 * detects the failed write and calls cancel() on the \Llama\Request
 * to stop generation within llama.
 * 
 * The test:
 * 1. Starts a streaming request with a long response (many tokens)
 * 2. Reads only 1-2 chunks
 * 3. Abruptly closes the connection (ungraceful disconnect)
 * 4. Server should detect failed write() and call cancel()
 * 5. Verify via timing - cancelled request should stop quickly
 * 
 * Usage:
 *   1. Start the server: php test_http_server.php -m /path/to/model.gguf
 *   2. Run this test: php test_http_disconnect.php
 * 
 * Watch the server console for "[...] Client disconnected mid-stream, calling cancel()"
 */

declare(strict_types=1);

use Swoole\Coroutine as Co;
use Swoole\Coroutine\Http\Client;
use Swoole\Coroutine\WaitGroup;

const SERVER_HOST = '127.0.0.1';
const SERVER_PORT = 9501;

/**
 * Wait for server to be ready
 */
function wait_for_server(int $timeoutSeconds = 30): bool
{
    $startTime = time();
    
    while (time() - $startTime < $timeoutSeconds) {
        $client = new Client(SERVER_HOST, SERVER_PORT);
        $client->set(['timeout' => 1.0]);
        $client->get('/health');
        
        if ($client->statusCode === 200) {
            $client->close();
            return true;
        }
        
        $client->close();
        Co::sleep(0.5);
    }
    
    return false;
}

/**
 * Start a streaming request and disconnect after N chunks
 * Returns how long the request took before we disconnected
 */
function start_and_abort_request(int $id, int $disconnectAfterChunks = 2): array
{
    $startTime = microtime(true);
    $chunksReceived = 0;
    $content = '';
    
    // Use socket directly for more control over disconnect timing
    $socket = new \Swoole\Coroutine\Socket(AF_INET, SOCK_STREAM, 0);
    
    if (!$socket->connect(SERVER_HOST, SERVER_PORT, 5.0)) {
        return [
            'id' => $id,
            'error' => 'Failed to connect',
            'time' => microtime(true) - $startTime,
        ];
    }
    
    // Prepare request with a prompt that should generate many tokens
    $body = json_encode([
        'model' => 'test',
        'messages' => [
            ['role' => 'user', 'content' => 'Write a detailed essay about the history of computing, including at least 10 paragraphs covering the evolution from mechanical calculators to modern AI systems.']
        ],
        'stream' => true,
        'max_tokens' => 500,  // Request many tokens to ensure stream lasts
    ]);
    
    $request = "POST /v1/chat/completions HTTP/1.1\r\n";
    $request .= "Host: " . SERVER_HOST . ":" . SERVER_PORT . "\r\n";
    $request .= "Content-Type: application/json\r\n";
    $request .= "Accept: text/event-stream\r\n";
    $request .= "Content-Length: " . strlen($body) . "\r\n";
    $request .= "Connection: close\r\n";
    $request .= "\r\n";
    $request .= $body;
    
    $socket->send($request);
    
    // Read response headers first
    $headerBuffer = '';
    $headersComplete = false;
    
    while (!$headersComplete) {
        $data = $socket->recv(1024, 1.0);
        if ($data === false || $data === '') {
            break;
        }
        $headerBuffer .= $data;
        if (str_contains($headerBuffer, "\r\n\r\n")) {
            $headersComplete = true;
            // Extract body portion if any
            $parts = explode("\r\n\r\n", $headerBuffer, 2);
            if (isset($parts[1]) && $parts[1] !== '') {
                // Process any body data received with headers
                $bodyData = $parts[1];
                $lines = explode("\n", $bodyData);
                foreach ($lines as $line) {
                    if (str_starts_with($line, 'data: ')) {
                        $chunkData = substr($line, 6);
                        if ($chunkData !== '[DONE]') {
                            $chunksReceived++;
                            $parsed = json_decode($chunkData, true);
                            if (isset($parsed['choices'][0]['delta']['content'])) {
                                $content .= $parsed['choices'][0]['delta']['content'];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Continue reading chunks until we hit our limit
    while ($chunksReceived < $disconnectAfterChunks) {
        $data = $socket->recv(1024, 2.0);
        if ($data === false || $data === '') {
            break;
        }
        
        $lines = explode("\n", $data);
        foreach ($lines as $line) {
            if (str_starts_with($line, 'data: ')) {
                $chunkData = substr($line, 6);
                if ($chunkData === '[DONE]') {
                    // Stream finished before we could disconnect
                    $socket->close();
                    return [
                        'id' => $id,
                        'chunks' => $chunksReceived,
                        'content' => $content,
                        'disconnected' => false,
                        'message' => 'Stream completed before disconnect point',
                        'time' => microtime(true) - $startTime,
                    ];
                }
                $chunksReceived++;
                $parsed = json_decode($chunkData, true);
                if (isset($parsed['choices'][0]['delta']['content'])) {
                    $content .= $parsed['choices'][0]['delta']['content'];
                }
            }
        }
    }
    
    // ABRUPT DISCONNECT - close socket without reading remaining data
    $disconnectTime = microtime(true);
    echo sprintf("[Client %d] Disconnecting after %d chunks (%.3fs)\n", 
        $id, $chunksReceived, $disconnectTime - $startTime);
    
    $socket->close();
    
    return [
        'id' => $id,
        'chunks' => $chunksReceived,
        'content' => $content,
        'disconnected' => true,
        'time' => microtime(true) - $startTime,
    ];
}

/**
 * Send a complete request (no disconnect) for timing comparison
 * Uses raw socket to properly count streaming chunks
 */
function complete_request(int $id): array
{
    $startTime = microtime(true);
    $chunksReceived = 0;
    $content = '';
    
    $socket = new \Swoole\Coroutine\Socket(AF_INET, SOCK_STREAM, 0);
    
    if (!$socket->connect(SERVER_HOST, SERVER_PORT, 5.0)) {
        return [
            'id' => $id,
            'error' => 'Failed to connect',
            'chunks' => 0,
            'content' => '',
            'time' => microtime(true) - $startTime,
        ];
    }
    
    $body = json_encode([
        'model' => 'test',
        'messages' => [
            ['role' => 'user', 'content' => 'Write a detailed essay about the history of computing.']
        ],
        'stream' => true,
        'max_tokens' => 100,  // Reduced for faster baseline
    ]);
    
    $request = "POST /v1/chat/completions HTTP/1.1\r\n";
    $request .= "Host: " . SERVER_HOST . ":" . SERVER_PORT . "\r\n";
    $request .= "Content-Type: application/json\r\n";
    $request .= "Accept: text/event-stream\r\n";
    $request .= "Content-Length: " . strlen($body) . "\r\n";
    $request .= "Connection: close\r\n";
    $request .= "\r\n";
    $request .= $body;
    
    $socket->send($request);
    
    // Read and parse SSE stream until [DONE]
    $buffer = '';
    $done = false;
    
    while (!$done) {
        $data = $socket->recv(4096, 120.0);  // 120s timeout for full generation
        if ($data === false || $data === '') {
            break;
        }
        $buffer .= $data;
        
        // Process complete lines
        while (($pos = strpos($buffer, "\n")) !== false) {
            $line = substr($buffer, 0, $pos);
            $buffer = substr($buffer, $pos + 1);
            
            $line = trim($line);
            if (str_starts_with($line, 'data: ')) {
                $chunkData = substr($line, 6);
                if ($chunkData === '[DONE]') {
                    $done = true;
                    break;
                }
                $chunksReceived++;
                $parsed = json_decode($chunkData, true);
                if (isset($parsed['choices'][0]['delta']['content'])) {
                    $content .= $parsed['choices'][0]['delta']['content'];
                }
            }
        }
    }
    
    $socket->close();
    
    return [
        'id' => $id,
        'chunks' => $chunksReceived,
        'content' => $content,
        'time' => microtime(true) - $startTime,
    ];
}

// Main test
Co\run(function () {
    echo "=== Llama PHP HTTP Disconnect Cancellation Test ===\n\n";
    
    // Wait for server
    echo "Waiting for server at http://" . SERVER_HOST . ":" . SERVER_PORT . "...\n";
    if (!wait_for_server()) {
        echo "ERROR: Server not available. Make sure to start it with:\n";
        echo "  php test_http_server.php -m /path/to/model.gguf\n";
        return;
    }
    echo "Server is ready!\n\n";
    
    // Test 1: Baseline - complete a full request
    echo "--- Test 1: Baseline Complete Request ---\n";
    echo "Sending request and waiting for full completion...\n";
    $baseline = complete_request(0);
    echo sprintf("Complete request: %.3fs, %d chunks, %d chars\n\n",
        $baseline['time'], $baseline['chunks'], strlen($baseline['content']));
    
    // Test 2: Single disconnect test
    echo "--- Test 2: Single Disconnect Test ---\n";
    echo "Sending request and disconnecting after 2 chunks...\n";
    echo "(Watch server console for 'Client disconnected mid-stream, calling cancel()' message)\n\n";
    
    $abortResult = start_and_abort_request(1, 2);
    
    if ($abortResult['disconnected'] ?? false) {
        echo sprintf("Aborted after: %.3fs, %d chunks received\n",
            $abortResult['time'], $abortResult['chunks']);
        echo "Content before abort: " . substr($abortResult['content'], 0, 100) . "...\n\n";
        
        // The key test: if cancel() works, the server should stop generating
        // almost immediately. Wait a moment then check if server is responsive.
        echo "Waiting 1 second for server to process cancellation...\n";
        Co::sleep(1.0);
        
        // Quick health check to verify server is still responsive
        $client = new Client(SERVER_HOST, SERVER_PORT);
        $client->set(['timeout' => 2.0]);
        $client->get('/health');
        
        if ($client->statusCode === 200) {
            echo "✓ Server is responsive after disconnect (cancel() likely worked)\n\n";
        } else {
            echo "⚠ Server may be stuck (cancel() may not have worked)\n\n";
        }
        $client->close();
    } else {
        echo "⚠ Request completed before disconnect point: " . ($abortResult['message'] ?? '') . "\n\n";
    }
    
    // Test 3: Multiple concurrent disconnects
    echo "--- Test 3: Multiple Concurrent Disconnects ---\n";
    echo "Starting 3 parallel requests, each will disconnect after 2-3 chunks...\n\n";
    
    $wg = new WaitGroup();
    $results = [];
    
    for ($i = 0; $i < 3; $i++) {
        $wg->add();
        $disconnectAfter = 2 + $i;  // Disconnect after 2, 3, 4 chunks
        go(function () use ($i, $disconnectAfter, &$results, $wg) {
            $results[$i] = start_and_abort_request($i, $disconnectAfter);
            $wg->done();
        });
    }
    
    $wg->wait();
    
    echo "\nResults:\n";
    foreach ($results as $r) {
        $status = ($r['disconnected'] ?? false) ? 'DISCONNECTED' : 'COMPLETED';
        echo sprintf("  Client %d: %s after %.3fs, %d chunks\n",
            $r['id'], $status, $r['time'], $r['chunks']);
    }
    
    // Final health check
    Co::sleep(0.5);
    $client = new Client(SERVER_HOST, SERVER_PORT);
    $client->set(['timeout' => 2.0]);
    $client->get('/health');
    
    if ($client->statusCode === 200) {
        echo "\n✓ Server healthy after all disconnects\n";
    } else {
        echo "\n⚠ Server may have issues after disconnects\n";
    }
    $client->close();
    
    echo "\n=== Disconnect Test Complete ===\n";
    echo "\nVERIFICATION: Check server console for 'Client disconnected mid-stream' messages.\n";
    echo "Each disconnect should trigger a cancel() call, visible in server output.\n";
});
