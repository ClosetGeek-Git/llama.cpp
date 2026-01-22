<?php
/**
 * Concurrent HTTP Client Test for Llama PHP Server
 * 
 * Tests that multiple parallel requests are properly batched through
 * the llama inference loop, resulting in better throughput than
 * sequential processing.
 * 
 * Usage:
 *   1. Start the server: php test_http_server.php -m /path/to/model.gguf
 *   2. Run this test: php test_http_client.php
 */

declare(strict_types=1);

use Swoole\Coroutine as Co;
use Swoole\Coroutine\Http\Client;
use Swoole\Coroutine\WaitGroup;

const SERVER_HOST = '127.0.0.1';
const SERVER_PORT = 9501;
const NUM_CONCURRENT_REQUESTS = 4;

/**
 * Send a single chat completion request and measure time
 */
function send_chat_request(int $id, string $prompt, bool $stream = true): array
{
    $startTime = microtime(true);
    
    $client = new Client(SERVER_HOST, SERVER_PORT);
    $client->setHeaders([
        'Content-Type' => 'application/json',
        'Accept' => 'text/event-stream',
    ]);
    
    $body = json_encode([
        'model' => 'test',
        'messages' => [
            ['role' => 'user', 'content' => $prompt]
        ],
        'stream' => $stream,
        'max_tokens' => 50,
    ]);
    
    $client->post('/v1/chat/completions', $body);
    
    $content = '';
    $chunkCount = 0;
    
    if ($stream && $client->statusCode === 200) {
        // For streaming, we need to read the response body which contains SSE data
        $responseBody = $client->body;
        
        // Parse SSE chunks from response
        $lines = explode("\n", $responseBody);
        foreach ($lines as $line) {
            if (str_starts_with($line, 'data: ')) {
                $data = substr($line, 6);
                if ($data === '[DONE]') {
                    break;
                }
                $chunkCount++;
                $parsed = json_decode($data, true);
                if (isset($parsed['choices'][0]['delta']['content'])) {
                    $content .= $parsed['choices'][0]['delta']['content'];
                }
            }
        }
    } else {
        // Non-streaming response
        $parsed = json_decode($client->body, true);
        if (isset($parsed['choices'][0]['message']['content'])) {
            $content = $parsed['choices'][0]['message']['content'];
        }
        $chunkCount = 1;
    }
    
    $endTime = microtime(true);
    $client->close();
    
    return [
        'id' => $id,
        'status' => $client->statusCode,
        'content' => $content,
        'chunks' => $chunkCount,
        'time' => $endTime - $startTime,
    ];
}

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

// Main test
Co\run(function () {
    echo "=== Llama PHP HTTP Client Test ===\n\n";
    
    // Wait for server
    echo "Waiting for server at http://" . SERVER_HOST . ":" . SERVER_PORT . "...\n";
    if (!wait_for_server()) {
        echo "ERROR: Server not available. Make sure to start it with:\n";
        echo "  php test_http_server.php -m /path/to/model.gguf\n";
        return;
    }
    echo "Server is ready!\n\n";
    
    // Test 1: Single request baseline
    echo "--- Test 1: Single Request Baseline ---\n";
    $singleStart = microtime(true);
    $result = send_chat_request(0, "Count from 1 to 5", true);
    $singleTime = microtime(true) - $singleStart;
    echo sprintf("Single request: %.3fs, status=%d, chunks=%d\n", 
        $result['time'], $result['status'], $result['chunks']);
    echo "Response: " . substr($result['content'], 0, 100) . "...\n\n";
    
    // Test 2: Concurrent requests (should benefit from batching)
    echo "--- Test 2: Concurrent Requests (N=" . NUM_CONCURRENT_REQUESTS . ") ---\n";
    
    $prompts = [
        "What is 2+2?",
        "Name three colors.",
        "What is the capital of France?",
        "Count to 3.",
    ];
    
    $results = [];
    $wg = new WaitGroup();
    $concurrentStart = microtime(true);
    
    for ($i = 0; $i < NUM_CONCURRENT_REQUESTS; $i++) {
        $wg->add();
        go(function () use ($i, $prompts, &$results, $wg) {
            $results[$i] = send_chat_request($i, $prompts[$i % count($prompts)], true);
            $wg->done();
        });
    }
    
    $wg->wait();
    $concurrentTime = microtime(true) - $concurrentStart;
    
    // Print results
    foreach ($results as $r) {
        echo sprintf("  Request %d: %.3fs, status=%d, chunks=%d\n",
            $r['id'], $r['time'], $r['status'], $r['chunks']);
    }
    
    echo sprintf("\nTotal concurrent time: %.3fs\n", $concurrentTime);
    echo sprintf("Expected sequential time: %.3fs (N × single)\n", $singleTime * NUM_CONCURRENT_REQUESTS);
    
    // Calculate speedup
    $expectedSequential = $singleTime * NUM_CONCURRENT_REQUESTS;
    $speedup = $expectedSequential / $concurrentTime;
    echo sprintf("Speedup factor: %.2fx\n", $speedup);
    
    if ($speedup > 1.2) {
        echo "✓ Batching appears to be working (speedup > 1.2x)\n";
    } else {
        echo "⚠ Batching may not be effective (speedup <= 1.2x)\n";
    }
    
    // Test 3: Non-streaming request
    echo "\n--- Test 3: Non-Streaming Request ---\n";
    $result = send_chat_request(0, "Say hello in one word", false);
    echo sprintf("Non-streaming: %.3fs, status=%d\n", $result['time'], $result['status']);
    echo "Response: " . $result['content'] . "\n";
    
    echo "\n=== Tests Complete ===\n";
});
