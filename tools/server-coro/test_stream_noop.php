<?php
/**
 * Test script for /test/stream endpoint - no model inference, tests full PHP extension path
 * 
 * Usage:
 *   LD_LIBRARY_PATH=/home/jason-dev/swoole/llama.cpp/build/bin \
 *   php -dextension=modules/swoole_llama.so test_stream_noop.php \
 *       --model /home/jason-dev/swoole/qwen.gguf --ctx-size 512 --parallel 1 --log-disable
 */

use Swoole\Coroutine;
use Llama\Request;

Coroutine::set(['hook_flags' => SWOOLE_HOOK_ALL]);

echo "=== Test Stream No-Op Endpoint ===\n\n";

Coroutine\run(function() {
    global $argv;
    
    // Initialize
    echo "Loading model...\n";
    if (!swoole_llama_init($argv)) {
        echo "Failed to initialize\n";
        return;
    }
    
    // Wait for model load
    while (!swoole_llama_ready()) {
        Coroutine::sleep(0.1);
    }
    echo "Model loaded.\n\n";
    
    // Test 1: Non-streaming
    echo "--- Test 1: Non-streaming (stream=false) ---\n";
    $request = new Request([
        'method' => 'POST',
        'path' => '/test/stream',
        'body' => json_encode([
            'n_chunks' => 5,
            'stream' => false
        ]),
        'headers' => ['content-type' => ['application/json']],
    ]);
    
    echo "isStream: " . ($request->isStream() ? "true" : "false") . "\n";
    $result = $request->getData();
    echo "Result: $result\n\n";
    
    // Test 2: Streaming
    echo "--- Test 2: Streaming (stream=true) ---\n";
    $request2 = new Request([
        'method' => 'POST',
        'path' => '/test/stream',
        'body' => json_encode([
            'n_chunks' => 5,
            'stream' => true
        ]),
        'headers' => ['content-type' => ['application/json']],
    ]);
    
    echo "isStream: " . ($request2->isStream() ? "true" : "false") . "\n";
    
    $chunk_count = 0;
    while (($chunk = $request2->next()) !== null) {
        $chunk_count++;
        echo "Chunk $chunk_count: " . json_encode($chunk) . "\n";
    }
    echo "Total chunks received: $chunk_count\n\n";
    
    // Test 3: Concurrent streaming requests
    echo "--- Test 3: Concurrent streaming requests ---\n";
    
    $n_concurrent = 4;
    $wg = new Coroutine\WaitGroup();
    
    for ($i = 0; $i < $n_concurrent; $i++) {
        $wg->add();
        Coroutine::create(function() use ($i, $wg) {
            $req = new Request([
                'method' => 'POST',
                'path' => '/test/stream',
                'body' => json_encode([
                    'n_chunks' => 3,
                    'stream' => true
                ]),
                'headers' => ['content-type' => ['application/json']],
            ]);
            
            $chunks = [];
            while (($chunk = $req->next()) !== null) {
                $chunks[] = $chunk;
            }
            echo "  Request $i: received " . count($chunks) . " chunks\n";
            $wg->done();
        });
    }
    
    $wg->wait();
    echo "All concurrent requests completed.\n\n";
    
    // Test 4: High-volume streaming for memory leak testing
    echo "--- Test 4: High-volume streaming (50 requests) ---\n";
    
    gc_collect_cycles();
    $rss_before = file_exists('/proc/self/status') 
        ? (int)(preg_match('/VmRSS:\s+(\d+)/', file_get_contents('/proc/self/status'), $m) 
            ? $m[1] : 0)
        : 0;
    
    $total_chunks = 0;
    for ($i = 0; $i < 50; $i++) {
        $req = new Request([
            'method' => 'POST',
            'path' => '/test/stream',
            'body' => json_encode([
                'n_chunks' => 10,
                'stream' => true
            ]),
            'headers' => ['content-type' => ['application/json']],
        ]);
        
        while (($chunk = $req->next()) !== null) {
            $total_chunks++;
        }
        
        if (($i + 1) % 10 === 0) {
            gc_collect_cycles();
            echo "  Completed " . ($i + 1) . " requests, total chunks: $total_chunks\n";
        }
    }
    
    gc_collect_cycles();
    $rss_after = file_exists('/proc/self/status') 
        ? (int)(preg_match('/VmRSS:\s+(\d+)/', file_get_contents('/proc/self/status'), $m) 
            ? $m[1] : 0)
        : 0;
    
    echo "Total chunks: $total_chunks\n";
    if ($rss_before > 0 && $rss_after > 0) {
        $diff = $rss_after - $rss_before;
        echo "RSS change: " . ($diff >= 0 ? "+$diff" : $diff) . " KB\n";
    }
    
    echo "\n=== All tests passed ===\n";
    
    swoole_llama_shutdown();
});
