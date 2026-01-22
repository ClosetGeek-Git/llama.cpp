<?php
/**
 * Memory leak test - runs continuous streaming requests
 * Use with valgrind or watch process memory to verify no leaks
 */

if (!extension_loaded('swoole_llama')) {
    die("Error: swoole_llama extension not loaded\n");
}

const BATCH_SIZE = 1;
const ITERATIONS = 1;
const REPORT_EVERY = 1;

\Co\run(function () {
    global $argv;
    
    $modelIdx = array_search('--model', $argv);
    if ($modelIdx === false || !isset($argv[$modelIdx + 1])) {
        echo "Usage: php test_memory_leak.php --model /path/to/model.gguf [other args]\n";
        return;
    }
    
    if (!swoole_llama_init($argv)) {
        echo "ERROR: swoole_llama_init() failed\n";
        return;
    }
    
    while (swoole_llama_ready() === 0) {
        \Co::sleep(0.1);
    }
    if (swoole_llama_ready() === -1) {
        echo "ERROR: Model loading failed\n";
        return;
    }
    
    echo "=== Memory Leak Test ===\n";
    echo "Iterations: " . ITERATIONS . " x " . BATCH_SIZE . " = " . (ITERATIONS * BATCH_SIZE) . " requests\n";
    echo "PHP Memory at start: " . number_format(memory_get_usage(true) / 1024) . " KB\n\n";
    
    $prompts = [
        "Count from 1 to 5",
        "Name 3 colors",
        "List 2 fruits",
        "Say hello",
    ];
    
    $start_time = microtime(true);
    $total_chunks = 0;
    $total_requests = 0;
    
    for ($iter = 0; $iter < ITERATIONS; $iter++) {
        $wg = new \Co\WaitGroup();
        $batch_chunks = 0;
        
        for ($i = 0; $i < BATCH_SIZE; $i++) {
            $wg->add();
            go(function () use ($i, $prompts, $wg, &$batch_chunks) {
                $prompt = $prompts[$i % count($prompts)];
                $chunks = 0;
                
                $params = [
                    'method' => 'POST',
                    'path' => '/v1/chat/completions',
                    'body' => json_encode([
                        'model' => 'llama',
                        'messages' => [['role' => 'user', 'content' => $prompt]],
                        'max_tokens' => 10,
                        'temperature' => 0.0,
                        'stream' => true,
                    ]),
                    'headers' => ['Content-Type' => ['application/json']],
                ];
                
                try {
                    $req = new \Llama\Request($params);
                    
                    $firstData = $req->getData();
                    if ($firstData !== null) {
                        $chunks++;
                    }
                    
                    while (($chunk = $req->next()) !== null) {
                        $chunks++;
                    }
                    
                    $batch_chunks += $chunks;
                } catch (Exception $e) {
                    echo "ERROR: " . $e->getMessage() . "\n";
                }
                $wg->done();
            });
        }
        
        $wg->wait();
        $total_chunks += $batch_chunks;
        $total_requests += BATCH_SIZE;
        
        if (($iter + 1) % REPORT_EVERY === 0) {
            $elapsed = microtime(true) - $start_time;
            $php_mem = memory_get_usage(true) / 1024;
            $php_peak = memory_get_peak_usage(true) / 1024;
            
            printf("Iter %d/%d: %d requests, %d chunks, PHP: %.0f KB (peak: %.0f KB), %.1f req/s\n",
                $iter + 1, ITERATIONS, $total_requests, $total_chunks,
                $php_mem, $php_peak, $total_requests / $elapsed);
        }
    }
    
    $elapsed = microtime(true) - $start_time;
    
    echo "\n=== Final Report ===\n";
    echo "Total requests: $total_requests\n";
    echo "Total chunks: $total_chunks\n";
    echo "Total time: " . number_format($elapsed, 2) . "s\n";
    echo "Requests/sec: " . number_format($total_requests / $elapsed, 2) . "\n";
    echo "PHP Memory: " . number_format(memory_get_usage(true) / 1024) . " KB\n";
    echo "PHP Peak: " . number_format(memory_get_peak_usage(true) / 1024) . " KB\n";
    
    gc_collect_cycles();
    echo "After GC: " . number_format(memory_get_usage(true) / 1024) . " KB\n";
    
    swoole_llama_shutdown();
});
