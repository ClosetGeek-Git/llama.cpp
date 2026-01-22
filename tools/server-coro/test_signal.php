<?php
/**
 * Test signal handling during streaming requests
 * 
 * Usage:
 *   LD_LIBRARY_PATH=/home/jason-dev/swoole/llama.cpp/build/bin \
 *   php -dextension=modules/swoole_llama.so test_signal.php \
 *       --model /home/jason-dev/swoole/qwen.gguf --ctx-size 512 --parallel 1 --log-disable
 *
 * Then send SIGINT (Ctrl+C) or SIGTERM during streaming to test graceful shutdown
 */

use Swoole\Coroutine;
use Llama\Request;

Coroutine::set(['hook_flags' => SWOOLE_HOOK_ALL]);

echo "=== Signal Handling Test ===\n\n";
echo "This test runs a long streaming operation.\n";
echo "Press Ctrl+C (SIGINT) or send SIGTERM to test graceful shutdown.\n\n";

$shutdown_requested = false;

// Install signal handlers
pcntl_async_signals(true);

pcntl_signal(SIGINT, function() use (&$shutdown_requested) {
    echo "\n[SIGINT received - initiating graceful shutdown]\n";
    $shutdown_requested = true;
});

pcntl_signal(SIGTERM, function() use (&$shutdown_requested) {
    echo "\n[SIGTERM received - initiating graceful shutdown]\n";
    $shutdown_requested = true;
});

Coroutine\run(function() use (&$shutdown_requested) {
    global $argv;
    
    echo "Loading model...\n";
    if (!swoole_llama_init($argv)) {
        echo "Failed to initialize\n";
        return;
    }
    
    while (!swoole_llama_ready()) {
        Coroutine::sleep(0.1);
    }
    echo "Model loaded.\n\n";
    
    $total_chunks = 0;
    $request_num = 0;
    
    echo "Starting continuous streaming (100 chunks per request, infinite loop)...\n";
    echo "Send SIGINT (Ctrl+C) or SIGTERM to test shutdown.\n\n";
    
    while (!$shutdown_requested) {
        $request_num++;
        echo "Request $request_num: ";
        
        $req = new Request([
            'method' => 'POST',
            'path' => '/test/stream',
            'body' => json_encode([
                'n_chunks' => 100,
                'stream' => true
            ]),
            'headers' => ['content-type' => ['application/json']],
        ]);
        
        $chunks = 0;
        while (($chunk = $req->next()) !== null) {
            $chunks++;
            $total_chunks++;
            
            // Check for shutdown between chunks
            if ($shutdown_requested) {
                echo "interrupted at chunk $chunks";
                break;
            }
            
            // Small delay to make it interruptible
            Coroutine::sleep(0.01);
        }
        
        if (!$shutdown_requested) {
            echo "$chunks chunks\n";
        }
        
        // Brief pause between requests
        Coroutine::sleep(0.1);
    }
    
    echo "\n\n=== Shutdown Summary ===\n";
    echo "Total requests started: $request_num\n";
    echo "Total chunks received: $total_chunks\n";
    
    echo "\nCalling swoole_llama_shutdown()...\n";
    $start = microtime(true);
    swoole_llama_shutdown();
    $elapsed = (microtime(true) - $start) * 1000;
    printf("Shutdown completed in %.2f ms\n", $elapsed);
    
    echo "\n=== Clean shutdown successful ===\n";
});

echo "\nProcess exiting normally.\n";
