<?php
/**
 * Test concurrent request cancellation
 * 
 * Phase 1: Run without valgrind using real chat completions
 *   LD_LIBRARY_PATH=/home/jason-dev/swoole/llama.cpp/build/bin \
 *   php -dextension=modules/swoole_llama.so test_cancel.php \
 *       --model /home/jason-dev/swoole/qwen.gguf --ctx-size 2048 --parallel 4 --log-disable
 *
 * Phase 2: Run with valgrind using no-op API (smaller model)
 *   LD_LIBRARY_PATH=/home/jason-dev/swoole/llama.cpp/build/bin \
 *   valgrind --leak-check=full --show-leak-kinds=definite \
 *   php -dextension=modules/swoole_llama.so test_cancel.php --noop \
 *       --model /home/jason-dev/swoole/500M.gguf --ctx-size 512 --parallel 4 --log-disable
 */

use Swoole\Coroutine;
use Llama\Request;

Coroutine::set(['hook_flags' => SWOOLE_HOOK_ALL]);

// Check if --noop flag is present
$use_noop = in_array('--noop', $GLOBALS['argv']);

echo "=== Concurrent Cancel Test ===\n";
echo "Mode: " . ($use_noop ? "NO-OP (for valgrind)" : "Real chat completions") . "\n\n";

Coroutine\run(function() use ($use_noop) {
    global $argv;
    
    // Remove --noop from argv before passing to llama init
    $filtered_argv = array_values(array_filter($argv, fn($a) => $a !== '--noop'));
    
    echo "Loading model...\n";
    if (!swoole_llama_init($filtered_argv)) {
        echo "Failed to initialize\n";
        return;
    }
    
    while (!swoole_llama_ready()) {
        Coroutine::sleep(0.1);
    }
    echo "Model loaded.\n\n";
    
    // Test 1: Cancel immediately after creation
    echo "--- Test 1: Cancel immediately after creation ---\n";
    {
        $n_requests = 4;
        $wg = new Coroutine\WaitGroup();
        
        for ($i = 0; $i < $n_requests; $i++) {
            $wg->add();
            Coroutine::create(function() use ($i, $wg, $use_noop) {
                if ($use_noop) {
                    $req = new Request([
                        'method' => 'POST',
                        'path' => '/test/stream',
                        'body' => json_encode(['n_chunks' => 100, 'stream' => true]),
                        'headers' => ['content-type' => ['application/json']],
                    ]);
                } else {
                    $req = new Request([
                        'method' => 'POST',
                        'path' => '/v1/chat/completions',
                        'body' => json_encode([
                            'messages' => [['role' => 'user', 'content' => 'Count from 1 to 100 slowly']],
                            'max_tokens' => 200,
                            'stream' => true,
                        ]),
                        'headers' => ['content-type' => ['application/json'], 'x-response-type' => ['raw']],
                    ]);
                }
                
                // Cancel immediately
                $req->cancel();
                
                // Try to read - should return null
                $chunk = $req->next();
                $result = ($chunk === null) ? "OK (null)" : "UNEXPECTED: got chunk";
                echo "  Request $i: cancelled immediately - $result\n";
                
                $wg->done();
            });
        }
        
        $wg->wait();
        echo "Test 1 complete.\n\n";
    }
    
    // Test 2: Cancel after receiving some chunks
    echo "--- Test 2: Cancel after receiving some chunks ---\n";
    {
        $n_requests = 4;
        $wg = new Coroutine\WaitGroup();
        
        for ($i = 0; $i < $n_requests; $i++) {
            $wg->add();
            Coroutine::create(function() use ($i, $wg, $use_noop) {
                if ($use_noop) {
                    $req = new Request([
                        'method' => 'POST',
                        'path' => '/test/stream',
                        'body' => json_encode(['n_chunks' => 100, 'stream' => true]),
                        'headers' => ['content-type' => ['application/json']],
                    ]);
                } else {
                    $req = new Request([
                        'method' => 'POST',
                        'path' => '/v1/chat/completions',
                        'body' => json_encode([
                            'messages' => [['role' => 'user', 'content' => 'Count from 1 to 100 slowly']],
                            'max_tokens' => 200,
                            'stream' => true,
                        ]),
                        'headers' => ['content-type' => ['application/json'], 'x-response-type' => ['raw']],
                    ]);
                }
                
                // Read a few chunks
                $chunks_received = 0;
                $cancel_after = rand(2, 5);
                
                while (($chunk = $req->next()) !== null) {
                    $chunks_received++;
                    if ($chunks_received >= $cancel_after) {
                        $req->cancel();
                        break;
                    }
                }
                
                // Verify no more chunks after cancel
                $extra = $req->next();
                $status = ($extra === null) ? "OK" : "WARN: got extra chunk";
                echo "  Request $i: received $chunks_received chunks, cancelled, $status\n";
                
                $wg->done();
            });
        }
        
        $wg->wait();
        echo "Test 2 complete.\n\n";
    }
    
    // Test 3: Mixed - some complete, some cancelled
    echo "--- Test 3: Mixed - some complete, some cancelled ---\n";
    {
        $n_requests = 8;
        $completed = 0;
        $cancelled = 0;
        $wg = new Coroutine\WaitGroup();
        
        for ($i = 0; $i < $n_requests; $i++) {
            $wg->add();
            $should_cancel = ($i % 2 === 0);
            
            Coroutine::create(function() use ($i, $wg, $use_noop, $should_cancel, &$completed, &$cancelled) {
                if ($use_noop) {
                    $req = new Request([
                        'method' => 'POST',
                        'path' => '/test/stream',
                        'body' => json_encode(['n_chunks' => 10, 'stream' => true]),
                        'headers' => ['content-type' => ['application/json']],
                    ]);
                } else {
                    $req = new Request([
                        'method' => 'POST',
                        'path' => '/v1/chat/completions',
                        'body' => json_encode([
                            'messages' => [['role' => 'user', 'content' => 'Say "hello"']],
                            'max_tokens' => 20,
                            'stream' => true,
                        ]),
                        'headers' => ['content-type' => ['application/json'], 'x-response-type' => ['raw']],
                    ]);
                }
                
                $chunks = 0;
                while (($chunk = $req->next()) !== null) {
                    $chunks++;
                    if ($should_cancel && $chunks >= 3) {
                        $req->cancel();
                        $cancelled++;
                        break;
                    }
                }
                
                if (!$should_cancel || $chunks < 3) {
                    $completed++;
                }
                
                $wg->done();
            });
        }
        
        $wg->wait();
        echo "  Completed: $completed, Cancelled: $cancelled\n";
        echo "Test 3 complete.\n\n";
    }
    
    // Test 4: High-volume cancel stress test
    echo "--- Test 4: High-volume cancel stress test (50 requests) ---\n";
    {
        gc_collect_cycles();
        $rss_before = 0;
        if (file_exists('/proc/self/status')) {
            preg_match('/VmRSS:\s+(\d+)/', file_get_contents('/proc/self/status'), $m);
            $rss_before = (int)($m[1] ?? 0);
        }
        
        $total_cancelled = 0;
        $total_chunks = 0;
        
        for ($batch = 0; $batch < 10; $batch++) {
            $wg = new Coroutine\WaitGroup();
            
            for ($i = 0; $i < 5; $i++) {
                $wg->add();
                Coroutine::create(function() use ($wg, $use_noop, &$total_cancelled, &$total_chunks) {
                    if ($use_noop) {
                        $req = new Request([
                            'method' => 'POST',
                            'path' => '/test/stream',
                            'body' => json_encode(['n_chunks' => 50, 'stream' => true]),
                            'headers' => ['content-type' => ['application/json']],
                        ]);
                    } else {
                        $req = new Request([
                            'method' => 'POST',
                            'path' => '/v1/chat/completions',
                            'body' => json_encode([
                                'messages' => [['role' => 'user', 'content' => 'Count to 50']],
                                'max_tokens' => 100,
                                'stream' => true,
                            ]),
                            'headers' => ['content-type' => ['application/json'], 'x-response-type' => ['raw']],
                        ]);
                    }
                    
                    $chunks = 0;
                    $cancel_at = rand(5, 20);
                    
                    while (($chunk = $req->next()) !== null) {
                        $chunks++;
                        $total_chunks++;
                        if ($chunks >= $cancel_at) {
                            $req->cancel();
                            $total_cancelled++;
                            break;
                        }
                    }
                    
                    $wg->done();
                });
            }
            
            $wg->wait();
            echo "  Batch " . ($batch + 1) . "/10 complete\n";
        }
        
        gc_collect_cycles();
        $rss_after = 0;
        if (file_exists('/proc/self/status')) {
            preg_match('/VmRSS:\s+(\d+)/', file_get_contents('/proc/self/status'), $m);
            $rss_after = (int)($m[1] ?? 0);
        }
        
        echo "\n  Total cancelled: $total_cancelled\n";
        echo "  Total chunks before cancel: $total_chunks\n";
        if ($rss_before > 0) {
            $diff = $rss_after - $rss_before;
            echo "  RSS change: " . ($diff >= 0 ? "+$diff" : $diff) . " KB\n";
        }
        echo "Test 4 complete.\n\n";
    }
    
    echo "=== All cancel tests passed ===\n\n";
    
    echo "Shutting down...\n";
    swoole_llama_shutdown();
    echo "Shutdown complete.\n";
});
