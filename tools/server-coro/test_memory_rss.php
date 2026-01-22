<?php
/**
 * Fast memory leak detection via RSS monitoring (no valgrind needed)
 * If RSS grows linearly with iterations, there's a leak.
 */

use Swoole\Coroutine;
use Llama\Request;

const ITERATIONS = 50;
const BATCH_SIZE = 4;  // requests per batch

function get_rss_kb(): int {
    // Read from /proc/self/status
    $status = file_get_contents('/proc/self/status');
    if (preg_match('/VmRSS:\s+(\d+)\s+kB/', $status, $m)) {
        return (int)$m[1];
    }
    return 0;
}

Coroutine::set(['hook_flags' => SWOOLE_HOOK_ALL]);

Coroutine\run(function() {
    global $argc, $argv;
    
    echo "=== RSS Memory Leak Test ===\n";
    echo "Iterations: " . ITERATIONS . "\n";
    echo "Batch size: " . BATCH_SIZE . "\n\n";
    
    $rss_before_init = get_rss_kb();
    echo "RSS before init: " . number_format($rss_before_init) . " KB\n";
    
    // Initialize - pass raw argv like test_memory_leak.php does
    // The extension expects the full argv including program name
    if (!swoole_llama_init($argv)) {
        echo "Failed to initialize\n";
        return;
    }
    
    // Wait for model load
    while (!swoole_llama_ready()) {
        Coroutine::sleep(0.1);
    }
    
    $rss_after_init = get_rss_kb();
    echo "RSS after init:  " . number_format($rss_after_init) . " KB\n";
    echo "Model overhead:  " . number_format($rss_after_init - $rss_before_init) . " KB\n\n";
    
    // Force GC
    gc_collect_cycles();
    $rss_baseline = get_rss_kb();
    echo "RSS baseline (after GC): " . number_format($rss_baseline) . " KB\n\n";
    
    $samples = [];
    $total_requests = 0;
    $total_chunks = 0;
    
    for ($iter = 1; $iter <= ITERATIONS; $iter++) {
        $batch_chunks = 0;
        
        // Run batch of concurrent requests
        $wg = new Coroutine\WaitGroup();
        for ($b = 0; $b < BATCH_SIZE; $b++) {
            $wg->add();
            Coroutine::create(function() use ($wg, &$batch_chunks) {
                $req = new Request([
                    'method' => 'POST',
                    'path' => '/v1/chat/completions',
                    'body' => json_encode([
                        'messages' => [['role' => 'user', 'content' => 'Say hi']],
                        'max_tokens' => 10,
                        'stream' => true,
                    ]),
                    'headers' => [
                        'content-type' => ['application/json'],
                        'x-response-type' => ['raw'],
                    ],
                ]);
                
                $chunks = 0;
                while (($chunk = $req->next()) !== null) {
                    $chunks++;
                }
                
                $batch_chunks += $chunks;
                $wg->done();
            });
        }
        $wg->wait();
        
        $total_requests += BATCH_SIZE;
        $total_chunks += $batch_chunks;
        
        // Force GC and measure RSS
        gc_collect_cycles();
        $rss = get_rss_kb();
        $samples[] = $rss;
        
        // Print progress every 5 iterations
        if ($iter % 5 === 0 || $iter === 1) {
            $growth = $rss - $rss_baseline;
            $per_req = $total_requests > 0 ? $growth / $total_requests : 0;
            printf("Iter %3d: RSS=%s KB (+%s KB, ~%.1f KB/req)\n",
                $iter,
                number_format($rss),
                number_format($growth),
                $per_req
            );
        }
    }
    
    echo "\n=== Summary ===\n";
    echo "Total requests: $total_requests\n";
    echo "Total chunks:   $total_chunks\n";
    
    $rss_final = get_rss_kb();
    $total_growth = $rss_final - $rss_baseline;
    $per_request_kb = $total_requests > 0 ? $total_growth / $total_requests : 0;
    
    echo "RSS growth:     " . number_format($total_growth) . " KB\n";
    printf("Per request:    %.2f KB\n", $per_request_kb);
    
    // Linear regression on samples to detect leak trend
    $n = count($samples);
    if ($n >= 5) {
        $sum_x = $sum_y = $sum_xy = $sum_x2 = 0;
        for ($i = 0; $i < $n; $i++) {
            $x = $i + 1;
            $y = $samples[$i];
            $sum_x += $x;
            $sum_y += $y;
            $sum_xy += $x * $y;
            $sum_x2 += $x * $x;
        }
        $slope = ($n * $sum_xy - $sum_x * $sum_y) / ($n * $sum_x2 - $sum_x * $sum_x);
        $slope_per_req = $slope / BATCH_SIZE;
        
        printf("\nLeak trend:     %.2f KB/iteration (%.2f KB/request)\n", $slope, $slope_per_req);
        
        if (abs($slope_per_req) < 1.0) {
            echo "Result:         PASS (no significant leak detected)\n";
        } else if ($slope_per_req > 0) {
            echo "Result:         FAIL (memory growing ~" . number_format($slope_per_req, 1) . " KB/request)\n";
        } else {
            echo "Result:         OK (memory stable or shrinking)\n";
        }
    }
    
    echo "\nShutting down...\n";
    swoole_llama_shutdown();
    
    $rss_after = get_rss_kb();
    echo "RSS after shutdown: " . number_format($rss_after) . " KB\n";
    echo "Freed: " . number_format($rss_final - $rss_after) . " KB\n";
});
