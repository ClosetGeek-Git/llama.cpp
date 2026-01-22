<?php
/**
 * Test concurrent streaming requests to verify batching
 */

if (!extension_loaded('swoole_llama')) {
    die("Error: swoole_llama extension not loaded\n");
}

\Co\run(function () {
    global $argv;
    
    $modelIdx = array_search('--model', $argv);
    if ($modelIdx === false || !isset($argv[$modelIdx + 1])) {
        echo "Usage: php test_concurrent.php --model /path/to/model.gguf [other args]\n";
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
    echo "Model loaded\n\n";
    
    // Launch 4 concurrent streaming requests
    $prompts = [
        "Count from 1 to 10",
        "List the days of the week",
        "Name 5 colors",
        "Count backwards from 5 to 1",
    ];
    
    $wg = new \Co\WaitGroup();
    $results = [];
    
    foreach ($prompts as $i => $prompt) {
        $wg->add();
        go(function () use ($i, $prompt, $wg, &$results) {
            $start = microtime(true);
            $params = [
                'method' => 'POST',
                'path' => '/v1/chat/completions',
                'body' => json_encode([
                    'model' => 'llama',
                    'messages' => [['role' => 'user', 'content' => $prompt]],
                    'max_tokens' => 64,
                    'temperature' => 0.0,
                    'stream' => true,
                ]),
                'headers' => ['Content-Type' => ['application/json']],
            ];
            
            try {
                $req = new \Llama\Request($params);
                $content = '';
                $chunks = 0;
                
                $firstData = $req->getData();
                if ($firstData !== null) {
                    $chunks++;
                    $data = json_decode($firstData, true);
                    if (isset($data['choices'][0]['delta']['reasoning_content'])) {
                        $content .= $data['choices'][0]['delta']['reasoning_content'];
                    }
                    if (isset($data['choices'][0]['delta']['content'])) {
                        $content .= $data['choices'][0]['delta']['content'];
                    }
                }
                
                while (($chunk = $req->next()) !== null) {
                    $chunks++;
                    if (isset($chunk['choices'][0]['delta']['reasoning_content'])) {
                        $content .= $chunk['choices'][0]['delta']['reasoning_content'];
                    }
                    if (isset($chunk['choices'][0]['delta']['content'])) {
                        $content .= $chunk['choices'][0]['delta']['content'];
                    }
                }
                
                $elapsed = microtime(true) - $start;
                $results[$i] = [
                    'prompt' => $prompt,
                    'chunks' => $chunks,
                    'time' => round($elapsed * 1000),
                    'content' => substr($content, 0, 100),
                ];
            } catch (Exception $e) {
                $results[$i] = ['prompt' => $prompt, 'error' => $e->getMessage()];
            }
            $wg->done();
        });
    }
    
    $wg->wait();
    
    echo "=== Results ===\n";
    ksort($results);
    foreach ($results as $i => $r) {
        echo "Request $i: \"{$r['prompt']}\"\n";
        if (isset($r['error'])) {
            echo "  ERROR: {$r['error']}\n";
        } else {
            echo "  Chunks: {$r['chunks']}, Time: {$r['time']}ms\n";
            echo "  Content: {$r['content']}...\n";
        }
        echo "\n";
    }
    
    swoole_llama_shutdown();
});
