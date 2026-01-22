<?php
/**
 * Basic test to verify swoole_llama extension loads and initializes llama.cpp
 * 
 * Usage: php test_init.php --model /path/to/model.gguf [--ctx-size 4096] [--parallel 2]
 */

// Check extension loaded
if (!extension_loaded('swoole_llama')) {
    die("Error: swoole_llama extension not loaded\n");
}

if (!extension_loaded('swoole')) {
    die("Error: swoole extension not loaded\n");
}

echo "Extensions loaded: swoole, swoole_llama\n";
echo "Functions available:\n";
echo "  - swoole_llama_init: " . (function_exists('swoole_llama_init') ? 'yes' : 'no') . "\n";
echo "  - swoole_llama_ready: " . (function_exists('swoole_llama_ready') ? 'yes' : 'no') . "\n";
echo "  - swoole_llama_shutdown: " . (function_exists('swoole_llama_shutdown') ? 'yes' : 'no') . "\n";
echo "  - Llama\\Request class: " . (class_exists('Llama\\Request') ? 'yes' : 'no') . "\n";
echo "\n";

\Co\run(function () {
    global $argv;
    
    // Check for --model argument
    $modelIdx = array_search('--model', $argv);
    if ($modelIdx === false || !isset($argv[$modelIdx + 1])) {
        echo "Usage: php test_init.php --model /path/to/model.gguf [other llama.cpp args]\n";
        echo "\nExample:\n";
        echo "  php test_init.php --model /home/jason-dev/swoole/qwen.gguf --ctx-size 4096 --parallel 2\n";
        return;
    }
    
    echo "Initializing llama.cpp with args: " . implode(' ', array_slice($argv, 1)) . "\n";
    
    $start = microtime(true);
    
    if (!swoole_llama_init($argv)) {
        echo "ERROR: swoole_llama_init() failed\n";
        return;
    }
    
    $initTime = microtime(true) - $start;
    echo "swoole_llama_init() returned in " . round($initTime * 1000, 2) . " ms (thread spawned)\n";
    echo "\n";
    
    // Poll for model ready (this yields to other coroutines)
    echo "Waiting for model to load...\n";
    echo "---\n";
    $pollStart = microtime(true);
    while (true) {
        $status = swoole_llama_ready();
        if ($status === 1) {
            break; // Ready
        }
        if ($status === -1) {
            echo "ERROR: Model loading failed\n";
            return;
        }
        // Yield and check again
        \Co::sleep(0.1);
    }
    $loadTime = microtime(true) - $pollStart;
    echo "---\n";
    echo "SUCCESS: Model loaded in " . round($loadTime, 2) . " seconds\n";
    echo "\n";
    
    // Test non-streaming chat completion using Llama\Request class
    echo "=== Non-streaming chat completion ===\n";
    $params = [
        'method' => 'POST',
        'path' => '/v1/chat/completions',
        'body' => json_encode([
            'model' => 'llama',
            'messages' => [
                ['role' => 'user', 'content' => 'What is 2+2? Answer with just the number.']
            ],
            'max_tokens' => 32,
            'temperature' => 0.0,
        ]),
        'headers' => ['Content-Type' => ['application/json']],
    ];
    
    $start = microtime(true);
    try {
        $req = new \Llama\Request($params);
        
        if ($req->isStream()) {
            echo "ERROR: Expected non-streaming response\n";
        } else {
            $response = $req->getData();
            $elapsed = microtime(true) - $start;
            
            echo "Raw response length: " . strlen($response) . "\n";
            echo "Raw response (first 1000 chars): " . substr($response, 0, 1000) . "\n";
            $data = json_decode($response, true);
            if (isset($data['choices'][0]['message']['content'])) {
                echo "Parsed content: " . trim($data['choices'][0]['message']['content']) . "\n";
            } else if ($data) {
                echo "JSON keys: " . implode(', ', array_keys($data)) . "\n";
            }
            echo "Time: " . round($elapsed * 1000, 1) . " ms\n";
        }
    } catch (Exception $e) {
        echo "ERROR: " . $e->getMessage() . "\n";
    }
    echo "\n";
    
    // Test streaming chat completion using Llama\Request class
    echo "=== Streaming chat completion ===\n";
    $params['body'] = json_encode([
        'model' => 'llama',
        'messages' => [
            ['role' => 'user', 'content' => 'Count from 1 to 5, one number per line.']
        ],
        'max_tokens' => 64,
        'temperature' => 0.0,
        'stream' => true,
    ]);
    
    $start = microtime(true);
    try {
        $req = new \Llama\Request($params);
        
        if (!$req->isStream()) {
            echo "ERROR: Expected streaming response\n";
        } else {
            echo "Response: ";
            $chunkCount = 0;
            $content = '';
            
            // First, get initial data if any
            $firstData = $req->getData();
            if ($firstData !== null) {
                $chunkCount++;
                $data = json_decode($firstData, true);
                if (isset($data['choices'][0]['delta']['content'])) {
                    $content .= $data['choices'][0]['delta']['content'];
                }
                if (isset($data['choices'][0]['delta']['reasoning_content'])) {
                    $content .= $data['choices'][0]['delta']['reasoning_content'];
                }
            }
            
            // Then iterate through subsequent chunks
            while (($chunk = $req->next()) !== null) {
                $chunkCount++;
                if (isset($chunk['choices'][0]['delta']['content'])) {
                    $content .= $chunk['choices'][0]['delta']['content'];
                }
                if (isset($chunk['choices'][0]['delta']['reasoning_content'])) {
                    $content .= $chunk['choices'][0]['delta']['reasoning_content'];
                }
            }
            $elapsed = microtime(true) - $start;
            echo $content;
            echo "\n";
            echo "Chunks: $chunkCount, Time: " . round($elapsed * 1000, 1) . " ms\n";
        }
    } catch (Exception $e) {
        echo "ERROR: " . $e->getMessage() . "\n";
    }
    echo "\n";
    
    echo "Shutting down...\n";
    
    if (!swoole_llama_shutdown()) {
        echo "ERROR: swoole_llama_shutdown() failed\n";
        return;
    }
    
    echo "SUCCESS: Clean shutdown complete\n";
});
