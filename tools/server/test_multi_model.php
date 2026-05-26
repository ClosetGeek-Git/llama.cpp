<?php
/**
 * Multi-model API test, ported from tools/server-coro/test_multi_model.php.
 *
 * Original used the swoole_llama extension's in-process model registry:
 *   - swoole_llama_load_model([...]) loaded a GGUF inside this PHP process.
 *   - swoole_llama_model_ready('<name>') polled the loader state.
 *   - swoole_llama_list_models() / swoole_llama_unload_model() managed it.
 *   - new \Llama\Request([...]) dispatched directly into in-process handlers.
 *
 * This port uses the new tools/server transport stack:
 *   - llama-server is spawned as a separate process (one per model) via
 *     Swoole\Process, listening on its own ZMQ ipc:// endpoint.
 *   - \Llama\Zmq\Server wraps spawn + waitReady + shutdown.
 *   - \Llama\Zmq\Endpoint maps model name -> ZMQ endpoint.
 *   - \Llama\Zmq\Request is a drop-in replacement for \Llama\Request
 *     that talks DEALER<->ROUTER over ZMQ.
 *
 * Model paths use what's on this machine (the originals are not present).
 *   LLM:       /home/jason-dev/rapier_babylon/Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf
 *   Embedding: /home/jason-dev/rapier_babylon/embeddinggemma-300m-qat-Q4_0.gguf
 *
 * The model *alias* used in request bodies is the filename minus `.gguf`,
 * which is what llama-server itself reports via /v1/models.
 */

require_once __DIR__ . '/php/autoload.php';

use Llama\Zmq\Endpoint;
use Llama\Zmq\Request;
use Llama\Zmq\Server;

// Make sure the deps we need are loaded. Match the original test's check.
foreach (['swoole', 'swoole_zmq', 'zmq'] as $ext) {
    if (!extension_loaded($ext)) {
        fwrite(STDERR, "Error: required extension '$ext' not loaded\n");
        exit(1);
    }
}

const LLAMA_SERVER_BIN = '/home/jason-dev/swoole/llama.cpp/build/bin/llama-server';

const LLM_MODEL_PATH = '/home/jason-dev/rapier_babylon/Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf';
const LLM_MODEL_NAME = 'Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M';
const LLM_ENDPOINT   = 'ipc:///tmp/llama-zmq-llm.sock';

const EMB_MODEL_PATH = '/home/jason-dev/rapier_babylon/embeddinggemma-300m-qat-Q4_0.gguf';
const EMB_MODEL_NAME = 'embeddinggemma-300m-qat-Q4_0';
const EMB_ENDPOINT   = 'ipc:///tmp/llama-zmq-emb.sock';

echo "=== Multi-Model API Test (ZMQ transport) ===\n\n";

// Swoole\Process::start() forks; it MUST be called outside any coroutine
// context. (The swoole_llama extension's swoole_llama_load_model did NOT
// fork — it loaded the model inside the calling PHP process, so doing it
// inside \Co\run was fine. With out-of-process llama-server children the
// fork happens up-front.) Spawn all model servers first, then enter \Co\run
// for the wait/request/cancel logic.

echo "3. Spawning LLM model '" . LLM_MODEL_NAME . "'...\n";
$llmServer = Server::spawn([
    'binary'   => LLAMA_SERVER_BIN,
    'endpoint' => LLM_ENDPOINT,
    'model'    => LLM_MODEL_NAME,
    'args'     => [
        '-m', LLM_MODEL_PATH,
        '--ctx-size', '4096',
        '--parallel', '2',
        '--n-gpu-layers', '-1',
        // Disable chain-of-thought so a 32-token completion has room for
        // actual content rather than just the reasoning preamble.
        '--reasoning', 'off',
    ],
]);
echo "   OK: pid={$llmServer->pid()} endpoint={$llmServer->endpoint()}\n";

echo "10. Spawning embedding model '" . EMB_MODEL_NAME . "'...\n";
$embServer = Server::spawn([
    'binary'   => LLAMA_SERVER_BIN,
    'endpoint' => EMB_ENDPOINT,
    'model'    => EMB_MODEL_NAME,
    'args'     => [
        '-m', EMB_MODEL_PATH,
        '--embeddings', '-c', '512',
        '--pooling', 'mean',
        '--n-gpu-layers', '0',
    ],
]);
echo "   OK: pid={$embServer->pid()} endpoint={$embServer->endpoint()}\n";

\Co\run(function () use (&$llmServer, &$embServer) {
    try {

        // 4. Wait for it to be ready
        echo "4. Waiting for '" . LLM_MODEL_NAME . "' to be ready...\n";
        $llmServer->waitReady();
        echo "   OK: Model ready\n";

        // 5. List models
        echo "5. Listing models...\n";
        $models = Endpoint::list();
        echo "   OK: " . json_encode($models) . "\n";

        // 7. Make a non-streaming chat completion request
        echo "7. Making request with model='" . LLM_MODEL_NAME . "'...\n";
        try {
            $req = new Request([
                'method' => 'POST',
                'path'   => '/v1/chat/completions',
                'body'   => json_encode([
                    'model'       => LLM_MODEL_NAME,
                    'messages'    => [['role' => 'user', 'content' => 'Say hello in spanish']],
                    'max_tokens'  => 32,
                    'temperature' => 0.0,
                    'stream'      => false,
                ]),
                'headers' => ['Content-Type' => ['application/json']],
            ]);

            if ($req->isStream()) {
                $chunks   = 0;
                $content  = '';
                $reasoning = '';
                $firstData = $req->getData();
                if ($firstData !== null) {
                    $chunks++;
                    if (isset($firstData['choices'][0]['delta']['reasoning_content'])) {
                        $reasoning .= $firstData['choices'][0]['delta']['reasoning_content'];
                    }
                    if (isset($firstData['choices'][0]['delta']['content'])) {
                        $content .= $firstData['choices'][0]['delta']['content'];
                    }
                }
                while (($chunk = $req->next()) !== null) {
                    $chunks++;
                    if (isset($chunk['choices'][0]['delta']['reasoning_content'])) {
                        $reasoning .= $chunk['choices'][0]['delta']['reasoning_content'];
                    }
                    if (isset($chunk['choices'][0]['delta']['content'])) {
                        $content .= $chunk['choices'][0]['delta']['content'];
                    }
                }
                echo "   Reasoning: " . ($reasoning ?: '(empty)') . "\n";
                echo "   Content:   " . ($content   ?: '(empty)') . "\n";
            } else {
                $data    = $req->getData();
                $choice  = $data['choices'][0] ?? [];
                $message = $choice['message']  ?? [];
                $content   = $message['content']           ?? '';
                $reasoning = $message['reasoning_content'] ?? '';

                echo "   Reasoning: " . ($reasoning ?: '(empty)') . "\n";
                echo "   Content:   " . ($content   ?: '(empty)') . "\n";
            }
        } catch (\Throwable $e) {
            echo "   FAILED: " . $e->getMessage() . "\n";
        }

        // 8. Wrong model name
        echo "8. Making request with wrong model name...\n";
        try {
            $req = new Request([
                'method' => 'POST',
                'path'   => '/v1/chat/completions',
                'body'   => json_encode([
                    'model'      => 'nonexistent-model',
                    'messages'   => [['role' => 'user', 'content' => 'Hello']],
                    'max_tokens' => 16,
                    'stream'     => true,
                ]),
                'headers' => ['Content-Type' => ['application/json']],
            ]);
            $status = $req->getStatusCode();
            $data   = $req->getData();
            $msg    = $data['error']['message'] ?? '(no message)';
            // Wrong model used to be 404 via the in-process registry.
            // Here, there is no in-process registry — the lookup fails at
            // the Endpoint stage before any RPC, so we expect Request to
            // throw. The "OK" branches handle either outcome.
            if ($status === 404 || $status === 400) {
                echo "   OK: Got $status error response: $msg\n";
            } else {
                echo "   FAILED: Expected 404/400, got status $status\n";
            }
        } catch (\Throwable $e) {
            echo "   OK: Correctly threw: " . $e->getMessage() . "\n";
        }

        // 9. Missing model field
        echo "9. Making request without model field...\n";
        try {
            $req = new Request([
                'method' => 'POST',
                'path'   => '/v1/chat/completions',
                'body'   => json_encode([
                    'messages'   => [['role' => 'user', 'content' => 'Hello']],
                    'max_tokens' => 16,
                    'stream'     => true,
                ]),
                'headers' => ['Content-Type' => ['application/json']],
            ]);
            $status = $req->getStatusCode();
            echo "   FAILED: expected an exception (no endpoint resolvable), got status $status\n";
        } catch (\Throwable $e) {
            echo "   OK: Correctly threw: " . $e->getMessage() . "\n";
        }

        // 11. Wait for embed model
        echo "11. Waiting for '" . EMB_MODEL_NAME . "' to be ready...\n";
        $embServer->waitReady();
        echo "   OK: Model ready\n";

        // 12. Single-input embedding
        echo "12. Testing single-input embedding...\n";
        try {
            $req = new Request([
                'method' => 'POST',
                'path'   => '/v1/embeddings',
                'body'   => json_encode([
                    'model' => EMB_MODEL_NAME,
                    'input' => 'Hello, world!',
                ]),
                'headers' => ['Content-Type' => ['application/json']],
            ]);
            $data = $req->getData();
            if (isset($data['data'][0]['embedding']) &&
                is_array($data['data'][0]['embedding']) &&
                count($data['data'][0]['embedding']) > 0) {
                echo "   OK: Got embedding with " . count($data['data'][0]['embedding']) . " dimensions\n";
            } else {
                echo "   FAILED: Invalid embedding response: " . json_encode($data) . "\n";
            }
        } catch (\Throwable $e) {
            echo "   FAILED: " . $e->getMessage() . "\n";
        }

        // 13. Multi-input embedding
        echo "13. Testing multi-input embedding...\n";
        try {
            $req = new Request([
                'method' => 'POST',
                'path'   => '/v1/embeddings',
                'body'   => json_encode([
                    'model' => EMB_MODEL_NAME,
                    'input' => ['Text one', 'Text two', 'Text three'],
                ]),
                'headers' => ['Content-Type' => ['application/json']],
            ]);
            $data = $req->getData();
            if (isset($data['data']) && count($data['data']) === 3) {
                $valid = true;
                foreach ($data['data'] as $item) {
                    if (!isset($item['embedding']) || !is_array($item['embedding']) || count($item['embedding']) === 0) {
                        $valid = false;
                        break;
                    }
                }
                if ($valid) {
                    echo "   OK: Got 3 embeddings, each with " . count($data['data'][0]['embedding']) . " dimensions\n";
                } else {
                    echo "   FAILED: Invalid embedding data in response\n";
                }
            } else {
                echo "   FAILED: Expected 3 embeddings, got: " . json_encode($data) . "\n";
            }
        } catch (\Throwable $e) {
            echo "   FAILED: " . $e->getMessage() . "\n";
        }

        // 14. Unload embedding model
        echo "14. Unloading '" . EMB_MODEL_NAME . "'...\n";
        $embServer->shutdown();
        $embServer = null;
        echo "   OK: Model unloaded\n";

        // 16. Unload LLM
        echo "16. Unloading '" . LLM_MODEL_NAME . "'...\n";
        $llmServer->shutdown();
        $llmServer = null;
        echo "   OK: Model unloaded\n";

        // 17. Verify registry empty
        echo "17. Verifying model list is empty...\n";
        $models = Endpoint::list();
        if (empty($models)) {
            echo "   OK: Model list is empty\n";
        } else {
            echo "   FAILED: Model list not empty: " . json_encode($models) . "\n";
        }

        echo "\n=== All tests completed ===\n";
    } finally {
        // Belt-and-braces: kill anything we still own on the way out so a
        // mid-test fatal doesn't leave a stray llama-server hogging the GPU.
        if ($embServer !== null) $embServer->shutdown();
        if ($llmServer !== null) $llmServer->shutdown();
    }
});
