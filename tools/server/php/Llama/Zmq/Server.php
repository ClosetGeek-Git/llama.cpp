<?php
declare(strict_types=1);

namespace Llama\Zmq;

use Swoole\Coroutine\ZMQ\Socket;
use Swoole\Process;
use Swoole\ZMQ\Context;

/**
 * Spawn + supervise a llama-server child over Swoole\Process.
 *
 * Replaces the swoole_llama extension's swoole_llama_load_model() /
 * swoole_llama_model_ready() / swoole_llama_unload_model() trio: instead of
 * loading models into the calling PHP process, each model becomes its own
 * llama-server subprocess listening on its own ZMQ endpoint. The PHP side
 * talks to it via \Llama\Zmq\Request (which finds the endpoint by looking
 * up the model name in {@see Endpoint}).
 *
 * Typical usage in test_multi_model.php style:
 *
 *     $srv = Llama\Zmq\Server::spawn([
 *         'binary'   => '/.../build/bin/llama-server',
 *         'endpoint' => 'ipc:///tmp/llama-qwen.sock',
 *         'args'     => ['-m', '/path/to/qwen.gguf', '--ctx-size', '4096'],
 *         'model'    => 'qwen-base',  // alias to register; defaults to the
 *                                     // model file's basename minus .gguf
 *     ]);
 *     $srv->waitReady();              // blocks (in coroutine) until /health 200
 *     // ... use Llama\Zmq\Request ...
 *     $srv->shutdown();               // SIGTERM, wait, SIGKILL if needed
 *
 * Implementation notes:
 *   - We call Swoole\Process::__construct(callable, redirect_stdio=true).
 *     Inside the child callable we exec the llama-server binary. Doing it
 *     this way (rather than passing argv to a separate constructor) means
 *     we don't depend on a particular Swoole version's `Process::exec` API.
 *   - Readiness is detected by polling /health via a short-lived ZMQ DEALER
 *     socket. /health is in server-zmq's `public_endpoints` set, so it
 *     answers immediately on bind; the readiness gate in server-zmq's worker
 *     loop is what we're actually waiting for (it flips when the inference
 *     subsystem reports a loaded model).
 *   - Shutdown sends SIGTERM, then waits up to $shutdownTimeoutSec seconds
 *     for graceful exit (the self-pipe signal handler in server.cpp drains
 *     in-flight requests), then SIGKILL.
 */
final class Server
{
    private Process $proc;
    private string $endpoint;
    private string $modelAlias;
    private bool   $ready    = false;
    private bool   $shutDown = false;
    private float  $readyTimeoutSec   = 600.0; // model load can be slow on CPU
    private float  $shutdownTimeoutSec = 15.0;

    private function __construct(Process $proc, string $endpoint, string $modelAlias)
    {
        $this->proc       = $proc;
        $this->endpoint   = $endpoint;
        $this->modelAlias = $modelAlias;
    }

    /**
     * @param array{
     *   binary?: string,
     *   endpoint: string,
     *   args: list<string>,
     *   model?: string,
     *   readyTimeoutSec?: float,
     *   shutdownTimeoutSec?: float,
     *   extraEnv?: array<string,string>
     * } $params
     */
    public static function spawn(array $params): self
    {
        $binary   = (string) ($params['binary']   ?? '/home/jason-dev/swoole/llama.cpp/build/bin/llama-server');
        $endpoint = (string) ($params['endpoint'] ?? '');
        $args     = (array)  ($params['args']     ?? []);
        if ($endpoint === '') {
            throw new \InvalidArgumentException("Server::spawn requires 'endpoint'");
        }
        $modelAlias = $params['model'] ?? self::deriveModelAlias($args);

        // Build the full argv for llama-server: caller-supplied args plus the
        // transport flags. We force --no-enable-http so the child doesn't
        // also bind a TCP port and surprise anyone scanning the host; flip
        // it off explicitly here rather than relying on absence of --port.
        $childArgs = array_merge([
            '--no-enable-http',
            '--enable-zmq',
            '--zmq-bind', $endpoint,
        ], $args);

        // Wipe a stale ipc socket file (from a prior crash) so bind() succeeds.
        if (strncmp($endpoint, 'ipc://', 6) === 0) {
            $path = substr($endpoint, 6);
            if ($path !== '' && file_exists($path)) {
                @unlink($path);
            }
        }

        $proc = new Process(
            function (Process $worker) use ($binary, $childArgs, $params): void {
                // child context: replace ourselves with llama-server.
                // Extra env, if requested, lets the caller flip things like
                // LLAMA_LOG_VERBOSITY without re-spawning.
                if (!empty($params['extraEnv'])) {
                    foreach ((array) $params['extraEnv'] as $k => $v) {
                        putenv("$k=$v");
                    }
                }
                // exec(file, args[]) — second arg is the argv list *after* argv[0].
                $worker->exec($binary, $childArgs);
            },
            false, // redirect_stdin_and_stdout — keep inheriting parent's stdio
            0,     // pipe_type — 0 disables the parent<->child pipe (we don't need it)
        );

        $pid = $proc->start();
        if ($pid === false) {
            throw new \RuntimeException("Server::spawn: Process::start failed for $binary");
        }

        $server = new self($proc, $endpoint, $modelAlias);
        if (isset($params['readyTimeoutSec'])) {
            $server->readyTimeoutSec = (float) $params['readyTimeoutSec'];
        }
        if (isset($params['shutdownTimeoutSec'])) {
            $server->shutdownTimeoutSec = (float) $params['shutdownTimeoutSec'];
        }

        // Register the alias->endpoint mapping right away so Request can resolve
        // even before waitReady() is called (some callers want to fire health
        // probes through Request themselves).
        Endpoint::register($modelAlias, $endpoint);

        return $server;
    }

    public function endpoint(): string  { return $this->endpoint; }
    public function modelAlias(): string { return $this->modelAlias; }
    public function pid(): int          { return (int) $this->proc->pid; }
    public function isReady(): bool     { return $this->ready; }

    /**
     * Block (cooperatively if inside a coroutine, with usleep if not) until
     * the child answers the readiness probe with status 200, or until the
     * timeout elapses. Throws on failure.
     */
    public function waitReady(): void
    {
        if ($this->ready) {
            return;
        }
        $deadline = microtime(true) + $this->readyTimeoutSec;
        $lastErr = '';
        while (microtime(true) < $deadline) {
            // child died? bail out with whatever exit info we have
            if (!self::pidAlive($this->pid())) {
                throw new \RuntimeException(
                    "Server::waitReady: child pid={$this->pid()} ({$this->modelAlias}) "
                    . "exited before becoming ready"
                );
            }

            $status = self::probeHealth($this->endpoint, $lastErr);
            if ($status === 200) {
                $this->ready = true;
                return;
            }
            // 503 ("Loading model") or refused connection = keep waiting
            self::sleep(0.25);
        }
        throw new \RuntimeException(
            "Server::waitReady: timed out after {$this->readyTimeoutSec}s "
            . "for {$this->modelAlias}@{$this->endpoint} (lastErr={$lastErr})"
        );
    }

    /**
     * Graceful stop. Sends SIGTERM, polls for exit up to shutdownTimeoutSec,
     * then SIGKILLs. Always unregisters the endpoint.
     *
     * Safe to call from inside or outside a coroutine — falls back to
     * usleep() when no coroutine scheduler is active (e.g. shutdown in the
     * outermost script scope, after Co\run has returned).
     */
    public function shutdown(): void
    {
        if ($this->shutDown) {
            return;
        }
        $this->shutDown = true;
        Endpoint::unregister($this->modelAlias);

        $pid = $this->pid();
        if ($pid > 0 && self::pidAlive($pid)) {
            Process::kill($pid, SIGTERM);
            $deadline = microtime(true) + $this->shutdownTimeoutSec;
            while (microtime(true) < $deadline && self::pidAlive($pid)) {
                self::sleep(0.1);
            }
            if (self::pidAlive($pid)) {
                Process::kill($pid, SIGKILL);
                self::sleep(0.2);
            }
        }
        // Reap so we don't leave a zombie. Non-blocking — we already polled.
        @Process::wait(false);

        // Clean up the ipc socket file if we own it.
        if (strncmp($this->endpoint, 'ipc://', 6) === 0) {
            $path = substr($this->endpoint, 6);
            if ($path !== '' && file_exists($path)) {
                @unlink($path);
            }
        }
    }

    /** Coroutine-aware sleep. Uses Co::sleep when in a Swoole coroutine,
     *  usleep when not. Both forms accept fractional seconds. */
    private static function sleep(float $seconds): void
    {
        if (\Swoole\Coroutine::getCid() > 0) {
            \Co::sleep($seconds);
        } else {
            usleep((int) ($seconds * 1_000_000));
        }
    }

    // ---- helpers ----------------------------------------------------------

    private static function pidAlive(int $pid): bool
    {
        if ($pid <= 0) return false;
        // posix_kill(pid, 0) is the standard "exists?" probe.
        return function_exists('posix_kill') ? posix_kill($pid, 0) : @file_exists("/proc/$pid");
    }

    /**
     * One-shot DEALER readiness probe.
     *
     * We can't use /health here: server-zmq.cpp puts /health (and /v1/models,
     * /api/tags, etc.) in its public_endpoints set, which bypasses the
     * is_ready gate — so /health answers 200 from the moment the socket is
     * bound, *before* the model finishes loading. We need an endpoint that
     * is subject to the gate, so we get 503 "Loading model" while loading
     * and a real 200 only once the inference subsystem is live.
     *
     * GET /props fits: it's the lightest non-public endpoint (no body, no
     * model lookup, just returns server metadata once the gate is open).
     *
     * Returns the envelope's status code, or 0 if we couldn't get a reply.
     */
    private static function probeHealth(string $endpoint, string &$lastErr): int
    {
        try {
            $ctx  = new Context();
            $sock = new Socket($ctx, \ZMQ::SOCKET_DEALER);
            $sock->setOption(\ZMQ::SOCKOPT_LINGER, 0);
            $sock->setOption(\ZMQ::SOCKOPT_RCVTIMEO, 300);
            $sock->setOption(\ZMQ::SOCKOPT_SNDTIMEO, 300);
            $sock->connect($endpoint);

            $rid = 'ready-' . bin2hex(random_bytes(4));
            $sock->send(json_encode([
                'method' => 'GET', 'path' => '/props', 'body' => '', 'id' => $rid,
            ], JSON_UNESCAPED_SLASHES));

            $frame = $sock->recv();
            $sock->close();

            if ($frame === false || $frame === '') {
                $lastErr = 'no reply';
                return 0;
            }
            $env = json_decode($frame, true);
            if (!is_array($env)) {
                $lastErr = 'malformed envelope';
                return 0;
            }
            return (int) ($env['status'] ?? 0);
        } catch (\Throwable $e) {
            $lastErr = $e->getMessage();
            return 0;
        }
    }

    /** Pull a model alias out of an argv list (the token after -m / --model). */
    private static function deriveModelAlias(array $args): string
    {
        $n = count($args);
        for ($i = 0; $i < $n - 1; $i++) {
            $a = (string) $args[$i];
            if ($a === '-m' || $a === '--model') {
                $path = (string) $args[$i + 1];
                $base = basename($path);
                if (str_ends_with($base, '.gguf')) {
                    $base = substr($base, 0, -strlen('.gguf'));
                }
                return $base;
            }
        }
        throw new \RuntimeException(
            "Server::spawn: could not derive 'model' alias from args; pass 'model' explicitly"
        );
    }
}
