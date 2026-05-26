<?php
declare(strict_types=1);

namespace Llama\Zmq;

use Swoole\Coroutine\ZMQ\Socket;
use Swoole\ZMQ\Context;

/**
 * Drop-in replacement for the swoole_llama extension's \Llama\Request class,
 * over ZMQ instead of in-process dispatch.
 *
 * Surface mirrors \Llama\Request (server-coro/coro-extension.cpp):
 *   - __construct(array $params)   — fires the request, blocks until at least
 *                                    the response envelope is in hand
 *   - isStream(): bool             — true if the body asked for stream:true
 *                                    AND the server confirmed via its envelope
 *   - getStatusCode(): int         — HTTP-equivalent status from envelope
 *   - getData(): ?array            — non-stream: full decoded JSON
 *                                    stream:     decoded *first* chunk
 *   - next(): ?array               — stream only: subsequent decoded chunks
 *                                    returns null when the stream terminates
 *   - cancel(): void               — best-effort CANCEL envelope; the in-flight
 *                                    request's should_stop flips and the
 *                                    handler unwinds
 *
 * Constructor params:
 *   method   string  required  "GET" | "POST"
 *   path     string  required  e.g. "/v1/chat/completions"
 *   body     string  optional  request body as a JSON-encoded string (the
 *                              extension version takes a string too; we pull
 *                              the "model" field out to resolve the endpoint)
 *   headers  array   optional  map of header name => string|string[]; the
 *                              wire format expects scalar string values, so
 *                              array values are flattened by taking the first
 *                              entry (matching how curl/cpp-httplib presents
 *                              multi-value headers to handlers)
 *   endpoint string  optional  explicit ZMQ endpoint; if absent, the wrapper
 *                              parses body['model'] and looks it up via
 *                              \Llama\Zmq\Endpoint::lookup()
 *   id       string  optional  correlation id; defaults to a random 16-hex-char
 *                              string. Echoed back in every reply envelope.
 *
 * Wire protocol (matches tools/server/server-zmq.cpp):
 *
 *   Request envelope (single DEALER frame, JSON):
 *     { "method": ..., "path": ..., "body": "<JSON string>",
 *       "headers": {k: v, ...}, "id": "<rid>" }
 *
 *   Non-stream reply (single ROUTER frame after our client_id is stripped):
 *     { "status": 200, "content_type": ..., "headers": {...},
 *       "rid": "<rid>", "stream": false, "data": "<raw response body>" }
 *
 *   Stream reply (multipart message):
 *     frame 1: { "status": 200, ..., "rid": "<rid>", "stream": true }
 *     frame 2..N-1: raw chunk bytes (typically SSE "data: {...}\n\n" lines)
 *     frame N:  empty terminator
 *
 *   Cancel (out-of-band, one-way, sent on the same DEALER socket):
 *     { "method": "CANCEL", "id": "<rid>" }
 *
 * Lifetime: the wrapper opens one DEALER socket per Request. The socket is
 * closed by __destruct(). For high-throughput Odin use the socket cost is
 * negligible compared to inference latency; if it ever matters, an opt-in
 * pooled-socket mode is a small extension.
 */
final class Request
{
    private const RID_BYTES = 8;

    private string $rid;
    private string $endpoint;
    private bool   $expectStream;

    private ?Context $ctx  = null;
    private ?Socket  $sock = null;

    /** First-frame envelope (always JSON-decoded once received). */
    private array $envelope = [];

    /** Cached parse of the body the caller passed (looked up by lookup logic). */
    private array $bodyDecoded = [];

    /**
     * Buffer holding the first stream chunk so getData() can return it.
     * Null after getData() has consumed it, or for non-stream responses.
     */
    private ?array $firstChunk = null;

    private bool $streamEnded = false;
    private bool $cancelled   = false;

    public function __construct(array $params)
    {
        $method  = (string) ($params['method'] ?? 'GET');
        $path    = (string) ($params['path']   ?? '/');
        $body    = (string) ($params['body']   ?? '');
        $headers = self::flattenHeaders((array) ($params['headers'] ?? []));
        $this->rid = (string) ($params['id'] ?? bin2hex(random_bytes(self::RID_BYTES)));

        // body field is itself a JSON string per the upstream \Llama\Request
        // convention. We parse it once: to find "stream" (so isStream() can
        // answer before the wire reply, matching the extension) and to find
        // "model" (so we can look up the endpoint when no explicit one is
        // given). Empty or non-JSON bodies are fine — get/v1/health etc.
        if ($body !== '') {
            $decoded = json_decode($body, true);
            if (is_array($decoded)) {
                $this->bodyDecoded = $decoded;
            }
        }
        $this->expectStream = !empty($this->bodyDecoded['stream']);

        $this->endpoint = (string) ($params['endpoint'] ?? '');
        if ($this->endpoint === '') {
            $model = (string) ($this->bodyDecoded['model'] ?? '');
            if ($model === '') {
                throw new \RuntimeException(
                    "Llama\\Zmq\\Request: neither 'endpoint' nor body['model'] given; "
                    . "cannot resolve transport for path $path"
                );
            }
            $resolved = Endpoint::lookup($model);
            if ($resolved === null) {
                throw new \RuntimeException(
                    "Llama\\Zmq\\Request: model '$model' is not registered. "
                    . "Call Llama\\Zmq\\Endpoint::register(\$model, \$endpoint) after "
                    . "spawning the llama-server child."
                );
            }
            $this->endpoint = $resolved;
        }

        $this->openSocket();
        $this->sendRequest($method, $path, $body, $headers);
        $this->receiveEnvelope();

        // For streams, pre-fetch the first chunk so getData() can return it
        // (matches the extension version which already had it buffered after
        // construction). next() will only ever yield subsequent chunks.
        if ($this->isStream() && !$this->streamEnded) {
            $this->firstChunk = $this->readNextChunk();
        }
    }

    public function __destruct()
    {
        $this->closeSocket();
    }

    public function isStream(): bool
    {
        return (bool) ($this->envelope['stream'] ?? false);
    }

    public function getStatusCode(): int
    {
        return (int) ($this->envelope['status'] ?? 200);
    }

    public function getData(): ?array
    {
        if ($this->isStream()) {
            $chunk = $this->firstChunk;
            $this->firstChunk = null;
            return $chunk;
        }
        $raw = $this->envelope['data'] ?? null;
        if (!is_string($raw) || $raw === '') {
            return null;
        }
        $decoded = json_decode($raw, true);
        return is_array($decoded) ? $decoded : null;
    }

    public function next(): ?array
    {
        if (!$this->isStream() || $this->streamEnded) {
            return null;
        }
        return $this->readNextChunk();
    }

    public function cancel(): void
    {
        if ($this->cancelled) {
            return;
        }
        $this->cancelled = true;
        // Best-effort: same socket, one CANCEL envelope. The server's worker
        // looks up the rid in its in-flight registry and flips the atomic.
        // We don't wait for a reply (CANCEL is one-way per the protocol).
        if ($this->sock !== null) {
            $payload = json_encode([
                'method' => 'CANCEL',
                'id'     => $this->rid,
            ], JSON_UNESCAPED_SLASHES);
            @$this->sock->send($payload);
        }
    }

    // ---- internals --------------------------------------------------------

    /** @param array<string, string|string[]> $headers */
    private static function flattenHeaders(array $headers): array
    {
        $out = [];
        foreach ($headers as $name => $value) {
            if (is_array($value)) {
                // cpp-httplib presents the first value to handlers; mirror.
                $value = $value[0] ?? '';
            }
            $out[(string) $name] = (string) $value;
        }
        return $out;
    }

    private function openSocket(): void
    {
        $this->ctx  = new Context();
        $this->sock = new Socket($this->ctx, \ZMQ::SOCKET_DEALER);
        // LINGER=0 so the destructor doesn't park waiting on buffered sends
        // that the server might never drain.
        $this->sock->setOption(\ZMQ::SOCKOPT_LINGER, 0);
        $this->sock->connect($this->endpoint);
    }

    private function closeSocket(): void
    {
        if ($this->sock !== null) {
            try { $this->sock->close(); } catch (\Throwable) { /* ignore */ }
            $this->sock = null;
        }
        $this->ctx = null;
    }

    private function sendRequest(string $method, string $path, string $body, array $headers): void
    {
        $payload = json_encode([
            'method'  => $method,
            'path'    => $path,
            'body'    => $body,
            'headers' => $headers,
            'id'      => $this->rid,
        ], JSON_UNESCAPED_SLASHES);
        if ($payload === false) {
            throw new \RuntimeException(
                "Llama\\Zmq\\Request: json_encode failed: " . json_last_error_msg()
            );
        }
        $ok = $this->sock->send($payload);
        if ($ok === false) {
            throw new \RuntimeException("Llama\\Zmq\\Request: zmq_send failed");
        }
    }

    private function receiveEnvelope(): void
    {
        $frame = $this->sock->recv();
        if ($frame === false || $frame === '') {
            throw new \RuntimeException("Llama\\Zmq\\Request: empty/failed first-frame recv");
        }
        $env = json_decode($frame, true);
        if (!is_array($env)) {
            throw new \RuntimeException(
                "Llama\\Zmq\\Request: malformed envelope (not JSON object): "
                . substr($frame, 0, 200)
            );
        }
        $this->envelope = $env;

        // Stream replies are a single multipart message of the shape
        // [header, chunk*, empty_terminator]. We've consumed `header`; the
        // remaining frames live in the same logical message and are pulled
        // via recv() while RCVMORE is set. If the server replied with a
        // *non*-stream envelope despite us asking for stream (e.g. an error
        // before streaming started), we trust the envelope and forget it.
    }

    /** @return array|null  decoded chunk, or null when stream ends */
    private function readNextChunk(): ?array
    {
        if ($this->streamEnded) {
            return null;
        }

        while (true) {
            // If the previous frame was the last one of the multipart message,
            // there is no more to receive — the stream is over.
            // RCVMORE reflects "is the most recently received frame followed
            // by another?". So we check RCVMORE after the last recv, not before.
            // For the first call here, the last recv was the header (which
            // always had RCVMORE since the empty terminator follows). For
            // subsequent calls, we re-check after each chunk.
            $more = (int) $this->sock->getOption(\ZMQ::SOCKOPT_RCVMORE);
            if ($more === 0) {
                $this->streamEnded = true;
                return null;
            }

            $frame = $this->sock->recv();
            if ($frame === false) {
                // socket error — treat as end of stream
                $this->streamEnded = true;
                return null;
            }
            if ($frame === '') {
                // empty-frame terminator
                $this->streamEnded = true;
                return null;
            }

            // SSE-style chunks from llama.cpp's stream handlers come as:
            //   "data: {json}\n\n"   ... and occasionally "data: [DONE]\n\n"
            // We accept either raw JSON or the SSE-prefixed form so the
            // wrapper survives whichever stream type the route emits.
            $parsed = self::parseChunk($frame);
            if ($parsed === null) {
                // Unparseable chunk — skip it and keep reading. This is
                // strictly forgiving; turning it into an error makes the
                // wrapper brittle to non-data SSE frames like keep-alives.
                continue;
            }
            return $parsed;
        }
    }

    /**
     * Parse a single stream chunk frame. Accepts either:
     *   - raw JSON (the extension version's TASK_RESPONSE_TYPE_RAW path)
     *   - SSE-formatted "data: <json>\n\n" (the cpp-httplib + server-zmq path)
     *
     * Returns null for non-data SSE chunks (keep-alives, [DONE], comments).
     */
    private static function parseChunk(string $raw): ?array
    {
        $raw = trim($raw, "\r\n");
        if ($raw === '') {
            return null;
        }

        if (strncmp($raw, 'data:', 5) === 0) {
            $payload = ltrim(substr($raw, 5));
            if ($payload === '[DONE]') {
                return null;
            }
            $decoded = json_decode($payload, true);
            return is_array($decoded) ? $decoded : null;
        }

        if ($raw[0] === '{' || $raw[0] === '[') {
            $decoded = json_decode($raw, true);
            return is_array($decoded) ? $decoded : null;
        }

        return null;
    }
}
