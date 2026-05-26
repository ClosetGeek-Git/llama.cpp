<?php
declare(strict_types=1);

namespace Llama\Zmq;

/**
 * Static registry mapping a model alias to a llama-server ZMQ endpoint.
 *
 * Each spawned llama-server child publishes one or more ZMQ endpoints
 * (`ipc://...` or `tcp://...`). PHP code calling {@see Request} only knows
 * the model name (passed in the JSON body's `model` field, matching the
 * Odin client and the upstream OpenAI API). This registry is what lets the
 * Request locate the right transport for that model.
 *
 * Population is the responsibility of whoever spawns the children — see
 * {@see Server::spawn()}, which calls {@see Endpoint::register()} on
 * successful ready-probe.
 *
 * Process-local: this is just a static PHP array. If you fork multiple
 * Swoole workers and want them to share the registry, you'll need to push
 * it through Swoole\Table or rebuild it in each worker's onStart. For the
 * common "single Co\run script spawns N children" pattern (which is what
 * the ported test does), the static array is enough.
 */
final class Endpoint
{
    /** @var array<string, string> model name => zmq endpoint */
    private static array $byModel = [];

    public static function register(string $modelName, string $endpoint): void
    {
        self::$byModel[$modelName] = $endpoint;
    }

    public static function unregister(string $modelName): void
    {
        unset(self::$byModel[$modelName]);
    }

    public static function lookup(string $modelName): ?string
    {
        return self::$byModel[$modelName] ?? null;
    }

    /** @return list<string> */
    public static function list(): array
    {
        return array_keys(self::$byModel);
    }

    public static function clear(): void
    {
        self::$byModel = [];
    }
}
