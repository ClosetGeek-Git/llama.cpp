<?php
/**
 * Minimal PSR-4-style autoloader for the Llama\Zmq wrapper.
 *
 * Map: Llama\Zmq\<Name>  ->  tools/server/php/Llama/Zmq/<Name>.php
 *
 * The wrapper is pure PHP and depends only on three extensions that must be
 * loaded in php.ini (or via `php -d extension=...`):
 *   - swoole          (Co\run, Swoole\Process, Swoole\Coroutine\Channel)
 *   - swoole_zmq      (Swoole\Coroutine\ZMQ\Socket, Swoole\ZMQ\Context)
 *   - zmq             (provides the ZMQ:: constants used by swoole_zmq)
 */

spl_autoload_register(static function (string $class): void {
    $prefix = 'Llama\\Zmq\\';
    if (strncmp($class, $prefix, strlen($prefix)) !== 0) {
        return;
    }
    $relative = substr($class, strlen($prefix));
    $path = __DIR__ . '/Llama/Zmq/' . str_replace('\\', '/', $relative) . '.php';
    if (is_file($path)) {
        require_once $path;
    }
});
