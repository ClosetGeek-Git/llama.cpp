/**
 * Node.js OpenAI Client Test for PHP Swoole Server
 * 
 * Uses the official OpenAI npm package to test SSE streaming.
 * This validates the server implementation against a known-good client.
 * 
 * Install: npm install openai
 * Run: node test_node_client.js
 */

const OpenAI = require('openai');

const client = new OpenAI({
    baseURL: 'http://127.0.0.1:9501/v1',
    apiKey: 'not-needed',  // Our server doesn't require auth
});

async function testHealth() {
    console.log('--- Test 1: Health Check ---');
    try {
        const response = await fetch('http://127.0.0.1:9501/health');
        const data = await response.json();
        console.log('Health:', JSON.stringify(data));
        return data.status === 'ok';
    } catch (e) {
        console.error('Health check failed:', e.message);
        return false;
    }
}

async function testNonStreaming() {
    console.log('\n--- Test 2: Non-Streaming Chat Completion ---');
    const startTime = Date.now();
    
    try {
        const completion = await client.chat.completions.create({
            model: 'test',
            messages: [{ role: 'user', content: 'Say hello in one word' }],
            max_tokens: 20,
            stream: false,
        });
        
        const elapsed = (Date.now() - startTime) / 1000;
        console.log(`Time: ${elapsed.toFixed(3)}s`);
        console.log('Response:', JSON.stringify(completion.choices[0].message, null, 2));
        return true;
    } catch (e) {
        console.error('Non-streaming failed:', e.message);
        return false;
    }
}

async function testStreaming() {
    console.log('\n--- Test 3: Streaming Chat Completion ---');
    const startTime = Date.now();
    let chunkCount = 0;
    let content = '';
    let reasoning = '';
    let firstChunkTime = null;
    
    try {
        const stream = await client.chat.completions.create({
            model: 'test',
            messages: [{ role: 'user', content: 'Count from 1 to 5' }],
            max_tokens: 50,
            stream: true,
        });
        
        for await (const chunk of stream) {
            if (firstChunkTime === null) {
                firstChunkTime = Date.now();
                console.log(`Time to first chunk: ${((firstChunkTime - startTime) / 1000).toFixed(3)}s`);
            }
            chunkCount++;
            const delta = chunk.choices[0]?.delta;
            content += delta?.content || '';
            reasoning += delta?.reasoning_content || '';
            
            // Print progress every 10 chunks
            if (chunkCount % 10 === 0) {
                process.stdout.write('.');
            }
        }
        
        const elapsed = (Date.now() - startTime) / 1000;
        console.log(`\nTotal time: ${elapsed.toFixed(3)}s`);
        console.log(`Chunks received: ${chunkCount}`);
        console.log(`Content length: ${content.length} chars`);
        if (reasoning.length > 0) {
            console.log(`Reasoning length: ${reasoning.length} chars`);
            console.log(`Reasoning preview: ${reasoning.substring(0, 100)}...`);
        }
        if (content.length > 0) {
            console.log(`Content preview: ${content.substring(0, 100)}...`);
        }
        return true;
    } catch (e) {
        console.error('Streaming failed:', e.message);
        return false;
    }
}

async function testConcurrent() {
    console.log('\n--- Test 4: Concurrent Streaming Requests ---');
    const prompts = [
        'What is 2+2?',
        'Name three colors.',
        'What is the capital of France?',
        'Count to 3.',
    ];
    
    const startTime = Date.now();
    
    const promises = prompts.map(async (prompt, i) => {
        const reqStart = Date.now();
        let chunks = 0;
        
        const stream = await client.chat.completions.create({
            model: 'test',
            messages: [{ role: 'user', content: prompt }],
            max_tokens: 50,
            stream: true,
        });
        
        for await (const chunk of stream) {
            chunks++;
        }
        
        const elapsed = (Date.now() - reqStart) / 1000;
        return { id: i, chunks, time: elapsed };
    });
    
    const results = await Promise.all(promises);
    const totalTime = (Date.now() - startTime) / 1000;
    
    for (const r of results) {
        console.log(`  Request ${r.id}: ${r.time.toFixed(3)}s, ${r.chunks} chunks`);
    }
    console.log(`Total concurrent time: ${totalTime.toFixed(3)}s`);
    
    return true;
}

async function testDisconnect() {
    console.log('\n--- Test 5: Client Disconnect (abort after 2 chunks) ---');
    const startTime = Date.now();
    let chunkCount = 0;
    
    try {
        const controller = new AbortController();
        
        const stream = await client.chat.completions.create({
            model: 'test',
            messages: [{ role: 'user', content: 'Write a long essay about computing history' }],
            max_tokens: 500,
            stream: true,
        }, {
            signal: controller.signal,
        });
        
        for await (const chunk of stream) {
            chunkCount++;
            if (chunkCount >= 2) {
                console.log(`Aborting after ${chunkCount} chunks...`);
                controller.abort();
                break;
            }
        }
        
        const elapsed = (Date.now() - startTime) / 1000;
        console.log(`Aborted after: ${elapsed.toFixed(3)}s`);
        console.log('(Check server console for cancel message)');
        return true;
    } catch (e) {
        if (e.name === 'AbortError') {
            const elapsed = (Date.now() - startTime) / 1000;
            console.log(`Aborted after: ${elapsed.toFixed(3)}s, ${chunkCount} chunks`);
            return true;
        }
        console.error('Disconnect test failed:', e.message);
        return false;
    }
}

async function main() {
    console.log('=== Node.js OpenAI Client Test ===\n');
    
    if (!await testHealth()) {
        console.error('\nServer not available. Start it with:');
        console.error('  php test_http_server.php -m /path/to/model.gguf');
        process.exit(1);
    }
    
    await testNonStreaming();
    await testStreaming();
    await testConcurrent();
    await testDisconnect();
    
    console.log('\n=== Tests Complete ===');
}

main().catch(console.error);
