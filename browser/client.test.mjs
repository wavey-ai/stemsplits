import test from 'node:test';
import assert from 'node:assert/strict';
import { splitSegments } from './client.mjs';

function session(count = 5) {
    let next = 0;
    return {
        count: () => count,
        request: index => new Uint8Array([index]),
        accept(index, samples) { assert.equal(index, next++); assert.equal(samples.length, 8); assert.equal(samples[0], index); return samples; },
    };
}
const response = samples => new Response(samples, { headers: { 'X-Stems-Format': 'soundkit-v2-opus-192' } });

test('bounded parallel requests are consumed in plan order', async () => {
    let active = 0, peak = 0;
    const received = [], progress = [];
    await splitSegments(session(), {
        fetcher: async (_, init) => {
            assert.equal(init.headers['X-Api-Key'], undefined);
            peak = Math.max(peak, ++active);
            const index = init.body[0];
            await new Promise(resolve => setTimeout(resolve, index % 2 ? 1 : 10));
            active--;
            return response(new Uint8Array(8).fill(index));
        },
        onSamples: samples => received.push(samples[0]), onProgress: value => progress.push(value),
    });
    assert.equal(peak, 5);
    assert.deepEqual(received, [0, 1, 2, 3, 4]);
    assert.equal(progress.at(-1).fraction, 1);
    assert.equal(progress.at(-1).assembled, 5);
});

test('truncated, oversized and unsupported responses fail before reconstruction', async () => {
    for (const reply of [response(new Uint8Array()), response(new Uint8Array(2 * 1024 * 1024 + 1)), new Response(new Uint8Array(8))]) {
        let accepted = false;
        await assert.rejects(splitSegments(session(1), { attempts: 1, fetcher: async () => reply, onSamples: () => { accepted = true; } }));
        assert.equal(accepted, false);
    }
});

test('authentication errors are shown and remaining requests are cancelled', async () => {
    let aborted = false;
    await assert.rejects(splitSegments(session(), {
        fetcher: async (_, init) => {
            if (init.body[0] === 0) return Response.json({ error: 'Sign in to split this track.' }, { status: 401 });
            return new Promise((_, reject) => init.signal.addEventListener('abort', () => {
                aborted = true; reject(new DOMException('Cancelled', 'AbortError'));
            }, { once: true }));
        }, onSamples: () => assert.fail('No samples expected'),
    }), /Sign in/);
    assert.equal(aborted, true);
});

test('cancellation starts no new requests', async () => {
    const controller = new AbortController();
    let started = 0;
    await assert.rejects(splitSegments(session(), {
        signal: controller.signal,
        fetcher: async () => { started++; return response(new Uint8Array(8)); },
        onSamples: () => controller.abort(),
    }), { name: 'AbortError' });
    assert.equal(started, 5);
});

test('every segment is offered before any response returns, without the old four-lane cap', async () => {
    let started = 0;
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    const splitting = splitSegments(session(104), {
        fetcher: async (_, { body }) => {
            if (++started === 104) release();
            await gate;
            return response(new Uint8Array(8).fill(body[0]));
        }, onSamples: () => {},
    });
    await splitting;
    assert.equal(started, 104);
});

test('an available lane starts more work while the first response is delayed', async () => {
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    const started = [];
    await splitSegments(session(5), {
        concurrency: 2,
        fetcher: async (_, { body }) => {
            const index = body[0]; started.push(index);
            if (index === 0) await gate;
            if (index === 4) release();
            return response(new Uint8Array(8).fill(index));
        }, onSamples: () => {},
    });
    assert.deepEqual(started, [0, 1, 2, 3, 4]);
});

test('only the incomplete segment is retried; each segment is reconstructed once', async () => {
    const calls = [0, 0, 0], accepted = [];
    await splitSegments(session(3), {
        retryDelayMs: 0,
        fetcher: async (_, { body }) => {
            const index = body[0];
            return ++calls[index] === 1 && index === 1 ? response(new Uint8Array()) : response(new Uint8Array(8).fill(index));
        }, onSamples: samples => accepted.push(samples[0]),
    });
    assert.deepEqual(calls, [1, 2, 1]);
    assert.deepEqual(accepted, [0, 1, 2]);
});

test('a stalled response body times out and retries before reconstruction', async () => {
    let calls = 0, cancelled = 0, accepted = 0;
    await splitSegments(session(1), {
        timeoutMs: 10, retryDelayMs: 0,
        fetcher: async () => ++calls === 1
            ? response(new ReadableStream({ cancel() { cancelled++; } }))
            : response(new Uint8Array(8)),
        onSamples: () => accepted++,
    });
    assert.equal(calls, 2); assert.equal(cancelled, 1); assert.equal(accepted, 1);
});

test('a permanent error in a later segment cancels a stalled earlier segment immediately', async () => {
    await assert.rejects(splitSegments(session(2), {
        timeoutMs: 1000,
        fetcher: async (_, { body }) => body[0] === 0
            ? new Promise(() => {})
            : Response.json({ error: 'Sign in.' }, { status: 401 }),
        onSamples: () => assert.fail('No samples expected'),
    }), /Sign in/);
});

test('disk-backed replies are consumed in order and released', async () => {
    const files = new Map(), accepted = [];
    const store = {
        async open(index) {
            const parts = [];
            return { write: bytes => parts.push(bytes.slice()), close: () => files.set(index, parts), abort: () => files.delete(index) };
        },
        async take(index) {
            const parts = files.get(index); files.delete(index);
            return new Uint8Array(await new Blob(parts).arrayBuffer());
        },
    };
    await splitSegments(session(6), {
        store, fetcher: async (_, { body }) => response(new Uint8Array(8).fill(body[0])),
        onSamples: samples => accepted.push(samples[0]),
    });
    assert.deepEqual(accepted, [0, 1, 2, 3, 4, 5]); assert.equal(files.size, 0);
});
