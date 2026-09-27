import { requestWithRetry } from './http.mjs';

// Rust owns the plan and reconstruction. Each planned segment has one request lane.
export async function splitSegments(session, {
    endpoint = '/api/stems/separate', signal, onProgress = () => {}, onSamples,
    fetcher = fetch, concurrency = session.count(), store, ...retryOptions
}) {
    const abort = new AbortController();
    const cancel = () => abort.abort(signal.reason);
    signal?.throwIfAborted();
    signal?.addEventListener('abort', cancel, { once: true });
    const count = session.count();
    const activity = Array.from({ length: count }, () => ({ stage: 'queued', bytes: 0, total: 0 }));
    const report = () => {
        let fraction = 0, received = 0, assembled = 0, bytes = 0, totalBytes = 0;
        for (const item of activity) {
            bytes += item.bytes; totalBytes += item.total;
            if (item.stage === 'assembled') { fraction += 1; received++; assembled++; }
            else if (item.stage === 'received') { fraction += 0.92; received++; }
            else if (item.stage === 'downloading') fraction += 0.62 + 0.3 * (item.total ? Math.min(1, item.bytes / item.total) : Math.min(0.9, item.bytes / (768 * 1024)));
            else if (item.stage === 'processing') fraction += 0.08;
        }
        onProgress({ fraction: fraction / count, received, assembled, count, bytes, totalBytes,
            processing: activity.filter(item => item.stage === 'processing').length,
            downloading: activity.filter(item => item.stage === 'downloading').length });
    };
    report();
    const slots = Math.max(1, Math.min(count, Math.floor(concurrency) || count));
    const ready = Array.from({ length: count }, () => {
        let resolve, reject;
        const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
        promise.catch(() => {});
        return { promise, resolve, reject };
    });
    let next = 0;
    const run = async () => {
        try {
            while (next < count) {
                abort.signal.throwIfAborted();
                const index = next++;
                activity[index] = { stage: 'processing', bytes: 0, total: 0 }; report();
                const result = await requestWithRetry(endpoint, {
                    method: 'POST', credentials: 'same-origin', cache: 'no-store',
                    headers: { 'Content-Type': 'application/octet-stream', 'X-Stems-Format': 'soundkit-v2-opus-192' },
                    body: session.request(index),
                }, {
                    timeoutMs: 150000, ...retryOptions, fetcher, signal: abort.signal,
                    attemptHeader: 'X-Stems-Attempt',
                    consume: async (response, attemptSignal) => {
                        if (response.headers.get('X-Stems-Format') !== 'soundkit-v2-opus-192') throw new Error('Unsupported stem response.');
                        const total = Number(response.headers.get('Content-Length')) || 0;
                        activity[index] = { stage: 'downloading', bytes: 0, total }; report();
                        return readSegment(response, 2 * 1024 * 1024, attemptSignal, store, index, (bytes) => {
                            activity[index].bytes = bytes; report();
                        });
                    },
                });
                abort.signal.throwIfAborted();
                ready[index].resolve(result);
                activity[index].stage = 'received'; report();
            }
        } catch (error) {
            abort.abort(error);
            for (const entry of ready) entry?.reject(error);
            throw error;
        }
    };
    const lanes = Array.from({ length: slots }, run);
    for (const lane of lanes) lane.catch(() => {});
    try {
        for (let index = 0; index < count; index++) {
            const result = await ready[index].promise;
            abort.signal.throwIfAborted();
            const samples = store ? await store.take(index) : result;
            abort.signal.throwIfAborted();
            await onSamples(session.accept(index, samples), index);
            abort.signal.throwIfAborted();
            activity[index].stage = 'assembled'; report();
            ready[index] = null;
        }
        await Promise.all(lanes);
    } finally {
        abort.abort(); signal?.removeEventListener('abort', cancel);
        await Promise.allSettled(lanes);
    }
}

async function readSegment(response, maximum, signal, store, index, onBytes = () => {}) {
    const reader = response.body.getReader();
    const cancel = () => { void reader.cancel(signal.reason).catch(() => {}); };
    signal.addEventListener('abort', cancel, { once: true });
    let writer;
    try {
        signal.throwIfAborted();
        writer = await store?.open(index);
        signal.throwIfAborted();
        const parts = store ? null : [];
        let offset = 0;
        for (;;) {
            const { done, value } = await reader.read();
            signal.throwIfAborted();
            if (done) break;
            if (offset + value.length > maximum) throw new Error('Stem response is too large.');
            if (writer) await writer.write(value); else parts.push(value);
            offset += value.length;
            onBytes(offset);
        }
        if (!offset) throw Object.assign(new Error('Stem response is incomplete.'), { retryable: true });
        await writer?.close();
        writer = null;
        if (!parts) return null;
        const bytes = new Uint8Array(offset);
        let position = 0;
        for (const part of parts) { bytes.set(part, position); position += part.length; }
        return bytes;
    } finally {
        signal.removeEventListener('abort', cancel);
        void reader.cancel().catch(() => {}); reader.releaseLock();
        if (writer) await writer.abort().catch(() => {});
    }
}

function checkScratchName(name) {
    if (!/^stems-[0-9a-f-]{36}$/.test(name)) throw new Error('Invalid stem scratch directory.');
}

export async function createSegmentStore(name) {
    checkScratchName(name);
    const root = await navigator.storage.getDirectory();
    let release = () => {};
    if (navigator.locks) {
        let acquired, failed;
        const entered = new Promise((resolve, reject) => { acquired = resolve; failed = reject; });
        navigator.locks.request(name, async () => {
            const held = new Promise(resolve => { release = resolve; });
            acquired(); await held;
        }).catch(failed);
        await entered;
    }
    let directory;
    try {
        // A terminated worker releases its lock. Remove only abandoned stem scratch files.
        if (navigator.locks) for await (const [other, handle] of root.entries()) {
            if (other === name || handle.kind !== 'directory' || !/^stems-[0-9a-f-]{36}$/.test(other)) continue;
            await navigator.locks.request(other, { ifAvailable: true }, async lock => {
                if (lock) await root.removeEntry(other, { recursive: true }).catch(error => {
                    if (error.name !== 'NotFoundError') throw error;
                });
            });
        }
        directory = await root.getDirectoryHandle(name, { create: true });
    } catch (error) { release(); throw error; }
    return {
        close: () => release(),
        async open(index) {
            const file = await directory.getFileHandle(String(index), { create: true });
            return file.createWritable();
        },
        async take(index) {
            const file = await (await directory.getFileHandle(String(index))).getFile();
            const samples = new Uint8Array(await file.arrayBuffer());
            await directory.removeEntry(String(index));
            return samples;
        },
    };
}

export async function removeSegmentStore(name) {
    checkScratchName(name);
    const root = await navigator.storage.getDirectory();
    try { await root.removeEntry(name, { recursive: true }); }
    catch (error) { if (error.name !== 'NotFoundError') throw error; }
}
