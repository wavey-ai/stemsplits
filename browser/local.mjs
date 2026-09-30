// Stems on this device: the pinned plan and seam from `SplitSession`, with
// HTDemucs (the `stemsplits-wasm` package) in a pool of workers.
//
// Segments go to the workers in plan order and come back in any order;
// `session.accept_samples` takes them in plan order, as `splitSegments`
// takes the cloud's responses.

/// Workers for this device. Each worker holds about 700 MB of WASM memory,
/// and more than four workers gave no more speed in a test on an 8-core Mac.
export function poolSize(nav = globalThis.navigator) {
    const touchMac = nav?.maxTouchPoints > 1 && /Macintosh/.test(nav?.userAgent || '');
    if (/Android|iPhone|iPad|Mobi/i.test(nav?.userAgent || '') || touchMac) return 1;
    const cores = nav?.hardwareConcurrency || 2;
    const byMemory = nav?.deviceMemory ? Math.max(1, Math.floor(nav.deviceMemory / 2)) : 4;
    return Math.max(1, Math.min(4, Math.floor(cores / 2), byMemory));
}

const hex = buffer => Array.from(new Uint8Array(buffer), byte => byte.toString(16).padStart(2, '0')).join('');

/// The bundle manifest and weight blob under `base`, with the blob's
/// digest checked against the manifest. Caching the blob is the host's work:
/// in BITNEEDLE the service worker keeps everything under `/wasm/`.
export async function loadModel(base, { signal, onProgress = () => {} } = {}) {
    const response = await fetch(`${base}/bundle.json`, { cache: 'no-cache', signal });
    if (!response.ok) throw new Error('Could not load the stem model.');
    const manifest = await response.text();
    const { file, sha256, byte_length: total } = JSON.parse(manifest).weights;
    const download = await fetch(`${base}/${file}`, { signal });
    if (!download.ok) throw new Error('Could not download the stem model.');
    const reader = download.body.getReader();
    const bytes = new Uint8Array(total);
    let received = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (received + value.length > total) throw new Error('The stem model did not download correctly. Try again.');
        bytes.set(value, received);
        received += value.length;
        onProgress({ bytes: received, total });
    }
    const digest = hex(await crypto.subtle.digest('SHA-256', bytes.subarray(0, received)));
    if (received !== total || digest !== sha256) throw new Error('The stem model did not download correctly. Try again.');
    return { manifest, weights: bytes.buffer, digest };
}

/// Separates every segment of `session` with `model` on `workers` workers.
/// `onSamples` receives what `session.accept_samples` returns, in plan order.
export async function splitSegmentsLocal(session, {
    model, workerUrl, moduleUrl, wasmUrl, workers = poolSize(), signal, onProgress = () => {}, onSamples,
}) {
    signal?.throwIfAborted();
    const count = session.count();
    const stage = Array(count).fill('queued');
    const report = () => {
        let fraction = 0, received = 0, assembled = 0, processing = 0;
        for (const item of stage) {
            if (item === 'assembled') { fraction += 1; received++; assembled++; }
            else if (item === 'received') { fraction += 0.95; received++; }
            else if (item === 'processing') processing++;
        }
        onProgress({ fraction: fraction / count, received, assembled, count, processing, workers: pool.length, local: true });
    };
    const pool = [];
    const results = new Map();
    let failure = null, wake = () => {};
    const fail = error => { failure ??= error; wake(); };
    const abort = () => fail(signal.reason ?? new DOMException('Stem splitting cancelled', 'AbortError'));
    signal?.addEventListener('abort', abort, { once: true });
    let next = 0;
    const dispatch = worker => {
        if (failure || next >= count) return;
        const index = next++;
        const samples = session.segment(index);
        stage[index] = 'processing'; report();
        worker.postMessage({ index, samples }, [samples.buffer]);
    };
    try {
        const size = Math.max(1, Math.min(count, workers));
        await Promise.all(Array.from({ length: size }, () => new Promise((resolve, reject) => {
            const worker = new Worker(workerUrl, { type: 'module' });
            pool.push(worker);
            worker.onerror = event => { const error = new Error(event.message || 'The stem model stopped.'); reject(error); fail(error); };
            worker.onmessage = ({ data }) => {
                if (data.error) { const error = new Error(data.error); reject(error); fail(error); }
                else if (data.ready) resolve();
                else {
                    results.set(data.index, data.samples);
                    stage[data.index] = 'received'; report();
                    dispatch(worker); wake();
                }
            };
            worker.postMessage({ init: { moduleUrl, wasmUrl, manifest: model.manifest, weights: model.weights, digest: model.digest } });
        })));
        report();
        for (const worker of pool) dispatch(worker);
        for (let index = 0; index < count; index++) {
            while (!results.has(index)) {
                if (failure) throw failure;
                await new Promise(resolve => { wake = resolve; });
            }
            if (failure) throw failure;
            const samples = results.get(index);
            results.delete(index);
            await onSamples(session.accept_samples(index, samples), index);
            stage[index] = 'assembled'; report();
        }
    } finally {
        signal?.removeEventListener('abort', abort);
        for (const worker of pool) worker.terminate();
    }
}
