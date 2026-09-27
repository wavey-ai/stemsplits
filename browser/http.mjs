// Retry only replayable requests. The deadline includes the complete response body.
export function retryAfterMs(value, now = Date.now()) {
    if (!value) return 0;
    const seconds = Number(value);
    return Math.max(0, Number.isFinite(seconds) ? seconds * 1000 : (Date.parse(value) - now) || 0);
}

export function waitForRetry(ms, signal) {
    signal?.throwIfAborted();
    return new Promise((resolve, reject) => {
        const finish = error => {
            clearTimeout(timer); signal?.removeEventListener('abort', cancel);
            error ? reject(error) : resolve();
        };
        const cancel = () => finish(signal.reason);
        const timer = setTimeout(() => finish(), ms);
        signal?.addEventListener('abort', cancel, { once: true });
    });
}

export async function requestWithRetry(url, init, {
    signal = init.signal, fetcher = fetch, timeoutMs = 31000, attempts = 3,
    retryDelayMs = 500, random = Math.random, sleep = waitForRetry,
    attemptHeader,
    consume = response => response.arrayBuffer(),
} = {}) {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || !Number.isInteger(attempts) || attempts < 1) {
        throw new RangeError('Invalid request deadline or attempt count.');
    }
    for (let attempt = 0; attempt < attempts; attempt++) {
        signal?.throwIfAborted();
        const controller = new AbortController();
        const cancel = () => controller.abort(signal.reason);
        signal?.addEventListener('abort', cancel, { once: true });
        const timer = setTimeout(() => controller.abort(new DOMException('The request timed out. Please try again.', 'TimeoutError')), timeoutMs);
        let response, httpError, removeAbort;
        let retry = false, delay = 0;
        try {
            const cancelled = new Promise((_, reject) => {
                const stop = () => reject(controller.signal.reason);
                controller.signal.addEventListener('abort', stop, { once: true });
                removeAbort = () => controller.signal.removeEventListener('abort', stop);
            });
            const operation = (async () => {
                const headers = attemptHeader
                    ? { ...Object.fromEntries(new Headers(init.headers)), [attemptHeader]: String(attempt) }
                    : init.headers;
                response = await fetcher(url, { ...init, headers, signal: controller.signal });
                controller.signal.throwIfAborted();
                if (!response.ok) {
                    httpError = new Error('Could not process this audio. Please try again.');
                    httpError.status = response.status;
                    httpError.retryable = [408, 425, 429, 500, 502, 503, 504].includes(response.status);
                    let detail;
                    try { detail = await response.json(); } catch {}
                    controller.signal.throwIfAborted();
                    if (typeof detail?.error === 'string') httpError.message = detail.error;
                    throw httpError;
                }
                const value = await consume(response, controller.signal);
                controller.signal.throwIfAborted();
                return value;
            })();
            return await Promise.race([operation, cancelled]);
        } catch (error) {
            signal?.throwIfAborted();
            // A permanent HTTP error remains permanent if its error body stalls.
            const failure = httpError || (controller.signal.aborted ? controller.signal.reason : error);
            retry = failure.retryable === true || failure instanceof TypeError || failure.name === 'TimeoutError';
            if (!retry || attempt + 1 === attempts) throw failure;
            const serverDelay = retryAfterMs(response?.headers.get('Retry-After'));
            // Do not retry before a long server deadline.
            if (serverDelay > 120000) throw failure;
            delay = Math.max(serverDelay, retryDelayMs * (2 ** attempt)) + random() * retryDelayMs;
        } finally {
            clearTimeout(timer); removeAbort?.(); signal?.removeEventListener('abort', cancel);
            controller.abort();
            if (response?.body && !response.body.locked) void response.body.cancel().catch(() => {});
        }
        if (retry) await sleep(delay, signal);
    }
}
