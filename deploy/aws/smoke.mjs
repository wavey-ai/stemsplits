// One synthetic segment through the deployed API. The key stays on this machine.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import init, { SplitSession } from '../../crates/stemsplits-web/pkg/stemsplits_web.js';

const endpoint = process.env.STEMS_ENDPOINT || 'https://d23dlwkayd.execute-api.eu-north-1.amazonaws.com/prod/separate';
const key = (await readFile(new URL('.api-key', import.meta.url), 'utf8')).trim();
await init({ module_or_path: await WebAssembly.compile(await readFile(
    new URL('../../crates/stemsplits-web/pkg/stemsplits_web_bg.wasm', import.meta.url))) });
const frames = 343980;
const input = new Int16Array(frames * 2);
for (let frame = 0; frame < frames; frame++) {
    input[frame * 2] = 6000 * Math.sin(2 * Math.PI * 440 * frame / 44100);
    input[frame * 2 + 1] = 4500 * Math.sin(2 * Math.PI * 330 * frame / 44100);
}
const session = new SplitSession(input, 2, 44100);
const request = session.request(0);
const started = performance.now();
const response = await fetch(endpoint, { method: 'POST', signal: AbortSignal.timeout(150000),
    headers: { 'Content-Type': 'application/octet-stream', 'Accept': 'application/octet-stream', 'X-Stems-Format': 'soundkit-v2-opus-192', 'X-Api-Key': key }, body: request });
assert.equal(response.status, 200, `Segment request: ${response.status}`);
assert.equal(response.headers.get('x-stems-format'), 'soundkit-v2-opus-192');
const bytes = new Uint8Array(await response.arrayBuffer());
const samples = session.accept(0, bytes), stems = [];
assert.equal(samples.length, frames * 8);
for (let stem = 0; stem < 4; stem++) {
    let peak = 0, squares = 0;
    for (const value of samples.subarray(stem * frames * 2, (stem + 1) * frames * 2)) {
        assert.ok(Number.isFinite(value)); peak = Math.max(peak, Math.abs(value)); squares += value * value;
    }
    stems.push({ peak, rms: Math.sqrt(squares / (frames * 2)) });
}
assert.ok(stems.some(stem => stem.peak > 0.001));
console.log(JSON.stringify({ seconds: (performance.now() - started) / 1000, bytes: bytes.byteLength,
    uploadBytes: request.byteLength, sha256: createHash('sha256').update(bytes).digest('hex'), stems }, null, 2));
session.free();
