// One HTDemucs model for `splitSegmentsLocal`. The first message builds the
// model; each later message is one segment, planar left then right.
let separator;

self.onmessage = async ({ data }) => {
    try {
        if (data.init) {
            const { moduleUrl, wasmUrl, manifest, weights, digest } = data.init;
            const module = await import(moduleUrl);
            await module.default({ module_or_path: wasmUrl });
            separator = new module.Separator(manifest, new Uint8Array(weights), digest);
            self.postMessage({ ready: true });
            return;
        }
        const half = data.samples.length / 2;
        const samples = separator.separate(data.samples.subarray(0, half), data.samples.subarray(half));
        self.postMessage({ index: data.index, samples }, [samples.buffer]);
    } catch (error) {
        self.postMessage({ error: error?.message || String(error) });
    }
};
