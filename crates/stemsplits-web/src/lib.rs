//! Browser preparation and reconstruction for the pinned HTDemucs contract.
use rubato::{FftFixedInOut, Resampler};
use stemsplits_demucs::{triangular_weight, ChunkPlan, OverlapAdd, StemKind};
use wasm_bindgen::prelude::*;

const RATE: usize = 44_100;
/// The longest track that the browser separates.
const BROWSER_SECONDS: usize = 600;

fn prepare(
    samples: &[i16],
    channels: usize,
    rate: usize,
    max_seconds: usize,
) -> Result<[Vec<f32>; 2], String> {
    if !(1..=2).contains(&channels)
        || !(8_000..=192_000).contains(&rate)
        || samples.is_empty()
        || !samples.len().is_multiple_of(channels)
    {
        return Err("Choose a mono or stereo audio track.".into());
    }
    let frames = samples.len() / channels;
    if frames > rate * max_seconds {
        return Err(format!(
            "Choose a track up to {} minutes long.",
            max_seconds / 60
        ));
    }
    let input: [Vec<f32>; 2] = std::array::from_fn(|channel| {
        samples
            .chunks_exact(channels)
            .map(|frame| frame[channel.min(channels - 1)] as f32 / 32768.0)
            .collect()
    });
    if rate == RATE {
        return Ok(input);
    }
    let (mut a, mut b) = (rate, RATE);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    // An even output block gives an integer filter delay.
    let unit = 2 * RATE / a;
    let chunk = 1024_usize.div_ceil(unit) * unit;
    let mut resampler =
        FftFixedInOut::<f32>::new(rate, RATE, chunk, 2).map_err(|e| e.to_string())?;
    let delay = resampler.output_delay();
    // WASM uses 32-bit usize. Multiply the frame count in u64.
    let total = ((frames as u64 * RATE as u64 + rate as u64 / 2) / rate as u64).max(1) as usize;
    let mut output = [Vec::new(), Vec::new()];
    let mut position = 0;
    while output[0].len() < total + delay {
        let count = resampler.input_frames_next();
        let available = frames.saturating_sub(position).min(count);
        let slices = [
            &input[0][position..position + available],
            &input[1][position..position + available],
        ];
        let block = resampler
            .process_partial(if available == 0 { None } else { Some(&slices) }, None)
            .map_err(|e| e.to_string())?;
        for channel in 0..2 {
            output[channel].extend_from_slice(&block[channel]);
        }
        position += available;
    }
    Ok(output.map(|channel| channel[delay..delay + total].to_vec()))
}

#[wasm_bindgen]
pub fn stem_titles() -> Vec<String> {
    StemKind::ALL
        .iter()
        .map(|stem| stem.title().into())
        .collect()
}

#[wasm_bindgen]
pub fn sample_rate() -> u32 {
    RATE as u32
}

/// Final WAV quantization, after overlap-add. This matches the reference CLI.
#[wasm_bindgen]
pub fn pcm16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|value| (value * 32767.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect()
}

#[wasm_bindgen]
pub struct SplitSession {
    input: [Vec<f32>; 2],
    offsets: Vec<usize>,
    window: Vec<f32>,
    accumulators: [OverlapAdd; 8],
    next: usize,
    flushed: usize,
}

impl SplitSession {
    /// A session for a server worker, which accepts tracks up to `max_seconds`.
    pub fn with_limit(
        samples: &[i16],
        channels: usize,
        rate: usize,
        max_seconds: usize,
    ) -> Result<SplitSession, String> {
        let input = prepare(samples, channels, rate, max_seconds)?;
        Ok(Self {
            offsets: ChunkPlan::CONTRACT.offsets(input[0].len()),
            window: triangular_weight(ChunkPlan::CONTRACT.segment_frames()),
            input,
            accumulators: std::array::from_fn(|_| OverlapAdd::new()),
            next: 0,
            flushed: 0,
        })
    }
}

#[wasm_bindgen]
impl SplitSession {
    #[wasm_bindgen(constructor)]
    pub fn new(samples: &[i16], channels: usize, rate: usize) -> Result<SplitSession, String> {
        Self::with_limit(samples, channels, rate, BROWSER_SECONDS)
    }

    pub fn frames(&self) -> usize {
        self.input[0].len()
    }
    pub fn count(&self) -> usize {
        self.offsets.len()
    }
    pub fn response_samples(&self) -> usize {
        self.window.len() * 8
    }

    fn input_segment(&self, index: usize) -> Result<[Vec<f32>; 2], String> {
        let offset = *self.offsets.get(index).ok_or("Invalid segment index")?;
        let mut output = [vec![0.0; self.window.len()], vec![0.0; self.window.len()]];
        let count = self.window.len().min(self.frames() - offset);
        for (channel, out) in output.iter_mut().enumerate() {
            out[..count].copy_from_slice(&self.input[channel][offset..offset + count]);
        }
        Ok(output)
    }

    /// Input is one framed 192 kbps stereo Opus stream, including a zero-padded short tail.
    pub fn request(&self, index: usize) -> Result<Vec<u8>, String> {
        stemsplits_transport::encode(&[self.input_segment(index)?])
    }

    /// The same segment as `request`, as planar f32 samples (left, then
    /// right) for a model that runs on this device.
    pub fn segment(&self, index: usize) -> Result<Vec<f32>, String> {
        Ok(self.input_segment(index)?.concat())
    }

    /// Consume framed Opus responses in plan order. Return final samples, with eight planar channels.
    pub fn accept(&mut self, index: usize, response: &[u8]) -> Result<Vec<f32>, String> {
        if index != self.next || index >= self.count() {
            return Err("Unexpected segment order".into());
        }
        let pairs = stemsplits_transport::decode(response, 4, self.window.len())?;
        let samples: Vec<f32> = pairs.into_iter().flatten().flatten().collect();
        self.accept_samples(index, &samples)
    }

    /// Consume one segment's stems as planar f32 samples, in plan order: drums,
    /// bass, other and vocals, each left then right. Returns what `accept`
    /// returns.
    pub fn accept_samples(&mut self, index: usize, samples: &[f32]) -> Result<Vec<f32>, String> {
        if index != self.next || index >= self.count() {
            return Err("Unexpected segment order".into());
        }
        if samples.len() != self.response_samples() || samples.iter().any(|s| !s.is_finite()) {
            return Err("Invalid stem response".into());
        }
        let offset = self.offsets[index];
        let until = self
            .offsets
            .get(index + 1)
            .copied()
            .unwrap_or(self.frames());
        let mut output = Vec::with_capacity((until - self.flushed) * 8);
        for (accumulator, channel) in self
            .accumulators
            .iter_mut()
            .zip(samples.chunks_exact(self.window.len()))
        {
            accumulator.add(offset, channel, &self.window);
            output.extend(accumulator.flush_until(until));
            accumulator.discard_flushed();
        }
        self.flushed = until;
        self.next += 1;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn final_pcm_rounding_and_clipping() {
        assert_eq!(
            pcm16(&[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]),
            [-32768, -32767, -16384, 0, 16384, 32767, 32767]
        );
    }
    #[test]
    fn mono_short_track_and_validation() {
        let mut session = SplitSession::new(&[8192; 101], 1, RATE).unwrap();
        let request = session.request(0).unwrap();
        let request = stemsplits_transport::decode(&request, 1, session.window.len()).unwrap();
        assert!(request[0][0][20..80]
            .iter()
            .all(|sample| (*sample - 0.25).abs() < 0.02));
        assert!(session.accept(0, &[0; 8]).is_err());
        let response = stemsplits_transport::encode(
            &(0..4)
                .map(|_| {
                    [
                        vec![0.25; session.window.len()],
                        vec![0.25; session.window.len()],
                    ]
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let output = session.accept(0, &response).unwrap();
        assert_eq!(output.len(), 101 * 8);
        assert!(output.iter().all(|sample| (*sample - 0.25).abs() < 0.03));
        assert!(session.accept(0, &[]).is_err());
        assert!(SplitSession::new(&[], 2, RATE).is_err());
    }
    #[test]
    fn streaming_seam_is_bit_exact_with_batch() {
        let total = 900_003;
        let mut session = SplitSession::new(&vec![0; total * 2], 2, RATE).unwrap();
        let mut batch: [OverlapAdd; 8] = std::array::from_fn(|_| OverlapAdd::new());
        let mut actual: [Vec<f32>; 8] = std::array::from_fn(|_| Vec::new());
        for index in 0..session.count() {
            let samples: Vec<f32> = (0..session.response_samples())
                .map(|i| ((i * 17 + index * 3) % 1009) as f32 / 1009.0 - 0.5)
                .collect();
            let pairs: Vec<[Vec<f32>; 2]> = samples
                .chunks_exact(session.window.len() * 2)
                .map(|pair| {
                    let (left, right) = pair.split_at(session.window.len());
                    [left.to_vec(), right.to_vec()]
                })
                .collect();
            let response = stemsplits_transport::encode(&pairs).unwrap();
            let decoded: Vec<f32> =
                stemsplits_transport::decode(&response, 4, session.window.len())
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .flatten()
                    .collect();
            for (accumulator, channel) in batch
                .iter_mut()
                .zip(decoded.chunks_exact(session.window.len()))
            {
                accumulator.add(session.offsets[index], channel, &session.window);
            }
            let output = session.accept(index, &response).unwrap();
            let frames = output.len() / 8;
            for (out, channel) in actual.iter_mut().zip(output.chunks_exact(frames)) {
                out.extend_from_slice(channel);
            }
            assert!(session
                .accumulators
                .iter()
                .all(|a| a.retained_frames() <= session.window.len()));
        }
        for (mut expected, actual) in batch.into_iter().zip(actual) {
            assert_eq!(expected.finish(), actual);
        }
    }
    #[test]
    fn local_segments_use_the_same_plan_and_seam() {
        let total = 700_001;
        let input: Vec<i16> = (0..total * 2)
            .map(|i| ((i * 37) % 2001) as i16 - 1000)
            .collect();
        let mut session = SplitSession::new(&input, 2, RATE).unwrap();
        let mut batch: [OverlapAdd; 8] = std::array::from_fn(|_| OverlapAdd::new());
        let mut actual: [Vec<f32>; 8] = std::array::from_fn(|_| Vec::new());
        for index in 0..session.count() {
            let segment = session.segment(index).unwrap();
            assert_eq!(segment.len(), 2 * session.window.len());
            let offset = session.offsets[index];
            let count = session.window.len().min(total - offset);
            assert_eq!(segment[..count], session.input[0][offset..offset + count]);
            assert!(segment[count..session.window.len()]
                .iter()
                .all(|s| *s == 0.0));
            // A stand-in model: each stem is the input scaled by its index.
            let samples: Vec<f32> = (0..4)
                .flat_map(|stem| segment.iter().map(move |s| s * (stem + 1) as f32))
                .collect();
            for (accumulator, channel) in batch
                .iter_mut()
                .zip(samples.chunks_exact(session.window.len()))
            {
                accumulator.add(offset, channel, &session.window);
            }
            let output = session.accept_samples(index, &samples).unwrap();
            let frames = output.len() / 8;
            for (out, channel) in actual.iter_mut().zip(output.chunks_exact(frames)) {
                out.extend_from_slice(channel);
            }
        }
        for (mut expected, actual) in batch.into_iter().zip(actual) {
            assert_eq!(expected.finish(), actual);
        }
        assert!(session.accept_samples(0, &[0.0; 8]).is_err());
    }

    #[test]
    fn a_worker_session_accepts_a_track_longer_than_the_browser_limit() {
        let rate = 8_000;
        let samples = vec![0_i16; rate * (BROWSER_SECONDS + 1)];
        assert_eq!(
            prepare(&samples, 1, rate, BROWSER_SECONDS).unwrap_err(),
            "Choose a track up to 10 minutes long."
        );
        assert!(prepare(&samples, 1, rate, 1800).is_ok());
    }

    #[test]
    fn resampling_preserves_duration_channels_and_timing() {
        for rate in [22_050, 48_000, 96_000] {
            let samples: Vec<i16> = (0..rate)
                .flat_map(|frame| {
                    let tone = (0.4
                        * (std::f32::consts::TAU * 440.0 * frame as f32 / rate as f32).sin()
                        * 32768.0) as i16;
                    [tone, -tone]
                })
                .collect();
            let output = prepare(&samples, 2, rate, BROWSER_SECONDS).unwrap();
            assert_eq!(output[0].len(), RATE);
            for (frame, actual) in output[0].iter().enumerate().take(RATE - 1000).skip(1000) {
                let expected =
                    0.4 * (std::f32::consts::TAU * 440.0 * frame as f32 / RATE as f32).sin();
                assert!(
                    (*actual - expected).abs() < 0.001,
                    "rate={rate} frame={frame} actual={} expected={expected}",
                    output[0][frame]
                );
                assert!((output[0][frame] + output[1][frame]).abs() < 0.000001);
            }
        }
    }
}
