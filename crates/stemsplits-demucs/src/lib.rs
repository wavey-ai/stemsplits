//! The parts of HTDemucs that are not the model: what a stem is, how a track
//! is cut into overlapping segments, and how those segments are sewn back
//! together.
//!
//! This is the Rust home for the seam. Wavey's `StemSeparator.swift`
//! already cuts a track into 7.8-second segments with 25% overlap and
//! triangular overlap-add; the point of pinning it here is that a cloud
//! fan-out and the phone must use the *same* plan, or the two produce
//! different stems for the same record. One definition, two callers.
//!
//! Nothing here runs a model. The segment is the model's own inference unit,
//! so the plan is what makes the segments independent and parallelisable.

use stemsplits_stft::Geometry;

/// The part of a mix a stem is. Four, because HTDemucs separates four.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StemKind {
    Drums,
    Bass,
    Other,
    Vocals,
}

impl StemKind {
    pub const ALL: [StemKind; 4] = [
        StemKind::Drums,
        StemKind::Bass,
        StemKind::Other,
        StemKind::Vocals,
    ];

    pub const fn title(self) -> &'static str {
        match self {
            StemKind::Drums => "DRUMS",
            StemKind::Bass => "BASS",
            StemKind::Other => "OTHER",
            StemKind::Vocals => "VOCALS",
        }
    }

    /// Which of the model's four output groups this part comes back in.
    pub const fn model_index(self) -> usize {
        match self {
            StemKind::Drums => 0,
            StemKind::Bass => 1,
            StemKind::Other => 2,
            StemKind::Vocals => 3,
        }
    }
}

/// How a track is cut into model segments.
///
/// The shipped plan is `StemSeparator.swift`'s: the contract geometry's
/// 343,980-frame segment, 25% overlap, so a segment owns 257,985 new frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkPlan {
    pub geometry: Geometry,
    pub overlap: f32,
}

impl ChunkPlan {
    pub const CONTRACT: ChunkPlan = ChunkPlan {
        geometry: Geometry::CONTRACT,
        overlap: 0.25,
    };

    pub fn segment_frames(&self) -> usize {
        self.geometry.segment
    }

    /// New frames each segment advances past the previous one.
    pub fn stride(&self) -> usize {
        (self.segment_frames() as f64 * (1.0 - self.overlap as f64)).round() as usize
    }

    /// The start frame of every segment. A short track is a single segment;
    /// a long one ends on a final segment pinned to the last full window.
    pub fn offsets(&self, total_frames: usize) -> Vec<usize> {
        let segment = self.segment_frames();
        if total_frames <= segment {
            return vec![0];
        }
        let mut offsets: Vec<usize> = (0..=(total_frames - segment))
            .step_by(self.stride())
            .collect();
        let final_offset = total_frames - segment;
        if offsets.last() != Some(&final_offset) {
            offsets.push(final_offset);
        }
        offsets
    }
}

/// The triangular overlap weight over one segment, normalised to peak 1.
pub fn triangular_weight(segment_frames: usize) -> Vec<f32> {
    let half = segment_frames / 2;
    let mut values = vec![0.0f32; segment_frames];
    for (index, value) in values.iter_mut().enumerate().take(half) {
        *value = (index + 1) as f32;
    }
    for (index, value) in values.iter_mut().enumerate().skip(half) {
        *value = (segment_frames - index) as f32;
    }
    let maximum = values.iter().copied().fold(0.0f32, f32::max);
    if maximum > 0.0 {
        let scale = 1.0 / maximum;
        for value in values.iter_mut() {
            *value *= scale;
        }
    }
    values
}

/// One channel's overlap-add accumulator.
///
/// Segments arrive in any order, each with the absolute frame it starts at.
/// Samples are only final once no later segment can still touch them, which
/// is exactly the one-segment lookahead a streaming stem player needs.
#[derive(Clone, Debug, Default)]
pub struct OverlapAdd {
    values: Vec<f64>,
    weights: Vec<f64>,
    flushed: usize,
}

impl OverlapAdd {
    pub fn new() -> Self {
        Self::default()
    }

    fn grow(&mut self, to: usize) {
        if self.values.len() < to {
            self.values.resize(to, 0.0);
            self.weights.resize(to, 0.0);
        }
    }

    /// Adds one segment's contribution at `offset`.
    pub fn add(&mut self, offset: usize, segment: &[f32], window: &[f32]) {
        assert_eq!(segment.len(), window.len(), "segment and window length");
        self.grow(offset + segment.len());
        for (index, (sample, weight)) in segment.iter().zip(window).enumerate() {
            // The Demucs seam weights the contribution and the normaliser by
            // the same linear triangular value — not its square, which is the
            // STFT inverse's convention.
            self.values[offset + index] += *sample as f64 * *weight as f64;
            self.weights[offset + index] += *weight as f64;
        }
    }

    /// Every sample before `frame` is final; returns them normalised.
    pub fn flush_until(&mut self, frame: usize) -> Vec<f32> {
        let frame = frame.min(self.values.len());
        let output = (self.flushed..frame)
            .map(|index| {
                let weight = self.weights[index];
                if weight > 0.0 {
                    (self.values[index] / weight) as f32
                } else {
                    0.0
                }
            })
            .collect();
        self.flushed = frame;
        output
    }

    /// The remaining samples, after the last segment has landed.
    pub fn finish(&mut self) -> Vec<f32> {
        let end = self.values.len();
        self.flush_until(end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_plan_matches_the_swift_contract() {
        let plan = ChunkPlan::CONTRACT;
        assert_eq!(plan.stride(), 257_985);
        assert_eq!(plan.offsets(343_980), vec![0]);
        let offsets = plan.offsets(1_000_000);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets.last(), Some(&(1_000_000 - 343_980)));
    }

    #[test]
    fn offsets_are_ordered_and_cover_the_tail() {
        let plan = ChunkPlan::CONTRACT;
        let offsets = plan.offsets(2_000_000);
        assert!(offsets.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(offsets.last(), Some(&(2_000_000 - 343_980)));
    }

    #[test]
    fn overlap_add_reconstructs_a_constant() {
        // Every segment carries the same constant; the weight normalisation
        // must return that constant everywhere, seams included.
        let plan = ChunkPlan::CONTRACT;
        let total = 900_000;
        let window = triangular_weight(plan.segment_frames());
        let constant = 0.5f32;
        let segment = vec![constant; plan.segment_frames()];

        let mut accumulator = OverlapAdd::new();
        for offset in plan.offsets(total) {
            accumulator.add(offset, &segment, &window);
        }
        let output = accumulator.finish();
        assert!(output.len() >= total);
        for value in &output[..total] {
            assert!((value - constant).abs() < 1e-4, "got {value}");
        }
    }

    #[test]
    fn streaming_flush_is_identical_to_flushing_at_the_end() {
        let plan = ChunkPlan::CONTRACT;
        let window = triangular_weight(plan.segment_frames());
        let constant = 0.25f32;
        let segment = vec![constant; plan.segment_frames()];
        let total = 800_000;
        let offsets = plan.offsets(total);

        let mut whole = OverlapAdd::new();
        for offset in &offsets {
            whole.add(*offset, &segment, &window);
        }
        let batch = whole.finish();

        let mut streamed = OverlapAdd::new();
        let mut pieces: Vec<f32> = Vec::new();
        for (index, offset) in offsets.iter().enumerate() {
            streamed.add(*offset, &segment, &window);
            // Once the next segment has landed, everything before it is final.
            let boundary = offsets.get(index + 1).copied().unwrap_or(total);
            pieces.extend(streamed.flush_until(boundary));
        }
        assert_eq!(batch, pieces);
    }
}
