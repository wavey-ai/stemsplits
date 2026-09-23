//! Separate a whole track into four stems.
//!
//! The model works on one 7.8-second segment. A track is cut into overlapping
//! segments with the pinned `ChunkPlan`, each is separated, and the results are
//! sewn back with the triangular overlap-add seam — the same plan and seam the
//! phone uses, so a seam can only ever mean one thing.

use stemsplits_demucs::{triangular_weight, ChunkPlan, OverlapAdd};
use stemsplits_stft::{Geometry, Spectrum, Stft};

use crate::model::HtDemucs;
use crate::tensor::Tensor;

/// Four stems, each with a left and a right channel, each as long as the
/// input. Order is drums, bass, other, vocals.
pub fn separate(
    model: &HtDemucs,
    left: &[f32],
    right: &[f32],
    mut progress: impl FnMut(usize, usize),
) -> Vec<[Vec<f32>; 2]> {
    let geometry = Geometry::CONTRACT;
    let plan = ChunkPlan::CONTRACT;
    let total = left.len().min(right.len());
    let segment = geometry.segment;
    let offsets = plan.offsets(total);
    let window = triangular_weight(segment);
    let scale = (geometry.fft_size as f32).sqrt();
    let planes = geometry.bins * geometry.frames;
    let mut stft = Stft::new(geometry);

    let mut accumulators: Vec<[OverlapAdd; 2]> = (0..4)
        .map(|_| [OverlapAdd::new(), OverlapAdd::new()])
        .collect();
    let mut left_segment = vec![0.0f32; segment];
    let mut right_segment = vec![0.0f32; segment];

    for (index, &offset) in offsets.iter().enumerate() {
        let count = (total - offset).min(segment);
        left_segment[..count].copy_from_slice(&left[offset..offset + count]);
        right_segment[..count].copy_from_slice(&right[offset..offset + count]);
        for value in &mut left_segment[count..] {
            *value = 0.0;
        }
        for value in &mut right_segment[count..] {
            *value = 0.0;
        }

        let magnitude = Tensor::new(
            vec![1, 4, geometry.bins, geometry.frames],
            stft.spectral_input(&left_segment, &right_segment),
        );
        let mut interleaved = Vec::with_capacity(2 * segment);
        interleaved.extend_from_slice(&left_segment);
        interleaved.extend_from_slice(&right_segment);
        let waveform = Tensor::new(vec![1, 2, segment], interleaved);

        let (frequency, time) = model.forward(&magnitude, &waveform);
        for (stem, channels) in accumulators.iter_mut().enumerate() {
            for (channel, accumulator) in channels.iter_mut().enumerate() {
                let real_channel = stem * 4 + channel * 2;
                let real =
                    frequency.data[real_channel * planes..(real_channel + 1) * planes].to_vec();
                let imaginary = frequency.data
                    [(real_channel + 1) * planes..(real_channel + 2) * planes]
                    .to_vec();
                let inverse = stft.inverse(&Spectrum { real, imaginary });
                let time_base = (stem * 2 + channel) * segment;
                let mut rendered = vec![0.0f32; segment];
                for (position, value) in rendered.iter_mut().enumerate() {
                    *value = time.data[time_base + position] + inverse[position] * scale;
                }
                accumulator.add(offset, &rendered, &window);
            }
        }
        progress(index + 1, offsets.len());
    }

    accumulators
        .into_iter()
        .map(|channels| {
            let mut output = [Vec::new(), Vec::new()];
            for (channel, mut accumulator) in channels.into_iter().enumerate() {
                let mut values = accumulator.finish();
                values.truncate(total);
                output[channel] = values;
            }
            output
        })
        .collect()
}
