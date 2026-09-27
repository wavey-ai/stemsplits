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

/// One model segment. Output order is drums, bass, other, vocals, with planar channels.
pub fn separate_segment(
    model: &HtDemucs,
    left: &[f32],
    right: &[f32],
    stft: &mut Stft,
) -> Vec<[Vec<f32>; 2]> {
    let geometry = Geometry::CONTRACT;
    let segment = geometry.segment;
    assert_eq!(left.len(), segment);
    assert_eq!(right.len(), segment);
    let magnitude = Tensor::new(
        vec![1, 4, geometry.bins, geometry.frames],
        stft.spectral_input(left, right),
    );
    let mut planar = Vec::with_capacity(2 * segment);
    planar.extend_from_slice(left);
    planar.extend_from_slice(right);
    let waveform = Tensor::new(vec![1, 2, segment], planar);
    let (frequency, time) = model.forward(&magnitude, &waveform);
    let planes = geometry.bins * geometry.frames;
    let scale = (geometry.fft_size as f32).sqrt();
    (0..4)
        .map(|stem| {
            std::array::from_fn(|channel| {
                let real_channel = stem * 4 + channel * 2;
                let real =
                    frequency.data[real_channel * planes..(real_channel + 1) * planes].to_vec();
                let imaginary = frequency.data
                    [(real_channel + 1) * planes..(real_channel + 2) * planes]
                    .to_vec();
                let inverse = stft.inverse(&Spectrum { real, imaginary });
                let time_base = (stem * 2 + channel) * segment;
                inverse
                    .iter()
                    .enumerate()
                    .map(|(position, value)| time.data[time_base + position] + value * scale)
                    .collect()
            })
        })
        .collect()
}

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

        let rendered = separate_segment(model, &left_segment, &right_segment, &mut stft);
        for (stem, channels) in accumulators.iter_mut().enumerate() {
            for (channel, accumulator) in channels.iter_mut().enumerate() {
                accumulator.add(offset, &rendered[stem][channel], &window);
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
