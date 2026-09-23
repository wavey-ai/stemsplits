//! Minimal 16-bit PCM WAV reading and writing.
//!
//! Enough to feed the separator and write its stems; the app's own encoder
//! takes over from there. Handles the one format the tools produce.

use std::io::Write;
use std::path::Path;

use anyhow::{bail, Context, Result};

pub struct Wav {
    pub sample_rate: u32,
    pub channels: u16,
    /// Interleaved 16-bit samples.
    pub samples: Vec<i16>,
}

fn u16le(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn u32le(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

pub fn read(path: &Path) -> Result<Wav> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file: {}", path.display());
    }
    let mut cursor = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits = 0u16;
    let mut samples = Vec::new();
    while cursor + 8 <= bytes.len() {
        let id = &bytes[cursor..cursor + 4];
        let size = u32le(&bytes, cursor + 4) as usize;
        let body = cursor + 8;
        if id == b"fmt " && body + 16 <= bytes.len() {
            channels = u16le(&bytes, body + 2);
            sample_rate = u32le(&bytes, body + 4);
            bits = u16le(&bytes, body + 14);
        } else if id == b"data" && body + size <= bytes.len() {
            if bits != 16 {
                bail!("only 16-bit PCM is supported, this file is {bits}-bit");
            }
            samples = bytes[body..body + size]
                .chunks_exact(2)
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                .collect();
        }
        cursor = body + size + (size & 1);
    }
    if samples.is_empty() || channels == 0 {
        bail!("no audio data in {}", path.display());
    }
    Ok(Wav {
        sample_rate,
        channels,
        samples,
    })
}

/// Writes interleaved 16-bit PCM.
pub fn write(path: &Path, sample_rate: u32, channels: u16, samples: &[i16]) -> Result<()> {
    let payload = (samples.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + payload as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + payload).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * channels as u32 * 2).to_le_bytes());
    bytes.extend_from_slice(&(channels * 2).to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&payload.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    let mut file =
        std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    file.write_all(&bytes)?;
    Ok(())
}
