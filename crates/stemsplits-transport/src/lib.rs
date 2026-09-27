//! The compressed browser-to-model wire contract.
use frame_header::EncodingFlag;
use rubato::{FftFixedInOut, Resampler};
use soundkit::frame_stream::{SoundKitFrameStream, SoundKitFrameStreamOptions};
use soundkit_opus::Decoder as OpusDecoder;
use soundkit_stream::{encode_interleaved_i16_to_opus_soundkit_stream, PcmOpusStreamOptions};

pub const FORMAT: &str = "soundkit-v2-opus-192";
pub const MAX_REQUEST_BYTES: usize = 512 * 1024;
pub const MAX_RESPONSE_BYTES: usize = 4 * MAX_REQUEST_BYTES;
const MAGIC: &[u8; 8] = b"STEMSK2\0";
const MODEL_RATE: usize = 44_100;
const OPUS_RATE: usize = 48_000;
const FRAME: usize = 960;
const BITRATE: u32 = 192_000;
const PRE_SKIP: u16 = 120;

pub type Pair = [Vec<f32>; 2];

fn resample(input: &Pair, from: usize, to: usize, total: usize) -> Result<Pair, String> {
    if from == to {
        return Ok(input.clone());
    }
    let (mut a, mut b) = (from, to);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let unit = 2 * to / a;
    let chunk = 1024_usize.div_ceil(unit) * unit;
    let mut converter =
        FftFixedInOut::<f32>::new(from, to, chunk, 2).map_err(|error| error.to_string())?;
    let delay = converter.output_delay();
    let mut output = [Vec::new(), Vec::new()];
    let mut position = 0;
    while output[0].len() < total + delay {
        let count = converter.input_frames_next();
        let available = input[0].len().saturating_sub(position).min(count);
        let slices = [
            &input[0][position..position + available],
            &input[1][position..position + available],
        ];
        let block = converter
            .process_partial(if available == 0 { None } else { Some(&slices) }, None)
            .map_err(|error| error.to_string())?;
        for channel in 0..2 {
            output[channel].extend_from_slice(&block[channel]);
        }
        position += available;
    }
    Ok(output.map(|channel| channel[delay..delay + total].to_vec()))
}

fn encode_pair(pair: &Pair, frames: usize) -> Result<Vec<u8>, String> {
    if pair.iter().any(|channel| channel.len() != frames) {
        return Err("Opus input has invalid channel geometry".into());
    }
    let scaled = frames as u64 * OPUS_RATE as u64;
    let opus_frames = (scaled / MODEL_RATE as u64) as usize;
    if opus_frames as u64 * MODEL_RATE as u64 != scaled {
        return Err("Opus input duration does not map exactly to 48 kHz".into());
    }
    let converted = resample(pair, MODEL_RATE, OPUS_RATE, opus_frames)?;
    let padded_frames = (opus_frames + usize::from(PRE_SKIP)).div_ceil(FRAME) * FRAME;
    let mut pcm = vec![0_i16; padded_frames * 2];
    for frame in 0..opus_frames {
        for channel in 0..2 {
            let value = converted[channel][frame];
            if !value.is_finite() {
                return Err("Opus input contains a non-finite sample".into());
            }
            pcm[frame * 2 + channel] = (value * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
        }
    }
    // The extra tail covers the codec delay removed after decode. Each
    // encoded packet is an ordinary SoundKit v2 frame, exactly like the
    // stream.sk2 packet ranges sent by ECDC.
    let encoded = encode_interleaved_i16_to_opus_soundkit_stream(
        &pcm,
        PcmOpusStreamOptions {
            sample_rate: OPUS_RATE as u32,
            channels: 2,
            frame_size: FRAME as u32,
            bitrate: BITRATE,
            start_pts: 0,
            include_packet_crc32: true,
        },
    )?;
    Ok(encoded.stream)
}

fn decode_pair(stream: &[u8], frames: usize) -> Result<Pair, String> {
    let scaled = frames as u64 * OPUS_RATE as u64;
    let opus_frames = (scaled / MODEL_RATE as u64) as usize;
    if opus_frames as u64 * MODEL_RATE as u64 != scaled {
        return Err("Opus output duration does not map exactly from 48 kHz".into());
    }
    let mut frame_stream = SoundKitFrameStream::new(SoundKitFrameStreamOptions {
        max_buffered_bytes: MAX_REQUEST_BYTES,
        max_payload_bytes: 64 * 1024,
        verify_packet_crc32: true,
        cipher: None,
    });
    let packets = frame_stream.push(stream)?;
    if frame_stream.buffered_bytes() != 0 || packets.is_empty() {
        return Err("SoundKit v2 Opus stream is incomplete".into());
    }
    let mut decoder = OpusDecoder::new(OPUS_RATE as i32, 2).map_err(|error| error.to_string())?;
    let mut pcm = Vec::with_capacity((opus_frames + FRAME) * 2);
    for packet in packets {
        if *packet.header.encoding() != EncodingFlag::Opus
            || packet.header.sample_rate() != OPUS_RATE as u32
            || packet.header.channels() != 2
            || packet.header.frame_count() == 0
            || packet.header.frame_count() > FRAME as u32
        {
            return Err("SoundKit v2 stream has invalid Opus geometry".into());
        }
        let decoded = decoder
            .decode_i16_vec(&packet.payload, false)
            .map_err(|error| error.to_string())?;
        let declared = packet.header.frame_count() as usize * 2;
        if decoded.len() < declared {
            return Err("SoundKit v2 Opus packet decoded short".into());
        }
        pcm.extend_from_slice(&decoded[..declared]);
    }
    let skip = usize::from(PRE_SKIP) * 2;
    if pcm.len() < skip + opus_frames * 2 {
        return Err("SoundKit v2 Opus stream ended before the segment was complete".into());
    }
    let pcm = &pcm[skip..skip + opus_frames * 2];
    let converted: Pair = std::array::from_fn(|channel| {
        (0..opus_frames)
            .map(|frame| pcm[frame * 2 + channel] as f32 / 32768.0)
            .collect()
    });
    resample(&converted, OPUS_RATE, MODEL_RATE, frames)
}

pub fn encode(pairs: &[Pair]) -> Result<Vec<u8>, String> {
    let frames = pairs
        .first()
        .and_then(|pair| pair.first())
        .map(Vec::len)
        .ok_or("Opus transport needs at least one stereo pair")?;
    let mut output = Vec::new();
    output.extend_from_slice(MAGIC);
    output.push(u8::try_from(pairs.len()).map_err(|_| "Too many Opus streams")?);
    output.extend_from_slice(&[0; 3]);
    output.extend_from_slice(
        &u32::try_from(frames)
            .map_err(|_| "Opus segment is too long")?
            .to_le_bytes(),
    );
    for pair in pairs {
        let stream = encode_pair(pair, frames)?;
        output.extend_from_slice(
            &u32::try_from(stream.len())
                .map_err(|_| "Opus stream is too large")?
                .to_le_bytes(),
        );
        output.extend_from_slice(&stream);
    }
    Ok(output)
}

pub fn decode(bytes: &[u8], streams: usize, frames: usize) -> Result<Vec<Pair>, String> {
    if bytes.len() < 16 || &bytes[..8] != MAGIC || bytes[8] as usize != streams {
        return Err("Invalid stem SoundKit v2 envelope".into());
    }
    let declared = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    if declared != frames {
        return Err("Stem SoundKit v2 envelope has an invalid duration".into());
    }
    let mut cursor = 16;
    let mut output = Vec::with_capacity(streams);
    for _ in 0..streams {
        if cursor + 4 > bytes.len() {
            return Err("Stem SoundKit v2 envelope is incomplete".into());
        }
        let size = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        let end = cursor
            .checked_add(size)
            .ok_or("Stem SoundKit v2 stream size overflow")?;
        if size == 0 || end > bytes.len() {
            return Err("Stem SoundKit v2 stream is incomplete".into());
        }
        output.push(decode_pair(&bytes[cursor..end], frames)?);
        cursor = end;
    }
    if cursor != bytes.len() {
        return Err("Stem SoundKit v2 envelope has trailing bytes".into());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_keeps_geometry_and_is_compact() {
        let frames = 343_980;
        let pair = std::array::from_fn(|channel| {
            (0..frames)
                .map(|index| ((index as f32 * 0.017).sin() * 0.2) * (1.0 - channel as f32 * 0.1))
                .collect()
        });
        let encoded = encode(std::slice::from_ref(&pair)).unwrap();
        assert!(encoded.len() < 210_000, "{} bytes", encoded.len());
        let decoded = decode(&encoded, 1, frames).unwrap();
        assert_eq!(decoded[0][0].len(), frames);
        let signal: f64 = pair[0]
            .iter()
            .map(|sample| f64::from(*sample).powi(2))
            .sum();
        let error: f64 = pair[0]
            .iter()
            .zip(&decoded[0][0])
            .map(|(a, b)| f64::from(*a - *b).powi(2))
            .sum();
        assert!(10.0 * (signal / error).log10() > 30.0);
    }

    #[test]
    fn rejects_truncation_and_wrong_stream_count() {
        let pair = [vec![0.0; 343_980], vec![0.0; 343_980]];
        let encoded = encode(&[pair]).unwrap();
        assert!(decode(&encoded, 4, 343_980).is_err());
        assert!(decode(&encoded[..encoded.len() - 1], 1, 343_980).is_err());
    }
}
