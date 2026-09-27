//! Whole-track reconstruction uses the same SplitSession as the browser.
use aws_sdk_lambda::{primitives::Blob, types::InvocationType, Client as LambdaClient};
use aws_sdk_s3::{primitives::ByteStream, Client as S3Client};
use frame_header::EncodingFlag;
use futures::future::try_join_all;
use lambda_runtime::{service_fn, Error, LambdaEvent};
use rubato::{FftFixedInOut, Resampler};
use serde::Deserialize;
use serde_json::{json, Value};
use soundkit::frame_stream::SoundKitFrameStream;
use soundkit_library::v2::SoundKitV2Decoder;
use soundkit_stream::{encode_interleaved_i16_to_opus_soundkit_stream, PcmOpusStreamOptions};
use stemsplits_web::SplitSession;

const ROLES: [&str; 4] = ["drums", "bass", "other", "vocals"];
const MAX_SOURCE_BYTES: usize = 200_000_000;
const MAX_RESPONSE_BYTES: usize = 3_000_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    kind: String,
    bucket: String,
    source_key: String,
}

fn stem_keys(source_key: &str) -> Result<Vec<String>, Error> {
    let prefix = source_key
        .strip_suffix("/original-lossless.sk2")
        .ok_or("Invalid source key")?;
    let job = prefix.strip_prefix("jobs/").ok_or("Invalid job prefix")?;
    if job.len() != 36 || !job.bytes().all(|c| c.is_ascii_hexdigit() || c == b'-') {
        return Err("Invalid job ID".into());
    }
    Ok(ROLES
        .iter()
        .map(|role| format!("{prefix}/stems/{role}.opus.sk2"))
        .collect())
}

fn decode_source(bytes: &[u8]) -> Result<(Vec<i16>, usize, usize), Error> {
    if bytes.is_empty() || bytes.len() > MAX_SOURCE_BYTES {
        return Err("Source size exceeds mastering limit".into());
    }
    let mut frames = SoundKitFrameStream::default();
    let mut decoder = SoundKitV2Decoder::new();
    let (mut pcm, mut rate, mut channels) = (Vec::new(), 0, 0);
    for chunk in bytes.chunks(256 * 1024) {
        for frame in frames.push(chunk)? {
            if frame.header.encoding() != &EncodingFlag::FLAC {
                return Err("Expected a lossless original".into());
            }
        }
        for block in decoder.push_float(chunk)?.frames {
            let next_rate = block.sampling_rate() as usize;
            let next_channels = block.channel_count() as usize;
            if ![44_100, 48_000, 88_200, 96_000].contains(&next_rate)
                || ![1, 2].contains(&next_channels)
                || (rate != 0 && (rate != next_rate || channels != next_channels))
            {
                return Err("Invalid source geometry".into());
            }
            rate = next_rate;
            channels = next_channels;
            if !block.data().len().is_multiple_of(channels * 4) {
                return Err("Incomplete source PCM".into());
            }
            for sample in block.data().chunks_exact(4) {
                let value = f32::from_le_bytes(sample.try_into()?);
                if !value.is_finite() {
                    return Err("Non-finite source PCM".into());
                }
                pcm.push((value * 32768.0).round().clamp(-32768.0, 32767.0) as i16);
            }
            if pcm.len() > rate * channels * 300 {
                return Err("Source exceeds five minutes".into());
            }
        }
    }
    if rate == 0
        || pcm.len() < rate * channels * 3
        || frames.buffered_bytes() != 0
        || decoder.buffered_bytes() != 0
    {
        return Err("Incomplete or too short source".into());
    }
    Ok((pcm, channels, rate))
}

fn append_planar(out: &mut [Vec<i16>; 4], planar: &[f32]) -> Result<(), Error> {
    if !planar.len().is_multiple_of(8) {
        return Err("Invalid reconstructed stem count".into());
    }
    let frames = planar.len() / 8;
    for (role, output) in out.iter_mut().enumerate() {
        let left = &planar[(role * 2) * frames..(role * 2 + 1) * frames];
        let right = &planar[(role * 2 + 1) * frames..(role * 2 + 2) * frames];
        for (&l, &r) in left.iter().zip(right) {
            output.push(quantize(l));
            output.push(quantize(r));
        }
    }
    Ok(())
}

fn quantize(value: f32) -> i16 {
    (value * 32767.0).round().clamp(-32768.0, 32767.0) as i16
}

fn to_opus_rate(pcm: &[i16]) -> Result<Vec<i16>, Error> {
    let frames = pcm.len() / 2;
    let input: [Vec<f32>; 2] = std::array::from_fn(|ch| {
        pcm.chunks_exact(2)
            .map(|frame| frame[ch] as f32 / 32768.0)
            .collect()
    });
    let mut resampler = FftFixedInOut::<f32>::new(44_100, 48_000, 1280, 2)?;
    let delay = resampler.output_delay();
    let total = ((frames as u64 * 48_000 + 22_050) / 44_100) as usize;
    let mut output: [Vec<f32>; 2] = std::array::from_fn(|_| Vec::new());
    let mut position = 0;
    while output[0].len() < total + delay {
        let count = resampler.input_frames_next();
        let available = frames.saturating_sub(position).min(count);
        let slices = [
            &input[0][position..position + available],
            &input[1][position..position + available],
        ];
        let block =
            resampler.process_partial(if available == 0 { None } else { Some(&slices) }, None)?;
        for channel in 0..2 {
            output[channel].extend_from_slice(&block[channel]);
        }
        position += available;
    }
    let mut interleaved = Vec::with_capacity(total * 2);
    for (&left, &right) in output[0][delay..delay + total]
        .iter()
        .zip(&output[1][delay..delay + total])
    {
        interleaved.push(quantize(left));
        interleaved.push(quantize(right));
    }
    Ok(interleaved)
}

fn retryable(status: reqwest::StatusCode) -> bool {
    status.is_server_error() || [408, 425, 429].contains(&status.as_u16())
}

async fn separate_segment(
    client: &reqwest::Client,
    endpoint: &str,
    key: &str,
    body: Vec<u8>,
    index: usize,
) -> Result<Vec<u8>, Error> {
    for attempt in 0..3 {
        let result = client
            .post(format!("{endpoint}/separate"))
            .header("x-api-key", key)
            .header("x-stems-format", "soundkit-v2-opus-192")
            .header("content-type", "application/octet-stream")
            .body(body.clone())
            .send()
            .await;
        match result {
            Ok(response) if response.status().is_success() => {
                let bytes = response.bytes().await?;
                if bytes.is_empty() || bytes.len() > MAX_RESPONSE_BYTES {
                    return Err("Invalid stem segment response".into());
                }
                return Ok(bytes.to_vec());
            }
            Ok(response) if retryable(response.status()) && attempt < 2 => {}
            Ok(response) => {
                return Err(format!("Stem segment {index} failed: {}", response.status()).into())
            }
            Err(_) if attempt < 2 => {}
            Err(error) => return Err(error.into()),
        }
        tokio::time::sleep(std::time::Duration::from_millis(
            500 * (1 << attempt) + index as u64 * 37,
        ))
        .await;
    }
    Err("Stem segment retries exhausted".into())
}

async fn run_job(request: &Request, s3: &S3Client, lambda: &LambdaClient) -> Result<Value, Error> {
    let keys = stem_keys(&request.source_key)?;
    let input = s3
        .get_object()
        .bucket(&request.bucket)
        .key(&request.source_key)
        .send()
        .await?;
    if input.content_length().unwrap_or(0) > MAX_SOURCE_BYTES as i64 {
        return Err("Source size exceeds mastering limit".into());
    }
    let bytes = input.body.collect().await?.into_bytes();
    let (pcm, channels, rate) = decode_source(&bytes)?;
    let mut session = SplitSession::new(&pcm, channels, rate)?;
    let (endpoint, key) = (
        std::env::var("STEMS_ENDPOINT")?,
        std::env::var("STEMS_API_KEY")?,
    );
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()?;
    let mut output: [Vec<i16>; 4] = std::array::from_fn(|_| Vec::new());
    for start in (0..session.count()).step_by(8) {
        let stop = (start + 8).min(session.count());
        let calls = (start..stop).map(|index| {
            let body = session.request(index);
            let client = client.clone();
            let endpoint = endpoint.clone();
            let key = key.clone();
            async move {
                Ok::<_, Error>((
                    index,
                    separate_segment(&client, &endpoint, &key, body?, index).await?,
                ))
            }
        });
        for (index, response) in try_join_all(calls).await? {
            append_planar(&mut output, &session.accept(index, &response)?)?;
        }
    }
    let expected_samples = session.frames() * 2;
    for (samples, object_key) in output.into_iter().zip(&keys) {
        if samples.len() != expected_samples {
            return Err("Reconstructed stem duration changed".into());
        }
        let opus_pcm = to_opus_rate(&samples)?;
        let encoded = encode_interleaved_i16_to_opus_soundkit_stream(
            &opus_pcm,
            PcmOpusStreamOptions {
                sample_rate: 48_000,
                channels: 2,
                bitrate: 192_000,
                ..PcmOpusStreamOptions::default()
            },
        )?;
        let put = s3
            .put_object()
            .bucket(&request.bucket)
            .key(object_key)
            .if_none_match("*")
            .content_type("application/octet-stream")
            .body(ByteStream::from(encoded.stream.clone()))
            .send()
            .await;
        if let Err(error) = put {
            let existing = s3
                .get_object()
                .bucket(&request.bucket)
                .key(object_key)
                .send()
                .await?;
            let body = existing.body.collect().await?.into_bytes();
            if body.as_ref() != encoded.stream.as_slice() {
                return Err(format!("Stem upload conflict: {error}").into());
            }
        }
    }
    let invocation = lambda
        .invoke()
        .function_name(std::env::var("PLANNER_FUNCTION")?)
        .invocation_type(InvocationType::Event)
        .payload(Blob::new(
            json!({"kind":"plan","bucket":request.bucket,
            "source_key":request.source_key,"stem_keys":keys})
            .to_string(),
        ))
        .send()
        .await?;
    if invocation.status_code() != 202 {
        return Err("Mastering planner rejected stems".into());
    }
    Ok(json!({"schema":"stemsplits.mastering.1","stems":4,"frames":expected_samples / 2}))
}

async fn handler(
    event: LambdaEvent<Request>,
    s3: &S3Client,
    lambda: &LambdaClient,
) -> Result<Value, Error> {
    let request = event.payload;
    if request.kind != "split-for-mastering"
        || request.bucket != std::env::var("JOB_BUCKET")?
        || stem_keys(&request.source_key).is_err()
    {
        return Err("Invalid whole-track request".into());
    }
    match run_job(&request, s3, lambda).await {
        Ok(done) => Ok(done),
        Err(error) => {
            eprintln!("Whole-track separation failed: {error}");
            if let Some(prefix) = request.source_key.strip_suffix("/original-lossless.sk2") {
                let _ = s3
                    .put_object()
                    .bucket(&request.bucket)
                    .key(format!("{prefix}/stems-error.json"))
                    .if_none_match("*")
                    .content_type("application/json")
                    .body(ByteStream::from(
                        json!({"error":"Cloud stem separation failed. Please retry."})
                            .to_string()
                            .into_bytes(),
                    ))
                    .send()
                    .await;
            }
            Err(error)
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let (s3, lambda) = (S3Client::new(&config), LambdaClient::new(&config));
    lambda_runtime::run(service_fn(move |event| {
        let (s3, lambda) = (s3.clone(), lambda.clone());
        async move { handler(event, &s3, &lambda).await }
    }))
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_requires_exact_source_and_four_role_keys() {
        assert_eq!(
            stem_keys("jobs/12345678-1234-1234-1234-123456789abc/original-lossless.sk2").unwrap(),
            ROLES.map(|role| format!(
                "jobs/12345678-1234-1234-1234-123456789abc/stems/{role}.opus.sk2"
            ))
        );
        assert!(stem_keys("jobs/../original-lossless.sk2").is_err());
        assert!(stem_keys("jobs/12345678-1234-1234-1234-123456789abc/other.sk2").is_err());
    }

    #[test]
    fn planar_reconstruction_interleaves_each_stem() {
        let mut stems = std::array::from_fn(|_| Vec::new());
        append_planar(&mut stems, &[0.5, -0.5, 0.25, -0.25, 0.0, 0.0, 1.0, -1.0]).unwrap();
        assert_eq!(stems[0], stemsplits_web::pcm16(&[0.5, -0.5]));
        assert_eq!(stems[1], stemsplits_web::pcm16(&[0.25, -0.25]));
        assert_eq!(stems[3], stemsplits_web::pcm16(&[1.0, -1.0]));
    }

    #[test]
    fn lossless_source_and_opus_estimate_keep_expected_timeline() {
        use soundkit_stream::{encode_interleaved_i16_to_soundkit_streams, PcmI16StreamOptions};
        let input = vec![0i16; 44_100 * 3 * 2];
        let source = encode_interleaved_i16_to_soundkit_streams(
            &vec![0i16; 48_000 * 3 * 2],
            PcmI16StreamOptions::default(),
        )
        .unwrap();
        let (decoded, channels, rate) = decode_source(&source.flac.stream).unwrap();
        assert_eq!((channels, rate), (2, 48_000));
        assert_eq!(decoded.len(), 48_000 * 3 * 2);
        let resampled = to_opus_rate(&input).unwrap();
        assert_eq!(resampled.len(), 48_000 * 3 * 2);
        let encoded = encode_interleaved_i16_to_opus_soundkit_stream(
            &resampled,
            PcmOpusStreamOptions {
                bitrate: 192_000,
                ..PcmOpusStreamOptions::default()
            },
        )
        .unwrap();
        assert!(!encoded.stream.is_empty());
    }

    #[test]
    fn segment_retries_only_transient_statuses() {
        assert!(retryable(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(retryable(reqwest::StatusCode::BAD_GATEWAY));
        assert!(!retryable(reqwest::StatusCode::FORBIDDEN));
        assert!(!retryable(reqwest::StatusCode::BAD_REQUEST));
    }
}
