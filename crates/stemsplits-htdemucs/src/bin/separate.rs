//! Separate a stereo 44.1 kHz WAV into four stem WAVs.
//!
//!   cargo run --release -p stemsplits-htdemucs --bin separate -- <in.wav> <out-dir> [bundle-dir]
//!
//! Input is expected at the model's rate and channel count; prepare it with
//! `ffmpeg -ar 44100 -ac 2 -c:a pcm_s16le`.

use std::path::Path;

use stemsplits_htdemucs::model::HtDemucs;
use stemsplits_htdemucs::{separate, wav};
use stemsplits_model::Weights;

fn to_i16(value: f32) -> i16 {
    (value * 32_767.0).round().clamp(-32_768.0, 32_767.0) as i16
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 3 {
        eprintln!("usage: separate <in.wav> <out-dir> [bundle-dir]");
        std::process::exit(2);
    }
    let input = Path::new(&arguments[1]);
    let output_directory = Path::new(&arguments[2]);
    let bundle = arguments
        .get(3)
        .cloned()
        .unwrap_or_else(|| "tools/reference/out/bundle".into());

    let audio = wav::read(input).expect("read input wav");
    assert_eq!(audio.sample_rate, 44_100, "input must be 44.1 kHz");
    assert_eq!(audio.channels, 2, "input must be stereo");
    let left: Vec<f32> = audio
        .samples
        .iter()
        .step_by(2)
        .map(|sample| *sample as f32 / 32_768.0)
        .collect();
    let right: Vec<f32> = audio
        .samples
        .iter()
        .skip(1)
        .step_by(2)
        .map(|sample| *sample as f32 / 32_768.0)
        .collect();
    let seconds = left.len() as f64 / 44_100.0;
    println!("input: {:.1} s, {} frames", seconds, left.len());

    let weights = Weights::open(Path::new(&bundle)).expect("weight bundle");
    let model = HtDemucs::load(&weights).expect("model");

    let started = std::time::Instant::now();
    let stems = separate::separate(&model, &left, &right, |done, total| {
        eprintln!("segment {done}/{total}");
    });
    let elapsed = started.elapsed().as_secs_f64();
    println!("separated in {elapsed:.1} s (RTF {:.2})", elapsed / seconds);

    std::fs::create_dir_all(output_directory).expect("create output directory");
    let names = ["drums", "bass", "other", "vocals"];
    for (name, stem) in names.iter().zip(&stems) {
        let mut interleaved = Vec::with_capacity(left.len() * 2);
        for (left_sample, right_sample) in stem[0].iter().zip(&stem[1]) {
            interleaved.push(to_i16(*left_sample));
            interleaved.push(to_i16(*right_sample));
        }
        let path = output_directory.join(format!("{name}.wav"));
        wav::write(&path, 44_100, 2, &interleaved).expect("write stem");
        println!("wrote {}", path.display());
    }
}
