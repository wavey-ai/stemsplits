# Sample stems

Four stems produced by this implementation, for listening:

```
westside-30s/drums.wav
westside-30s/bass.wav
westside-30s/other.wav
westside-30s/vocals.wav
```

A 30-second excerpt of the Westside fixture, 44.1 kHz stereo, run through the
`separate` binary. The sum of the four stems tracks the mix to within ~2%
RMS, which is the model's own reconstruction error.

The WAVs are gitignored: they are tens of MB, and the source track's rights
are its own. The command is the record.

## Regenerate

```sh
ffmpeg -i <track> -ss 60 -t 30 -ar 44100 -ac 2 -c:a pcm_s16le /tmp/excerpt.wav
cargo run --release -p stemsplits-htdemucs --bin separate -- /tmp/excerpt.wav samples/westside-30s
```

A 30-second excerpt is five overlapping segments and took ~61 s (RTF 2.0) on
an Apple Silicon machine; the single-segment rate is RTF 1.6, and the
difference is the 25% segment overlap.
