# The Swift/vDSP oracle

`main.swift` is a copy of the STFT/iSTFT in Wavey's
`StemSeparator.swift`. It is the authoritative contract the Rust
port in `crates/stemsplits-stft` must match, because it is what the Core ML
HTDemucs package is fed on device.

It is a copy, not a dependency: Wavey ships to the App Store, and this
repository must be able to regenerate its own golden without building the
app. When `StemSeparator.swift` changes, change this too.

## Regenerate the golden

```sh
swift tools/oracle-stft/main.swift tools/oracle-stft/out
cp tools/oracle-stft/out/stft-small.bin crates/stemsplits-stft/tests/golden/
cargo test -p stemsplits-stft
```

`main.swift` writes two files:

- `stft-small.bin` — the committed golden. Small geometry so it fits in git.
- `stft-contract.bin` — the shipped geometry, ~20 MB, not committed. It
  prints the reconstruction SNR the Rust `roundtrip` test asserts against.

The binary layout is, per file: `left`, `right`, forward `real`, forward
`imaginary`, the four `spectral_input` planes, then the inverse — all little
endian `f32`, lengths fixed by the geometry.

`probe.swift` pins vDSP's inverse convention with controlled spectra. It
exists to settle arguments about scaling; it is not part of the test suite.
