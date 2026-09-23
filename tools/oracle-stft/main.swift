// The authoritative STFT/iSTFT contract, lifted from Wavey's on-device
// StemSeparator.swift. This is the oracle the Rust port in
// `crates/stemsplits-stft` is tested against: vDSP's exact arithmetic and
// scaling convention, which the Core ML HTDemucs package was fed with.
//
// It is deliberately a copy, not a dependency. Wavey ships to the App
// Store; this repo must be able to regenerate its own golden vectors without
// building the app. When StemSeparator.swift changes, change this too, and
// regenerate the golden.
//
// Run: swift tools/oracle-stft/main.swift <out-dir>

import Accelerate
import Foundation

struct Geometry {
    let fftSize: Int
    let hopSize: Int
    let bins: Int
    let frames: Int
    let segment: Int

    var padLeft: Int { hopSize / 2 * 3 }
    var padRight: Int { padLeft + frames * hopSize - segment }

    // The shipped contract.
    static let contract = Geometry(
        fftSize: 4_096,
        hopSize: 1_024,
        bins: 2_048,
        frames: 336,
        segment: 343_980
    )

    // Small enough to commit as a golden.
    static let small = Geometry(
        fftSize: 64,
        hopSize: 16,
        bins: 32,
        frames: 8,
        segment: 120
    )
}

func hann(_ geometry: Geometry) -> [Float] {
    (0..<geometry.fftSize).map { index in
        Float(
            0.5 * (1 - cos(
                2 * Double.pi * Double(index) / Double(geometry.fftSize)
            ))
        )
    }
}

func reflectPad(_ signal: [Float], left: Int, right: Int) -> [Float] {
    var output = [Float](repeating: 0, count: signal.count + left + right)
    for index in 0..<left {
        output[index] = signal[left - index]
    }
    output.replaceSubrange(left..<(left + signal.count), with: signal)
    for index in 0..<right {
        output[left + signal.count + index] = signal[signal.count - 2 - index]
    }
    return output
}

func forwardSpectrum(
    _ signal: [Float],
    _ geometry: Geometry,
    _ window: [Float]
) -> (real: [Float], imaginary: [Float]) {
    let exponent = vDSP_Length(log2(Float(geometry.fftSize)))
    guard let setup = vDSP_create_fftsetup(exponent, FFTRadix(kFFTRadix2))
    else { return ([], []) }
    defer { vDSP_destroy_fftsetup(setup) }
    let half = geometry.fftSize / 2
    var real = [Float](repeating: 0, count: geometry.bins * geometry.frames)
    var imaginary = real
    var frame = [Float](repeating: 0, count: geometry.fftSize)
    var splitReal = [Float](repeating: 0, count: half)
    var splitImaginary = splitReal
    for frameIndex in 0..<geometry.frames {
        let start = frameIndex * geometry.hopSize
        frame.replaceSubrange(
            0..<geometry.fftSize,
            with: signal[start..<(start + geometry.fftSize)]
        )
        vDSP_vmul(
            frame, 1, window, 1, &frame, 1, vDSP_Length(geometry.fftSize)
        )
        frame.withUnsafeBufferPointer { source in
            source.baseAddress!.withMemoryRebound(
                to: DSPComplex.self, capacity: half
            ) { complex in
                splitReal.withUnsafeMutableBufferPointer { realBuffer in
                    splitImaginary.withUnsafeMutableBufferPointer { imaginaryBuffer in
                        var split = DSPSplitComplex(
                            realp: realBuffer.baseAddress!,
                            imagp: imaginaryBuffer.baseAddress!
                        )
                        vDSP_ctoz(complex, 2, &split, 1, vDSP_Length(half))
                    }
                }
            }
        }
        splitReal.withUnsafeMutableBufferPointer { realBuffer in
            splitImaginary.withUnsafeMutableBufferPointer { imaginaryBuffer in
                var split = DSPSplitComplex(
                    realp: realBuffer.baseAddress!,
                    imagp: imaginaryBuffer.baseAddress!
                )
                vDSP_fft_zrip(
                    setup, &split, 1, exponent,
                    FFTDirection(kFFTDirection_Forward)
                )
            }
        }
        real[frameIndex] = splitReal[0] * 0.5
        for bin in 1..<geometry.bins {
            let index = bin * geometry.frames + frameIndex
            real[index] = splitReal[bin] * 0.5
            imaginary[index] = splitImaginary[bin] * 0.5
        }
    }
    return (real, imaginary)
}

func spectralInput(
    left: [Float],
    right: [Float],
    _ geometry: Geometry,
    _ window: [Float]
) -> [Float] {
    let planeCount = geometry.bins * geometry.frames
    var output = [Float](repeating: 0, count: 4 * planeCount)
    for (channel, signal) in [left, right].enumerated() {
        let padded = reflectPad(
            signal, left: geometry.padLeft, right: geometry.padRight
        )
        let spectrum = forwardSpectrum(padded, geometry, window)
        output.replaceSubrange(
            (channel * 2 * planeCount)..<((channel * 2 + 1) * planeCount),
            with: spectrum.real
        )
        output.replaceSubrange(
            ((channel * 2 + 1) * planeCount)..<((channel * 2 + 2) * planeCount),
            with: spectrum.imaginary
        )
    }
    var scale = Float(1 / sqrt(Double(geometry.fftSize)))
    vDSP_vsmul(output, 1, &scale, &output, 1, vDSP_Length(output.count))
    return output
}

func inverseSTFT(
    real: [Float],
    imaginary: [Float],
    frameCount: Int,
    outputCount: Int,
    _ geometry: Geometry,
    _ window: [Float]
) -> [Float] {
    let exponent = vDSP_Length(log2(Float(geometry.fftSize)))
    guard let setup = vDSP_create_fftsetup(exponent, FFTRadix(kFFTRadix2))
    else { return [] }
    defer { vDSP_destroy_fftsetup(setup) }
    let half = geometry.fftSize / 2
    var output = [Float](repeating: 0, count: outputCount)
    var weights = output
    var splitReal = [Float](repeating: 0, count: half)
    var splitImaginary = splitReal
    var frame = [Float](repeating: 0, count: geometry.fftSize)
    for frameIndex in 0..<frameCount {
        splitReal[0] = real[frameIndex] * 2
        splitImaginary[0] = 0
        for bin in 1..<geometry.bins {
            let index = bin * frameCount + frameIndex
            splitReal[bin] = real[index] * 2
            splitImaginary[bin] = imaginary[index] * 2
        }
        splitReal.withUnsafeMutableBufferPointer { realBuffer in
            splitImaginary.withUnsafeMutableBufferPointer { imaginaryBuffer in
                var split = DSPSplitComplex(
                    realp: realBuffer.baseAddress!,
                    imagp: imaginaryBuffer.baseAddress!
                )
                vDSP_fft_zrip(
                    setup, &split, 1, exponent,
                    FFTDirection(kFFTDirection_Inverse)
                )
                frame.withUnsafeMutableBufferPointer { outputBuffer in
                    outputBuffer.baseAddress!.withMemoryRebound(
                        to: DSPComplex.self, capacity: half
                    ) { complex in
                        vDSP_ztoc(&split, 1, complex, 2, vDSP_Length(half))
                    }
                }
            }
        }
        var scale = Float(1) / Float(2 * geometry.fftSize)
        vDSP_vsmul(
            frame, 1, &scale, &frame, 1, vDSP_Length(geometry.fftSize)
        )
        vDSP_vmul(
            frame, 1, window, 1, &frame, 1, vDSP_Length(geometry.fftSize)
        )
        let start = frameIndex * geometry.hopSize
        for sample in 0..<geometry.fftSize {
            let index = start + sample
            guard index < outputCount else { break }
            output[index] += frame[sample]
            weights[index] += window[sample] * window[sample]
        }
    }
    for index in output.indices where weights[index] > 0.000_000_01 {
        output[index] /= weights[index]
    }
    return output
}

func inverseSpectrum(
    real: [Float],
    imaginary: [Float],
    _ geometry: Geometry,
    _ window: [Float]
) -> [Float] {
    let totalFrames = geometry.frames + 4
    let pad = geometry.padLeft
    let center = geometry.fftSize / 2
    var paddedReal = [Float](repeating: 0, count: geometry.bins * totalFrames)
    var paddedImaginary = paddedReal
    for bin in 0..<geometry.bins {
        let source = bin * geometry.frames
        let destination = bin * totalFrames + 2
        paddedReal.replaceSubrange(
            destination..<(destination + geometry.frames),
            with: real[source..<(source + geometry.frames)]
        )
        paddedImaginary.replaceSubrange(
            destination..<(destination + geometry.frames),
            with: imaginary[source..<(source + geometry.frames)]
        )
    }
    let rawLength = (totalFrames - 1) * geometry.hopSize + geometry.fftSize
    let raw = inverseSTFT(
        real: paddedReal,
        imaginary: paddedImaginary,
        frameCount: totalFrames,
        outputCount: rawLength,
        geometry,
        window
    )
    let trim = center + pad
    return Array(raw[trim..<(trim + geometry.segment)])
}

// A deterministic input, so the Rust side reads it rather than regenerating it.
func inputSignal(count: Int, seed: UInt64) -> [Float] {
    var state = seed
    return (0..<count).map { _ in
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let unit = Float(Double(state >> 11) / Double(UInt64(1) << 53))
        return unit * 2 - 1
    }
}

func writeBinary(_ values: [Float], to handle: FileHandle) {
    var data = Data(capacity: values.count * 4)
    for value in values {
        withUnsafeBytes(of: value.bitPattern.littleEndian) {
            data.append(contentsOf: $0)
        }
    }
    handle.write(data)
}

let arguments = CommandLine.arguments
let outDirectory = arguments.count > 1 ? arguments[1] : "tools/oracle-stft/out"
try? FileManager.default.createDirectory(
    atPath: outDirectory, withIntermediateDirectories: true
)

for (name, geometry) in [("small", Geometry.small), ("contract", Geometry.contract)] {
    let window = hann(geometry)
    let left = inputSignal(count: geometry.segment, seed: 0x1234_5678)
    let right = inputSignal(count: geometry.segment, seed: 0x9abc_def0)
    let planes = spectralInput(
        left: left, right: right, geometry, window
    )
    let spectrum = forwardSpectrum(
        reflectPad(left, left: geometry.padLeft, right: geometry.padRight),
        geometry,
        window
    )
    let inverse = inverseSpectrum(
        real: spectrum.real, imaginary: spectrum.imaginary, geometry, window
    )

    // How well forward-then-inverse reconstructs, so the Rust round-trip test
    // has a reference number rather than a guess.
    var signalEnergy = 0.0
    var errorEnergy = 0.0
    for index in 0..<geometry.segment {
        let expected = Double(left[index])
        let actual = Double(inverse[index])
        signalEnergy += expected * expected
        let error = actual - expected
        errorEnergy += error * error
    }
    let snr = 10 * log10(signalEnergy / max(errorEnergy, .leastNonzeroMagnitude))
    print("\(name): segment=\(geometry.segment) snr=\(String(format: "%.2f", snr)) dB")

    let path = "\(outDirectory)/stft-\(name).bin"
    FileManager.default.createFile(atPath: path, contents: nil)
    guard let handle = FileHandle(forWritingAtPath: path) else {
        fatalError("cannot open \(path)")
    }
    writeBinary(left, to: handle)
    writeBinary(right, to: handle)
    writeBinary(spectrum.real, to: handle)
    writeBinary(spectrum.imaginary, to: handle)
    writeBinary(planes, to: handle)
    writeBinary(inverse, to: handle)
    try handle.close()
    print("\(name): wrote \(path)")
}
