// Pins vDSP_fft_zrip's inverse convention, so the Rust port can reproduce it
// rather than approximate it. Prints one inverse frame for controlled spectra.

import Accelerate
import Foundation

func inverseFrame(
    real: [Float],
    imaginary: [Float],
    fftSize: Int,
    bins: Int,
    frameCount: Int
) -> [Float] {
    let exponent = vDSP_Length(log2(Float(fftSize)))
    guard let setup = vDSP_create_fftsetup(exponent, FFTRadix(kFFTRadix2))
    else { return [] }
    defer { vDSP_destroy_fftsetup(setup) }
    let half = fftSize / 2
    var output = [Float](repeating: 0, count: fftSize)
    var splitReal = [Float](repeating: 0, count: half)
    var splitImaginary = splitReal
    var frame = [Float](repeating: 0, count: fftSize)
    for frameIndex in 0..<frameCount {
        splitReal[0] = real[frameIndex] * 2
        splitImaginary[0] = 0
        for bin in 1..<bins {
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
        var scale = Float(1) / Float(2 * fftSize)
        vDSP_vsmul(frame, 1, &scale, &frame, 1, vDSP_Length(fftSize))
        for sample in 0..<fftSize { output[sample] += frame[sample] }
    }
    return output
}

func emit(_ label: String, _ values: [Float]) {
    let text = values.map { String(format: "%.9g", $0) }.joined(separator: ", ")
    print("\(label): [\(text)]")
}

let n = 8, bins = 4, hop = 4

var real = [Float](repeating: 0, count: bins)
var imag = [Float](repeating: 0, count: bins)
real[1] = 1
emit("bin1_real1", inverseFrame(
    real: real, imaginary: imag, fftSize: n, bins: bins, frameCount: 1
))

real = [Float](repeating: 0, count: bins)
real[0] = 1
emit("dc1", inverseFrame(
    real: real, imaginary: imag, fftSize: n, bins: bins, frameCount: 1
))

real = [Float](repeating: 0, count: bins)
imag[1] = 1
emit("bin1_imag1", inverseFrame(
    real: real, imaginary: imag, fftSize: n, bins: bins, frameCount: 1
))
