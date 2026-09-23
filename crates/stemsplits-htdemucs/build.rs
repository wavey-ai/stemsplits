fn main() {
    println!("cargo:rerun-if-changed=kernels/gemm.c");
    let target = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let mut build = cc::Build::new();
    build.file("kernels/gemm.c").opt_level(3).warnings(false);
    match target.as_str() {
        "x86_64" | "x86" => {
            build.flag_if_supported("-mavx2");
            build.flag_if_supported("-mfma");
        }
        // aarch64 has NEON by default; nothing to add.
        _ => {}
    }
    build.compile("stemsplits_gemm");
}
