.PHONY: test fmt lint golden

test:
	cargo test

fmt:
	cargo fmt --all

lint:
	cargo clippy --all-targets -- -D warnings

# macOS only: regenerate the Swift/vDSP oracle and refresh the committed golden.
golden:
	swift tools/oracle-stft/main.swift tools/oracle-stft/out
	cp tools/oracle-stft/out/stft-small.bin crates/stemsplits-stft/tests/golden/
	cargo test -p stemsplits-stft
