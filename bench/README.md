# Benchmarks

The question this directory answers: is cloud stem separation worth it, and
for which devices?

The ECDC cloud-offload work (`the ECDC cloud-offload work`) set
the method. Follow it:

- Measure the **slowest supported device**, not the Mac.
- Report the real-time factor: wall seconds per audio second, lower is faster.
- Fan-out only counts if per-segment times hold under concurrency; a single
  instance's idle cores do not.

## First measurement

One HTDemucs segment (343,980 frames, 7.8 s) on arm64 CPU:

- wall time per segment, and the RTF for a 210-second track at the planned
  segment stride;
- resident memory, to size the Lambda;
- cold-start model load, since ~100 MB of weights are in the image.

If the RTF is comfortably below 1 and fan-out scales, cloud stems are a
background step like ECDC. If it is above 1 on the target, the phone stays
the only path.

## Layout

```
bench/
  rtf/        the real-time-factor harness
  results/    recorded runs, JSON
```

Numbers go in `results/` with the machine, the model revision and the plan
recorded, so a later reader can tell what changed.
