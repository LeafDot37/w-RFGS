# Diffuse-channel transfer

Protocol: a 60-instance diffuse variant of the benchmark, split 48/16 **by
location** (all array orientations seen in training). Per-scene SH and CORF-GS
see 48 views of the target instance; our field sees none.

See `metrics.json`. Headline: per-scene fits fall below the per-instance
constant-image floor (10.97/10.05 vs 11.22 dB), while the zero-shot field
reaches 11.87 dB and a 5k fine-tune 12.23 ± 0.14 dB over three seeds.
