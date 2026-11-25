# Cross-scene transfer (leave-one-room-out)

Six synthetic rooms (16k surface points, 4x4 transmitter, 24 train / 24 test
views). Each room is held out in turn; the material-conditioned head has no
per-room parameters.

See `metrics.json`. Zero-shot 17.50 ± 2.02 dB beats per-room scratch trained on
4/8/16 views (15.63/16.57/16.94 dB) in every fold, and a per-room free embedding
stays below zero-shot by 1.35/0.92/0.62 dB.
