# candle-extensions

Candle extensions for more specialized kernels.
They usually do not have any backward equivalent but are faster than
raw candle expressions, usually because they *fuse* kernels directly.


- [candle-layer-norm](./candle-layer-norm)
- [candle-rotary](./candle-rotary)
- [candle-flash-attn-v1](./candle-flash-attn-v1)
