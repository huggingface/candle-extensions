# candle-extensions

Candle extensions for more specialized kernels.
They usually do not have any backward equivalent but are faster than
raw candle expressions, usually because they *fuse* kernels directly.

- [candle-cublast](./candle-cublast)
- [candle-flash-attn-v1](./candle-flash-attn-v1)
- [candle-layer-norm](./candle-layer-norm)
- [candle-rotary](./candle-rotary)

> [!NOTE]
> To compile `candle-flash-attn-v1` you need first to pull the `cutlass` submodule as:
> ```bash
> git submodule update --init
> ```
