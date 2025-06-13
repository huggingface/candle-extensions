use anyhow::Result;
use candle::{DType, Device, Tensor};

#[test]
fn moe_sum() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let seq_len = 8;
    let top_k: usize = 2;
    let hidden_size = 4;

    let input =
        Tensor::randn(0.0, 1.0, (seq_len, top_k, hidden_size), &device)?.to_dtype(DType::F16)?;
    let output = Tensor::zeros((seq_len, hidden_size), DType::F16, &device)?;

    candle_moe::apply_moe_sum_inplace(&input, &output, seq_len, top_k, 1)?;

    Ok(())
}
