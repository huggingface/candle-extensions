use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_transformers::models::deepseek2::{TopKLastDimOp, TopKOutput};

fn to_vec2_round(t: Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec2::<f32>()?;
    let t = t
        .iter()
        .map(|row| {
            row.iter()
                .map(|val| (val * b).round() / b)
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>();
    Ok(t)
}

#[test]
fn topk_softmax() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let seq_len = 8;
    let num_experts = 4;
    let top_k = 2;

    let weights = Tensor::randn(0.0, 1.0, (seq_len, num_experts), &device)?.to_dtype(DType::F32)?;
    let softmax_weights = candle_nn::ops::softmax_last_dim(&weights)?;

    let TopKOutput {
        values: expected_values,
        indices: expected_indices,
    } = softmax_weights.topk(top_k)?;

    let topk_weight = Tensor::zeros((seq_len, top_k), DType::F32, &device)?;
    let topk_indices = Tensor::zeros((seq_len, top_k), DType::U32, &device)?;
    let token_expert_indices = Tensor::zeros((seq_len, top_k), DType::U32, &device)?;

    candle_moe::apply_topk_softmax_inplace(
        &weights,
        &topk_weight,
        &topk_indices,
        &token_expert_indices,
    )?;

    assert_eq!(
        to_vec2_round(expected_values, 3)?,
        to_vec2_round(topk_weight, 3)?
    );

    assert_eq!(
        expected_indices.to_vec2::<u32>()?,
        topk_indices.to_vec2::<u32>()?,
    );

    Ok(())
}
