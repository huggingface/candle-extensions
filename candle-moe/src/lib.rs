mod ffi;

use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{DType, Result, Storage, Tensor};
use half::{bf16, f16};

pub fn apply_topk_softmax_<
    T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
>(
    gating_output: &Tensor,
    topk_weight: &Tensor,
    topk_indices: &Tensor,
    token_expert_indices: &Tensor,
) -> Result<()> {
    let (g, g_l) = gating_output.storage_and_layout();
    let g: &candle::CudaStorage = match &*g {
        Storage::Cuda(g) => g,
        _ => candle::bail!("gating_output must be a cuda tensor"),
    };

    let (w, w_l) = topk_weight.storage_and_layout();
    let w = match &*w {
        Storage::Cuda(w) => w,
        _ => candle::bail!("topk_weight must be a cuda tensor"),
    };

    let (i, i_l) = topk_indices.storage_and_layout();
    let i = match &*i {
        Storage::Cuda(i) => i,
        _ => candle::bail!("topk_indices must be a cuda tensor"),
    };

    let (ei, ei_l) = token_expert_indices.storage_and_layout();
    let ei: &candle::CudaStorage = match &*ei {
        Storage::Cuda(ei) => ei,
        _ => candle::bail!("token_expert_indices must be a cuda tensor"),
    };

    let g_rank = g_l.stride().len();
    let w_rank = w_l.stride().len();
    let i_rank = i_l.stride().len();
    let ei_rank = ei_l.stride().len();

    if g_rank != 2 || w_rank != 2 || i_rank != 2 || ei_rank != 2 {
        candle::bail!(
            "apply_topk_softmax_inplace expects input tensors of rank 2 (w: {w_l:?}, i: {i_l:?}, ei: {ei_l:?}, g: {g_l:?})"
        )
    }

    // Get cuda slices for all tensors
    let g = g.as_cuda_slice::<T>()?;
    let w = w.as_cuda_slice::<T>()?;
    let i = i.as_cuda_slice::<u32>()?;
    let ei = ei.as_cuda_slice::<u32>()?;

    // Get cuda views for all tensors
    let g = g.slice(g_l.start_offset()..);
    let w = w.slice(w_l.start_offset()..);
    let i = i.slice(i_l.start_offset()..);
    let ei = ei.slice(ei_l.start_offset()..);

    let (num_tokens, top_k) = w_l.shape().dims2()?;
    let (_, num_experts) = g_l.shape().dims2()?;

    let is_pow2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if !is_pow2 || num_experts > 256 {
        candle::bail!(
            "num_experts should be power of 2 and smaller than 256 (num_experts: {num_experts:?})"
        )
    }

    if (num_tokens, top_k) != i_l.shape().dims2()? {
        candle::bail!(
            "shape mismatch topk_indices {:?}, expected {:?}",
            i_l.shape(),
            (num_tokens, top_k)
        )
    }

    if (num_tokens, top_k) != ei_l.shape().dims2()? {
        candle::bail!(
            "shape mismatch token_expert_indices {:?}, expected {:?}",
            ei_l.shape(),
            (num_tokens, top_k)
        )
    }

    let gate_ptr = *g.device_ptr() as *const core::ffi::c_void;
    let weight_ptr = *w.device_ptr() as *const core::ffi::c_void;
    let indices_ptr = *i.device_ptr() as *const core::ffi::c_void;
    let expert_indices_ptr = *ei.device_ptr() as *const core::ffi::c_void;

    unsafe {
        ffi::topk_softmax(
            gate_ptr,
            weight_ptr,
            indices_ptr,
            expert_indices_ptr,
            num_experts as i32,
            num_tokens as i32,
            top_k as i32,
        )
    }

    Ok(())
}

pub fn apply_topk_softmax_inplace(
    gating_output: &Tensor,
    topk_weight: &Tensor,
    topk_indices: &Tensor,
    token_expert_indices: &Tensor,
) -> Result<()> {
    match topk_weight.dtype() {
        DType::F16 => apply_topk_softmax_::<f16>(
            gating_output,
            topk_weight,
            topk_indices,
            token_expert_indices,
        ),
        DType::BF16 => apply_topk_softmax_::<bf16>(
            gating_output,
            topk_weight,
            topk_indices,
            token_expert_indices,
        ),
        DType::F32 => apply_topk_softmax_::<f32>(
            gating_output,
            topk_weight,
            topk_indices,
            token_expert_indices,
        ),
        dt => {
            candle::bail!(
                "apply_topk_softmax_inplace is only supported for f32, f16 and bf16 ({dt:?})"
            )
        }
    }
}

pub fn apply_moe_sum_<
    T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
>(
    input: &Tensor,
    output: &Tensor,
    num_token: usize,
    topk: usize,
    dtype: u32,
) -> Result<()> {
    let (i, i_l) = input.storage_and_layout();
    let i: &candle::CudaStorage = match &*i {
        Storage::Cuda(i) => i,
        _ => candle::bail!("input must be a cuda tensor"),
    };

    let (o, o_l) = output.storage_and_layout();
    let o: &candle::CudaStorage = match &*o {
        Storage::Cuda(o) => o,
        _ => candle::bail!("output must be a cuda tensor"),
    };

    let i_rank = i_l.stride().len();
    let o_rank = o_l.stride().len();

    if i_rank != 3 {
        candle::bail!("input should be rank 3 (input: {i_l:?})")
    }

    if o_rank != 2 {
        candle::bail!("output should be rank 2 (input: {o_l:?})")
    }

    // Get cuda slices for all tensors
    let i = i.as_cuda_slice::<T>()?;
    let o = o.as_cuda_slice::<T>()?;

    // Get cuda views for all tensors
    let i = i.slice(i_l.start_offset()..);
    let o = o.slice(o_l.start_offset()..);

    let (num_tokens, _, hidden_size) = i_l.shape().dims3()?;

    if (num_tokens, hidden_size) != o_l.shape().dims2()? {
        candle::bail!(
            "shape mismatch output {:?}, expected {:?}",
            o_l.shape(),
            (num_tokens, hidden_size)
        )
    }

    let input_ptr = *i.device_ptr() as *const core::ffi::c_void;
    let output_ptr = *o.device_ptr() as *const core::ffi::c_void;

    unsafe {
        ffi::moe_sum(
            input_ptr,
            output_ptr,
            hidden_size as i32,
            num_token as i32,
            topk as i32,
            dtype,
        )
    }

    Ok(())
}

pub fn apply_moe_sum_inplace(
    input: &Tensor,
    output: &Tensor,
    num_token: usize,
    topk: usize,
    dtype: u32,
) -> Result<()> {
    match input.dtype() {
        DType::F16 => apply_moe_sum_::<f16>(input, output, num_token, topk, dtype),
        DType::BF16 => apply_moe_sum_::<bf16>(input, output, num_token, topk, dtype),
        DType::F32 => apply_moe_sum_::<f32>(input, output, num_token, topk, dtype),
        dt => {
            candle::bail!("apply_moe_sum_inplace is only supported for f32, f16 and bf16 ({dt:?})")
        }
    }
}
