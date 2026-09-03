use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use embedding::Embedding;
use std::collections::HashMap;

#[test]
fn lookup_shape_and_rows() -> candle_core::Result<()> {
    let device = Device::Cpu;
    // 构造三行容易区分的权重，便于验证每个 token 的查表结果。
    let weights = Tensor::new(
        &[0f32, 1., 2., 3., 10., 11., 12., 13., 20., 21., 22., 23.],
        &device,
    )?
    .reshape((3, 4))?;
    let mut tensors = HashMap::new();
    tensors.insert("embedding.weight".to_string(), weights);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let embedding = Embedding::from_var_builder(3, 4, vb)?;

    // 输入顺序 [2, 0, 1, 2] 应依次查到第 2、0、1、2 行。
    let output = embedding.forward(&[2, 0, 1, 2])?;

    // 4 个 token ID 会生成 4 个向量，每个向量的维度 d_model 为 4。
    assert_eq!(output.dims(), &[4, 4]);
    assert_eq!(
        output.to_vec2::<f32>()?,
        vec![
            vec![20., 21., 22., 23.],
            vec![0., 1., 2., 3.],
            vec![10., 11., 12., 13.],
            vec![20., 21., 22., 23.],
        ]
    );
    Ok(())
}

#[test]
fn batch_shape_and_bounds() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let embedding = Embedding::new(16, 5, &device)?;
    let ids = Tensor::new(&[[1u32, 2], [3, 4]], &device)?;
    assert_eq!(embedding.forward_tensor(&ids)?.dims(), &[2, 2, 5]);
    assert!(embedding.forward(&[16]).is_err());
    Ok(())
}

#[test]
fn batch_forward_requires_two_dimensional_ids() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let embedding = Embedding::new(8, 3, &device)?;
    let one_dimensional = Tensor::new(&[1u32, 2], &device)?;
    assert!(embedding.forward_tensor(&one_dimensional).is_err());
    let wrong_dtype = Tensor::new(&[[1i64, 2]], &device)?;
    assert!(embedding.forward_tensor(&wrong_dtype).is_err());
    Ok(())
}
