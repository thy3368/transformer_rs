use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use embedding::Embedding;
use rstest::{fixture, rstest};
use std::collections::HashMap;

#[fixture]
fn fixed_embedding() -> Embedding {
    let device = Device::Cpu;
    let weights = Tensor::new(
        &[0f32, 1., 2., 3., 10., 11., 12., 13., 20., 21., 22., 23.],
        &device,
    )
    .unwrap()
    .reshape((3, 4))
    .unwrap();
    let mut tensors = HashMap::new();
    tensors.insert("embedding.weight".to_string(), weights);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    Embedding::from_var_builder(3, 4, vb).unwrap()
}

#[rstest]
#[case(0, vec![0., 1., 2., 3.])]
#[case(1, vec![10., 11., 12., 13.])]
#[case(2, vec![20., 21., 22., 23.])]
fn single_token_id_maps_to_vector(
    fixed_embedding: Embedding,
    #[case] token_id: u32,
    #[case] expected: Vec<f32>,
) {
    let output = fixed_embedding.forward(&[token_id]).unwrap();
    assert_eq!(output.dims(), &[1, 4]);
    assert_eq!(output.to_vec2::<f32>().unwrap(), vec![expected]);
}

#[rstest]
#[case(vec![2, 0, 1], vec![vec![20., 21., 22., 23.], vec![0., 1., 2., 3.], vec![10., 11., 12., 13.]])]
#[case(vec![1, 2, 0, 1], vec![vec![10., 11., 12., 13.], vec![20., 21., 22., 23.], vec![0., 1., 2., 3.], vec![10., 11., 12., 13.]])]
fn token_sequence_preserves_order(
    fixed_embedding: Embedding,
    #[case] token_ids: Vec<u32>,
    #[case] expected: Vec<Vec<f32>>,
) {
    let output = fixed_embedding.forward(&token_ids).unwrap();
    assert_eq!(output.dims(), &[token_ids.len(), 4]);
    assert_eq!(output.to_vec2::<f32>().unwrap(), expected);
}

#[rstest]
#[case(vec![2, 2, 0])]
#[case(vec![1, 1, 1, 2])]
fn repeated_token_ids_repeat_rows(fixed_embedding: Embedding, #[case] token_ids: Vec<u32>) {
    let output = fixed_embedding.forward(&token_ids).unwrap();
    let rows = output.to_vec2::<f32>().unwrap();
    for (id, row) in token_ids.iter().zip(rows.iter()) {
        assert_eq!(
            row,
            &vec![
                *id as f32 * 10.,
                *id as f32 * 10. + 1.,
                *id as f32 * 10. + 2.,
                *id as f32 * 10. + 3.
            ]
        );
    }
}

#[rstest]
fn two_dimensional_ids_preserve_batch_shape(fixed_embedding: Embedding) {
    let ids = Tensor::new(&[[2u32, 0], [1, 2]], &Device::Cpu).unwrap();
    let output = fixed_embedding.forward_tensor(&ids).unwrap();
    assert_eq!(output.dims(), &[2, 2, 4]);
    assert_eq!(
        output.to_vec3::<f32>().unwrap(),
        vec![
            vec![vec![20., 21., 22., 23.], vec![0., 1., 2., 3.]],
            vec![vec![10., 11., 12., 13.], vec![20., 21., 22., 23.]],
        ]
    );
}

#[rstest]
#[case(vec![3])]
#[case(vec![0, 4])]
fn out_of_range_token_id_returns_error(fixed_embedding: Embedding, #[case] token_ids: Vec<u32>) {
    assert!(fixed_embedding.forward(&token_ids).is_err());
}
