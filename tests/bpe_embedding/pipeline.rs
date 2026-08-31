use std::path::PathBuf;

use rstest::{fixture, rstest};
use transformer_rs::adapter::outbound::{BpeTokenizer, BpeTokens};
use transformer_rs::core::entity::{
    ModelConfig, Text, TokenIds, Tokenizer, TokenizerConfig, TransformerError, TransformerModel,
};

const D_MODEL: usize = 4;

#[fixture]
fn fixture_paths() -> (PathBuf, PathBuf) {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/bpe_embedding/fixtures");
    (fixtures.join("vocab.json"), fixtures.join("merges.txt"))
}

#[fixture]
fn tokenizer(fixture_paths: (PathBuf, PathBuf)) -> BpeTokenizer {
    BpeTokenizer::from_files(fixture_paths.0, fixture_paths.1).unwrap()
}

#[fixture]
fn model() -> TransformerModel {
    let config = ModelConfig {
        vocab_size: 15,
        max_seq_len: 16,
        d_model: D_MODEL,
        num_heads: 2,
        num_layers: 1,
        d_ff: 8,
    };
    let model = TransformerModel::new(config, 7).unwrap();
    let mut snapshot = model.snapshot(TokenizerConfig {
        kind: "bpe-v1".into(),
        vocab_size: 15,
        bos_token: 1,
        eos_token: 2,
        pad_token: 3,
    });
    let embedding = snapshot
        .parameters
        .iter_mut()
        .find(|parameter| parameter.name == "embedding.weight")
        .unwrap();
    embedding.values = (0..15 * D_MODEL).map(|value| value as f32).collect();
    TransformerModel::from_snapshot(snapshot).unwrap()
}

#[rstest]
// 验证完整词元的分词 -> ID -> embedding 链路。
#[case("low", &["low"], &[11])]
// 验证单词拆分为子词时的分词 -> ID -> embedding 链路。
#[case("lower", &["low", "er"], &[11, 12])]
// 验证包含多个单词的句子的分词 -> ID -> embedding 链路。
#[case("lower newer", &["low", "er", "new", "er"], &[11, 12, 14, 12])]
fn bpe_tokens_map_to_embedding_rows(
    tokenizer: BpeTokenizer,
    model: TransformerModel,
    #[case] text: &str,
    #[case] expected_tokens: &[&str],
    #[case] expected_ids: &[u32],
) {
    let tokens = tokenizer.tokenize(&Text(text.into())).unwrap();
    assert_eq!(tokens.0, expected_tokens);

    let ids = tokenizer.tokens_to_ids(&tokens, false).unwrap();
    assert_eq!(ids.0, expected_ids);

    assert_embedding_rows(&model, &ids.0);
}

#[rstest]
// 验证关闭 BOS/EOS token 时的 ID 和 embedding。
#[case(false, &[11, 12, 14, 12])]
// 验证启用 BOS/EOS token 时的 ID 和 embedding。
#[case(true, &[1, 11, 12, 14, 12, 2])]
fn special_tokens_are_embedded_at_the_sequence_ends(
    tokenizer: BpeTokenizer,
    model: TransformerModel,
    #[case] add_special_tokens: bool,
    #[case] expected_ids: &[u32],
) {
    let text = Text("lower newer".into());
    let tokens = tokenizer.tokenize(&text).unwrap();
    let ids = tokenizer
        .tokens_to_ids(&tokens, add_special_tokens)
        .unwrap();
    assert_eq!(ids.0, expected_ids);
    assert_eq!(tokenizer.encode(&text, add_special_tokens).unwrap(), ids);

    assert_embedding_rows(&model, &ids.0);
}

#[rstest]
// 验证未知子词会映射到 [UNK]。
fn unknown_subword_maps_to_unk(tokenizer: BpeTokenizer) {
    let tokens = tokenizer.tokenize(&Text("z".into())).unwrap();
    assert_eq!(tokens, BpeTokens(vec!["[UNK]".into()]));
    assert_eq!(
        tokenizer.tokens_to_ids(&tokens, false).unwrap(),
        TokenIds(vec![0])
    );
}

#[rstest]
// 验证越界 ID 会返回 InvalidData。
fn embedding_rejects_an_out_of_vocabulary_id(model: TransformerModel) {
    let error = model.embed_token_ids(&[15]).unwrap_err();
    assert!(matches!(error, TransformerError::InvalidData(_)));
}

fn assert_embedding_rows(model: &TransformerModel, ids: &[u32]) {
    // 逐行比较 embedding lookup 结果与确定性权重矩阵。
    let hidden = model.embed_token_ids(ids).unwrap();
    assert_eq!(hidden.0.shape, vec![ids.len(), D_MODEL]);
    for (row, &id) in ids.iter().enumerate() {
        let expected: Vec<f32> = (0..D_MODEL)
            .map(|column| (id as usize * D_MODEL + column) as f32)
            .collect();
        assert_eq!(&hidden.0.data[row * D_MODEL..(row + 1) * D_MODEL], expected);
    }
}
