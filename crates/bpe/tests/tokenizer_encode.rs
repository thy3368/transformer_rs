use bpe::{BpeTokenizer, BpeTrainer, BpeTrainerConfig};
use rstest::{fixture, rstest};
use tempfile::TempDir;

#[fixture]
fn corpus() -> (TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let dataset = dir.path().join("corpus.txt");
    std::fs::write(&dataset, "lower lower newer newer lowest widest\n").unwrap();
    (dir, dataset)
}

fn trainer(vocab_size: usize) -> BpeTrainer {
    BpeTrainer::new(BpeTrainerConfig {
        vocab_size,
        min_frequency: 1,
        show_progress: false,
    })
    .unwrap()
}

#[rstest]
#[case("lower")]
#[case("lower newer")]
#[case("lowest widest")]
// 验证句子可以被切分并稳定映射为词表中的 token ID。
fn sentence_can_be_encoded_to_token_ids(
    corpus: (TempDir, std::path::PathBuf),
    #[case] text: &str,
) {
    let (dir, dataset) = corpus;
    let (vocab, merges) = trainer(24)
        .train_file(dataset, dir.path().join("bpe"), false)
        .unwrap();
    let tokenizer = BpeTokenizer::from_files(vocab, merges).unwrap();

    let tokens = tokenizer.tokenize(text).unwrap();
    let ids = tokenizer.encode(text, false).unwrap();
    let mapped_ids = tokenizer.tokens_to_ids(&tokens, false).unwrap();

    // 直接编码应与先分词再逐个映射得到的结果一致。
    assert_eq!(ids, mapped_ids);
    // 每个分词都应对应一个 token ID，且 ID 必须位于词表范围内。
    assert_eq!(ids.len(), tokens.0.len());
    assert!(ids
        .iter()
        .all(|&id| (id as usize) < tokenizer.config().vocab_size));
}

#[rstest]
// 验证启用特殊 token 后，编码结果以 BOS 开始、EOS 结束。
fn special_tokens_wrap_encoded_sentence(corpus: (TempDir, std::path::PathBuf)) {
    let (dir, dataset) = corpus;
    let (vocab, merges) = trainer(24)
        .train_file(dataset, dir.path().join("bpe"), false)
        .unwrap();
    let tokenizer = BpeTokenizer::from_files(vocab, merges).unwrap();

    let text = "lower newer";
    let plain_ids = tokenizer.encode(text, false).unwrap();
    let special_ids = tokenizer.encode(text, true).unwrap();

    // 中间内容应保持与未添加特殊 token 时完全相同。
    assert_eq!(special_ids.first(), Some(&tokenizer.config().bos_token));
    assert_eq!(special_ids.last(), Some(&tokenizer.config().eos_token));
    assert_eq!(&special_ids[1..special_ids.len() - 1], plain_ids.as_slice());
    assert_eq!(special_ids.len(), plain_ids.len() + 2);
}
