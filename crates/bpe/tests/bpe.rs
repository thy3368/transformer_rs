use bpe::{BpeError, BpeTokenizer, BpeTrainer, BpeTrainerConfig, SPECIAL_TOKENS};
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
// 验证训练器会创建词表和合并规则文件，注册全部特殊 token，并支持基本编解码。
fn training_creates_loadable_files_with_special_tokens(corpus: (TempDir, std::path::PathBuf)) {
    let (dir, dataset) = corpus;
    let output = dir.path().join("nested/output");
    let (vocab, merges) = trainer(24).train_file(dataset, &output, false).unwrap();
    assert!(vocab.is_file());
    assert!(merges.is_file());

    let tokenizer = BpeTokenizer::from_files(vocab, merges).unwrap();
    assert!(tokenizer.config().vocab_size <= 24);
    assert!(tokenizer.config().vocab_size >= SPECIAL_TOKENS.len());
    assert_eq!(tokenizer.config().kind.len(), "bpe-v1:".len() + 64);
    assert_eq!(
        tokenizer
            .decode(&tokenizer.encode("lower", true).unwrap())
            .unwrap(),
        "lower"
    );
    for id in [
        tokenizer.config().unk_token,
        tokenizer.config().bos_token,
        tokenizer.config().eos_token,
        tokenizer.config().pad_token,
    ] {
        assert!((id as usize) < tokenizer.config().vocab_size);
    }
}

#[rstest]
// 验证未收录的显式 token 会映射到 [UNK] 的 ID。
fn unknown_tokens_and_explicit_tokens_map_to_unk(corpus: (TempDir, std::path::PathBuf)) {
    let (dir, dataset) = corpus;
    let (vocab, merges) = trainer(16)
        .train_file(dataset, dir.path().join("bpe"), false)
        .unwrap();
    let tokenizer = BpeTokenizer::from_files(vocab, merges).unwrap();
    let ids = tokenizer
        .tokens_to_ids(&bpe::BpeTokens(vec!["never-seen".into()]), false)
        .unwrap();
    assert_eq!(ids, vec![tokenizer.config().unk_token]);
}

#[rstest]
// 验证默认禁止覆盖已有输出文件，force=true 时允许重新生成。
fn existing_output_is_rejected_unless_forced(corpus: (TempDir, std::path::PathBuf)) {
    let (dir, dataset) = corpus;
    let output = dir.path().join("bpe");
    trainer(16).train_file(&dataset, &output, false).unwrap();
    assert!(matches!(
        trainer(16).train_file(&dataset, &output, false),
        Err(BpeError::OutputExists(_))
    ));
    trainer(18).train_file(dataset, output, true).unwrap();
}

#[rstest]
#[case(3, 1)]
#[case(10, 0)]
// 验证词表过小或最小频率为零等非法训练配置会被拒绝。
fn invalid_training_config_is_rejected(#[case] vocab_size: usize, #[case] min_frequency: u64) {
    assert!(matches!(
        BpeTrainer::new(BpeTrainerConfig {
            vocab_size,
            min_frequency,
            show_progress: false
        }),
        Err(BpeError::InvalidConfig(_))
    ));
}

#[rstest]
// 验证缺失文件和无法解析的词表/合并文件会返回明确错误。
fn missing_and_invalid_files_are_rejected(corpus: (TempDir, std::path::PathBuf)) {
    let (dir, _) = corpus;
    assert!(matches!(
        BpeTokenizer::from_files(dir.path().join("missing"), dir.path().join("also-missing")),
        Err(BpeError::Io(_))
    ));
    let vocab = dir.path().join("vocab.json");
    let merges = dir.path().join("merges.txt");
    std::fs::write(&vocab, "not-json").unwrap();
    std::fs::write(&merges, "not-merges").unwrap();
    assert!(matches!(
        BpeTokenizer::from_files(vocab, merges),
        Err(BpeError::Tokenizer(_))
    ));
}

#[rstest]
// 验证词表或合并规则内容变化会产生不同的 SHA-256 指纹，即使词表大小相同。
fn file_contents_are_part_of_the_fingerprint(corpus: (TempDir, std::path::PathBuf)) {
    let (dir, dataset) = corpus;
    let first = dir.path().join("first");
    let second = dir.path().join("second");
    let (v1, m1) = trainer(20).train_file(&dataset, &first, false).unwrap();
    std::fs::write(&dataset, "alpha alpha beta beta gamma gamma delta\n").unwrap();
    let (v2, m2) = trainer(20).train_file(dataset, &second, false).unwrap();
    let a = BpeTokenizer::from_files(v1, m1).unwrap();
    let b = BpeTokenizer::from_files(v2, m2).unwrap();
    assert_eq!(a.config().vocab_size, b.config().vocab_size);
    assert_ne!(a.config().fingerprint, b.config().fingerprint);
}
