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

fn trainer() -> BpeTrainer {
    BpeTrainer::new(BpeTrainerConfig {
        // Keep enough capacity for the corpus words to remain decodable with
        // the whitespace pre-tokenizer.
        vocab_size: 100,
        min_frequency: 1,
        show_progress: false,
    })
    .unwrap()
}

#[rstest]
#[case("lower")]
#[case("lower newer")]
#[case("lowest widest")]
fn text_roundtrips_through_encode_and_decode(
    corpus: (TempDir, std::path::PathBuf),
    #[case] text: &str,
) {
    let (dir, dataset) = corpus;
    let (vocab, merges) = trainer()
        .train_file(dataset, dir.path().join("bpe"), false)
        .unwrap();
    let tokenizer = BpeTokenizer::from_files(vocab, merges).unwrap();

    let ids = tokenizer.encode(text, false).unwrap();
    assert!(!ids.is_empty());
    assert_eq!(tokenizer.decode(&ids).unwrap(), text);

    let special_ids = tokenizer.encode(text, true).unwrap();
    assert_eq!(special_ids.first(), Some(&tokenizer.config().bos_token));
    assert_eq!(special_ids.last(), Some(&tokenizer.config().eos_token));
    assert_eq!(tokenizer.decode(&special_ids).unwrap(), text);
}
