# Getting started

Requires Rust 1.85 or newer.

```bash
mkdir -p data checkpoints
printf 'hello transformer\nhello transformer\n' > data/train.txt

cargo run --release -- train-bpe \
  --dataset data/train.txt

cargo run --release -- train-text \
  --dataset data/train.txt \
  --checkpoint checkpoints/model.trrs \
  --epochs 10

cargo run --release -- evaluate \
  --dataset data/train.txt \
  --checkpoint checkpoints/model.trrs

cargo run --release -- generate \
  --checkpoint checkpoints/model.trrs \
  --prompt 'hello' \
  --max-tokens 16
```

For a quick smoke test, add `--d-model 8 --heads 2 --layers 1 --d-ff 16` to
training. Evaluation and generation recover the architecture from the
checkpoint.

`train-bpe` writes `data/vocab.json` and `data/merges.txt` by default and
refuses to overwrite either file unless `--force` is supplied. `train-text`,
`evaluate`, and `generate` use those paths by default. When custom `--vocab`
and `--merges` paths are used, evaluation and generation must receive the same
files used to create the checkpoint; the checkpoint stores their SHA-256
fingerprint and rejects a different vocabulary even when its size is equal.
