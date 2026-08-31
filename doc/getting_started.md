# Getting started

Requires Rust 1.85 or newer.

```bash
mkdir -p data checkpoints
printf 'hello transformer\nhello transformer\n' > data/train.txt

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
