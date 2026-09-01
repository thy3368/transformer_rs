# Decoder-only Transformer design

This project implements next-token prediction with a CPU-only `f32` decoder.
The public data flow consists of fourteen stages: tokenizer encode, token
embedding, positional encoding, input combine, QKV projection, masked multi-head
self-attention, attention residual/norm, feed-forward, FFN residual/norm,
decoder layer, decoder stack, output projection, next-token loss, and next-token
selection. Head splitting, causal masking, stable softmax, and head merging are
internal attention details.

## Layers and roles

```text
core
  entity: values, invariants, TransformerModel, math, Adam
  use_case: TrainText, EvaluateText, GenerateText, outbound ports

adapter
  inbound: CLI translation and result presentation
  outbound: byte tokenizer, text dataset reader, binary checkpoint store

infra
  filesystem/runtime/framework mechanisms
```

Source dependency view:

```text
inbound -> use_case -> entity
outbound -> port <- use_case
outbound -> infra
```

Call flow view:

```text
inbound -> use_case -> outbound -> infra
```

Roles identify responsibility, source dependencies constrain imports, and call
flow describes runtime control. Entity code does not know commands or queries.
Inbound code contains no model math, while outbound adapters only implement
ports owned by the use-case layer.

## Model and training

Each post-norm decoder layer computes:

```text
QKV = Linear(H)
A   = MultiHeadCausalAttention(QKV)
H1  = LayerNorm(H + Linear(A))
F   = Linear(ReLU(Linear(H1)))
H2  = LayerNorm(H1 + F)
```

Training shifts sequences by one token, computes mean cross-entropy ignoring
PAD, backpropagates through explicit caches, clips the global gradient norm, and
updates parameters with Adam. Evaluation does only forward and loss. Generation
does only forward and token selection.

## Checkpoints

The core exports a filesystem-independent `ModelSnapshot`. The binary adapter
wraps it with `TRRS`, format version 1, and serializes named, shaped `f32`
parameters. Loading validates the envelope, configuration, names, uniqueness,
shapes, lengths, and finite values. Adam moments are not stored, so v1 supports
inference and evaluation rather than exact training resumption.

## Deliberate limits

There is no encoder, cross-attention, generic autograd, dropout, GPU backend,
distributed execution, or beam search. Those remain outside the small
training/checkpoint/evaluation/generation loop.
# Tokenization

The default command-line workflow uses the independent `bpe` workspace crate.
It trains a whitespace-pretokenized BPE vocabulary containing `[UNK]`, `[BOS]`,
`[EOS]`, and `[PAD]`. The application crate exposes a thin adapter implementing
the core `Tokenizer` port, while the byte-level tokenizer remains available to
library users.

Checkpoint tokenizer metadata includes a length-delimited SHA-256 fingerprint
of the raw vocabulary and merges files. Loading with different BPE files is
therefore rejected before model evaluation or generation.
