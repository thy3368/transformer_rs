# transformer_rs

A small decoder-only Transformer written from scratch in Rust. It trains on
plain text, saves a versioned checkpoint, evaluates next-token metrics, and
generates from a prompt on CPU with `f32` math.

See [getting started](doc/getting_started.md), [design](doc/design.md), and the
[math notes](doc/math.md).
