# Mathematical conventions

For sequence length `T`, model width `D`, `H` heads, and head width `Dh=D/H`,
embeddings are scaled by `sqrt(D)` and added to sinusoidal positions.

```text
S[i,j] = dot(Q[i], K[j]) / sqrt(Dh),  j <= i
S[i,j] = -infinity,                   j > i
A[i]   = softmax(S[i]) V
```

Softmax subtracts the row maximum. Layer normalization uses population variance
and epsilon `1e-5`. The loss is mean negative log likelihood over non-PAD
targets. Linear, ReLU, layer-normalization, attention, embedding, and residual
gradients are computed explicitly. The global gradient norm is clipped to 1;
Adam uses beta1 0.9, beta2 0.999, and epsilon `1e-8`.
