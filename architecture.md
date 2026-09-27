# Architecture

```text
ECG                              [B, 12, 1000]
Reshape                          [B*12, 1, 1000]
Conv1D(1,16,k=7,s=2,p=3), GELU
LayerNorm(16)                    [B*12, 500, 16]
Conv1D(16,32,k=5,s=2,p=2), GELU
LayerNorm(32)                    [B*12, 250, 32]
2 x bidirectional Mamba blocks   [B*12, 250, 32]
Restore lead axis                [B, 12, 250, 32]
Concatenate leads per time       [B, 250, 384]
Linear(384,64), GELU, dropout    [B, 250, 64]
Temporal mean and max, concat    [B, 128]
Linear(128,1)                    [B, 1] logits
Sigmoid                         P(MI)
```

The default model has 71,553 trainable parameters. Tokenizer and temporal weights
are shared across leads. LayerNorm acts only on features within one lead and
one time token, so there is no cross-lead interaction before the mixing layer.
Lead position is retained by concatenation in canonical order.

Each bidirectional block pre-normalizes its input, runs independent forward and
backward Mamba-1 modules, reverses the backward output back to chronological
order, concatenates both outputs, projects 64 to 32 features, applies dropout
0.2, and adds the original input as a residual.

## Mamba-1 and portable scan

Each direction expands to 64 channels, uses a causal depthwise convolution of
width 4 with SiLU, input-dependent B/C and time-step projections (rank 2), and
16 state dimensions. A is negative exp(A_log); D is a learned skip coefficient.
Positive time steps use softplus and initialize log-uniformly in [0.001, 0.1]
via the inverse-softplus bias. The output is gated with SiLU(z) and projected
back to 32 features.

For each time step, channel and state:

```text
a[t] = exp(delta[t] * A)
b[t] = delta[t] * B[t] * u[t]
h[t] = a[t] * h[t-1] + b[t],  h[-1] = 0
y[t] = sum_state(C[t] * h[t]) + D * u[t]
```

The associative composition `(a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1)` computes
prefixes within chunks of 64 tokens, carrying state between chunks. It avoids
dividing by tiny cumulative products. An analytic backward pass computes the
reverse adjoint recurrence and parameter/input gradients, retaining hidden
states instead of all prefix intermediates. Only standard PyTorch operations
are used. This is a selective SSM, not a fixed linear recurrence substitute.

Equations and initialization reference the official Mamba-1 sources:

- https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
- https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py

The implementation is independently written. Tests compare forward outputs and
all scan input gradients to a literal sequential recurrence across multiple
chunk boundaries; GPU checks compare CPU and MPS outputs and gradients.
