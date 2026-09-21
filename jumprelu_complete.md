# JumpReLU SAE: Complete Derivation

## 1. What we want

An SAE decomposes a language model activation vector $\mathbf{x}$ into a sparse feature vector $\mathbf{f}$, then reconstructs $\mathbf{x}$ from $\mathbf{f}$.

- **Encoder**: computes pre-activations $\pi(x) = W_{\text{enc}} x + b_{\text{enc}}$
- **Activation**: produces sparse features $f = \text{activation}(\pi(x))$
- **Decoder**: reconstructs $\hat{x} = W_{\text{dec}} f + b_{\text{dec}}$

Two objectives fight:
1. Reconstruct $x$ faithfully (small $\|x - \hat{x}\|^2$)
2. Use few features (small number of non-zero $f_i$)

Standard ReLU SAE uses L1 penalty on $f$ for sparsity. Problem: L1 penalizes magnitudes, not just sparsity. The model shrinks active features to near-zero while keeping them technically non-zero. Reconstruction suffers.

JumpReLU separates the two decisions:
- **Threshold $\theta_i$**: decides WHETHER feature $i$ fires (gate)
- **Pre-activation $\pi_i$**: decides HOW MUCH it fires if it passes (magnitude)

This prevents the shrinkage cheat.

---

## 2. The loss function

We penalize exact count of active features (L0 norm), not sum of magnitudes (L1 norm).

$$L(x) = \|x - \hat{x}\|_2^2 + \lambda \, \|f\|_0$$

where $\|f\|_0 = \sum_i H(\pi_i(x) - \theta_i)$ is the count of features with pre-activation above threshold.

$H$ is the Heaviside step function:
- $H(z) = 1$ if $z > 0$ (feature passes, counts as active)
- $H(z) = 0$ if $z < 0$ (feature blocked, does not count)

The JumpReLU activation in the forward pass is:

$$f_i = \text{JumpReLU}_{\theta_i}(\pi_i) = \pi_i \cdot H(\pi_i - \theta_i)$$

- If $\pi_i > \theta_i$: feature fires with magnitude $\pi_i$ (identity)
- If $\pi_i < \theta_i$: feature is zeroed (killed)

---

## 3. The problem: zero gradient

For a single sample, $H(\pi_i - \theta_i)$ is a step function in $\theta_i$:
- Flat at $0$ when $\theta_i > \pi_i$ (feature dead)
- Flat at $1$ when $\theta_i < \pi_i$ (feature alive)
- Vertical jump at $\theta_i = \pi_i$

The derivative w.r.t. $\theta_i$ is zero everywhere (flat regions) and undefined at the jump. SGD cannot learn $\theta_i$ because it receives zero gradient.

Same problem for JumpReLU: flat ($0$) below $\theta_i$, flat-with-slope-$1$ (identity) above $\theta_i$. The derivative w.r.t. $\theta_i$ is zero almost everywhere.

---

## 4. The key insight: expected loss is smooth

One sample gives a step function. But the average over many samples with different pre-activation values is smooth.

Think of a single coin flip: result is $0$ or $1$ (discrete step). Average of $1000$ coin flips is a smooth number between $0$ and $1$ (continuous).

Each sample contributes a step at its own $\pi_i$ value. The average over all samples is smooth because steps are spread across different locations.

Formally, the expected loss is:

$$\mathbb{E}_x[L(x)] = \mathbb{E}_x\bigl[\|x - \hat{x}\|^2\bigr] + \lambda \sum_i \mathbb{E}_x\bigl[H(\pi_i(x) - \theta_i)\bigr]$$

The second term is the expected L0 sparsity penalty. For one feature $i$:

$$\mathbb{E}_x\bigl[H(\pi_i - \theta_i)\bigr] = \int_{-\infty}^{+\infty} H(z - \theta_i) \, p_i(z) \, dz$$

where $p_i(z)$ is the probability density of pre-activations for feature $i$ (how many samples have pre-activation near $z$, per unit $z$).

Because $H(z - \theta_i) = 1$ when $z > \theta_i$ and $0$ otherwise, the integral simplifies to:

$$\mathbb{E}_x\bigl[H(\pi_i - \theta_i)\bigr] = \int_{\theta_i}^{+\infty} p_i(z) \, dz$$

This is the survival function ($1$ minus CDF). It is smooth because $p_i(z)$ is smooth. The derivative exists:

$$\frac{d}{d\theta_i} \mathbb{E}_x\bigl[H(\pi_i - \theta_i)\bigr] = -p_i(\theta_i)$$

The derivative is negative because increasing $\theta_i$ makes the threshold stricter, so fewer features fire, so expected count decreases.

---

## 5. What happens when we increase $\theta_i$?

Two effects:

**Effect 1: Reconstruction gets worse.**
Some features that were helping reconstruction are now killed. Any sample with $\pi_i \in [\theta_i, \theta_i + d\theta]$ was alive, now dead. Reconstruction error increases.

**Effect 2: Sparsity improves.**
Each killed feature saves $\lambda$ from the L0 penalty.

Net change in expected loss:

$$\frac{d\,\mathbb{E}[L]}{d\theta_i} = (\text{reconstruction cost of losing one boundary feature} - \text{sparsity savings}) \times (\text{how many features are at the boundary})$$

---

## 6. Defining each term

### 6.1 Residual

$$\text{residual} = x - \hat{x}$$

This is the vector that the SAE failed to reconstruct. It tells us what is missing.

### 6.2 Decoder direction $d_i$

$$d_i = W_{\text{dec}}[i, :]$$

This is the $i$-th row of the decoder matrix. It is the direction in activation space that feature $i$ contributes to the reconstruction. When feature $i$ fires with magnitude $f_i$, it adds $f_i \, d_i$ to $\hat{x}$.

### 6.3 Reconstruction impact $I_i$

$$I_i(x) = 2 \theta_i \, (d_i \cdot \text{residual})$$

This measures: if we kill feature $i$ (raise $\theta_i$ past it), how much does reconstruction error blow up?

Breakdown:
- $d_i \cdot \text{residual}$: Does this feature's decoder direction point into the missing part of the reconstruction? If yes, killing it hurts. If orthogonal to residual, killing it does nothing.
- $\theta_i$: The feature was firing at magnitude $\theta_i$ (since it was right at the threshold). Bigger features hurt more to lose.
- $2$: Comes from derivative of squared error $(x - \hat{x})^2$. The derivative of $z^2$ is $2z$.

High $I_i$: this feature is valuable for reconstruction. Keep it alive.
Low $I_i$: this feature is useless. Kill it.


### 6.4 Probability density $p_i(\theta_i)$

Over the entire dataset, look at all pre-activation values for feature $i$. Plot a histogram. The height of the histogram bin at $\theta_i$ is $p_i(\theta_i)$.

This is NOT a probability (number between $0$ and $1$). It is a density: probability per unit $\theta_i$. It can be greater than $1$.

$p_i(\theta_i)$ answers: how many samples have pre-activation values near $\theta_i$? If histogram is tall at $\theta_i$, nudging threshold affects many samples. If flat/zero at $\theta_i$, nudging affects nothing.

### 6.5 Conditional expectation $\mathbb{E}[I_i \mid \pi_i = \theta_i]$

Look at ONLY the samples whose pre-activation is near $\theta_i$ (the cars at the toll bridge height limit). Average their $I_i$ values.

This answers: on average, how much does reconstruction suffer when we block one car at the boundary?

---

## 7. The true gradient formula

Putting the two effects together:

$$\frac{d\,\mathbb{E}[L]}{d\theta_i} = \bigl(\mathbb{E}[I_i(x) \mid \pi_i = \theta_i] - \lambda\bigr) \, p_i(\theta_i)$$

| Term | Meaning |
|------|---------|
| $\mathbb{E}[I_i \mid \pi_i = \theta_i]$ | Average reconstruction cost of losing one feature at the boundary |
| $\lambda$ | Sparsity savings per killed feature |
| $\mathbb{E}[I_i] - \lambda$ | Net value of the feature. If positive, feature helps more than it costs. If negative, feature costs more than it helps. |
| $p_i(\theta_i)$ | How many features are at the boundary (density of samples near $\theta_i$) |

Sign check:
- If $\mathbb{E}[I_i] > \lambda$: feature is worth more than its sparsity cost. Gradient is positive. SGD decreases $\theta_i$ (lower the bridge, let more through).
- If $\mathbb{E}[I_i] < \lambda$: feature costs more than it is worth. Gradient is negative. SGD increases $\theta_i$ (raise the bridge, block more).
- If $p_i(\theta_i) \approx 0$: no samples at boundary. Gradient is near zero. Nothing changes.

---

## 8. Why we cannot use this formula directly

We do not know $p_i(\theta_i)$ or $\mathbb{E}[I_i \mid \pi_i = \theta_i]$. We only have a batch of $N$ samples (e.g., $4096$). We need to estimate these quantities from the batch.

---

## 9. The estimate: rectangle window (KDE)

We cannot look at samples exactly at $\theta_i$ (there may be none). So we look at samples NEAR $\theta_i$, within a small window $[\theta_i - \varepsilon/2, \theta_i + \varepsilon/2]$.

The rectangle function is:

$$\text{rect}\!\left(\frac{z - \theta_i}{\varepsilon}\right) = \begin{cases} 1 & \text{if } |z - \theta_i| < \varepsilon/2 \\ 0 & \text{otherwise} \end{cases}$$

This is a gate. It asks: is this sample close enough to the threshold to matter?

Over a batch of $N$ samples, the estimate is:

$$\frac{1}{N \varepsilon} \sum_{\alpha=1}^{N} \bigl(I_i(x_\alpha) - \lambda\bigr) \, \text{rect}\!\left(\frac{\pi_i(x_\alpha) - \theta_i}{\varepsilon}\right)$$

**Why divide by $\varepsilon$:**
The window has width $\varepsilon$. We want to estimate per unit $\theta_i$, not per window. Dividing by $\varepsilon$ turns the count into a density (samples per unit $\theta_i$).

**Why this works:**
As $N \to \infty$ and $\varepsilon \to 0$ (carefully), this converges to the true formula $(\mathbb{E}[I_i] - \lambda) \, p_i(\theta_i)$.

---

## 10. The STE trick: making PyTorch compute this automatically

We want PyTorch's autograd to compute the estimate above when we call `.backward()`. But PyTorch sees the loss as:

$$L = \|x - \hat{x}\|^2 + \lambda \sum_i H(\pi_i - \theta_i)$$

The Heaviside and JumpReLU are step functions. Their true derivatives w.r.t. $\theta_i$ are zero. So we define FAKE derivatives (pseudo-derivatives) with specific coefficients so that the chain rule spits out the KDE estimate.

### 10.1 JumpReLU pseudo-derivative

$$\frac{\partial^{\text{STE}} \text{JumpReLU}}{\partial^{\text{STE}} \theta_i} = -\frac{\theta_i}{\varepsilon} \, \text{rect}\!\left(\frac{\pi_i - \theta_i}{\varepsilon}\right)$$

The coefficient $-\theta_i/\varepsilon$ is chosen so that when multiplied by the chain rule term from reconstruction, the $\theta_i$ cancels algebraically.

### 10.2 Heaviside pseudo-derivative

$$\frac{\partial^{\text{STE}} H}{\partial^{\text{STE}} \theta_i} = -\frac{1}{\varepsilon} \, \text{rect}\!\left(\frac{\pi_i - \theta_i}{\varepsilon}\right)$$

The coefficient $-1/\varepsilon$ is chosen so that the $\lambda$ term matches the KDE estimate.

### 10.3 Why these coefficients

The chain rule through reconstruction:

$$\frac{dL_{\text{recon}}}{d\theta_i} = \frac{dL_{\text{recon}}}{df_i} \cdot \frac{\partial^{\text{STE}} f_i}{\partial^{\text{STE}} \theta_i}$$

We know $\frac{dL_{\text{recon}}}{df_i} = -2 \, d_i \cdot \text{residual} = -\frac{I_i}{\theta_i}$.

Substitute the pseudo-derivative:

$$= \left(-\frac{I_i}{\theta_i}\right) \left(-\frac{\theta_i}{\varepsilon} \, \text{rect}\right) = \frac{I_i}{\varepsilon} \, \text{rect}$$

The $\theta_i$ cancels. This is not an accident. The coefficient $-\theta_i/\varepsilon$ was chosen specifically for this cancellation.

The chain rule through sparsity:

$$\frac{dL_{\text{sparse}}}{d\theta_i} = \lambda \, \frac{\partial^{\text{STE}} H}{\partial^{\text{STE}} \theta_i} = \lambda \left(-\frac{1}{\varepsilon} \, \text{rect}\right) = -\frac{\lambda}{\varepsilon} \, \text{rect}$$

Add both paths:

$$\frac{dL_{\text{total}}}{d\theta_i} = \frac{I_i - \lambda}{\varepsilon} \, \text{rect}\!\left(\frac{\pi_i - \theta_i}{\varepsilon}\right)$$

Average over the batch:

$$\frac{1}{N} \sum_{\alpha} \frac{I_i(x_\alpha) - \lambda}{\varepsilon} \, \text{rect}\!\left(\frac{\pi_i(x_\alpha) - \theta_i}{\varepsilon}\right)$$

This is exactly the KDE estimate from section 9.

---

## 11. Why the input gradient is zeroed

In the Heaviside backward pass, we set $\frac{dH}{d\pi_i} = 0$.

**Why:**
The sparsity loss should NOT affect $W_{\text{enc}}$. If we let the gradient flow through, $W_{\text{enc}}$ would try to make pre-activations smaller just to reduce the L0 count, even if it hurts reconstruction. We want $W_{\text{enc}}$ to chase reconstruction only. The sparsity tradeoff is handled exclusively by $\theta_i$.

In the JumpReLU backward, $\frac{d(\text{JumpReLU})}{d\pi_i} = 1$ (identity above $\theta_i$, $0$ below). This is the true gradient. Reconstruction flows normally to $W_{\text{enc}}$.

---

## 12. Full algorithm summary

For each feature $i$, for each sample in the batch:
1. Is $\pi_i$ within $\varepsilon/2$ of $\theta_i$? If no, contribution = $0$.
2. If yes, compute $\text{residual} = x - \hat{x}$.
3. Compute $I_i = 2 \theta_i \, (d_i \cdot \text{residual})$.
4. Subtract $\lambda$.
5. Average over the batch and divide by $\varepsilon$.
6. This is the gradient for $\theta_i$.

SGD then:
- If $I_i > \lambda$ on average: gradient positive $\rightarrow$ decrease $\theta_i$ (keep feature).
- If $I_i < \lambda$ on average: gradient negative $\rightarrow$ increase $\theta_i$ (kill feature).

$W_{\text{enc}}$ and $W_{\text{dec}}$ only see the reconstruction gradient. They learn to reconstruct. $\theta_i$ mediates the sparsity tradeoff.

---

## 13. PyTorch implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class JumpReLU_STE(torch.autograd.Function):
    """Forward: JumpReLU activation. Backward: STE for threshold gradient."""

    @staticmethod
    def forward(ctx, z, theta, eps=0.001):
        ctx.save_for_backward(z, theta)
        ctx.eps = eps
        mask = (z > theta).float()
        return z * mask

    @staticmethod
    def backward(ctx, grad_output):
        z, theta = ctx.saved_tensors
        eps = ctx.eps

        # Gradient to z: normal (identity above theta, 0 below)
        grad_z = (z > theta).float() * grad_output

        # Pseudo-gradient to theta: -(theta/eps) * rect window
        in_window = ((z - theta).abs() < eps / 2).float()
        grad_theta = -(theta / eps) * in_window * grad_output
        grad_theta = grad_theta.sum(dim=0)  # sum over batch

        return grad_z, grad_theta, None


class Heaviside_STE(torch.autograd.Function):
    """Forward: Heaviside step. Backward: STE for threshold, zero to input."""

    @staticmethod
    def forward(ctx, z, theta, eps=0.001):
        ctx.save_for_backward(z, theta)
        ctx.eps = eps
        return (z > theta).float()

    @staticmethod
    def backward(ctx, grad_output):
        z, theta = ctx.saved_tensors
        eps = ctx.eps

        # CRITICAL: zero gradient to input (sparsity blocked from W_enc)
        grad_z = torch.zeros_like(z)

        # Pseudo-gradient to theta: -(1/eps) * rect window
        in_window = ((z - theta).abs() < eps / 2).float()
        grad_theta = -(1.0 / eps) * in_window * grad_output
        grad_theta = grad_theta.sum(dim=0)  # sum over batch

        return grad_z, grad_theta, None


class JumpReLU_SAE(nn.Module):
    def __init__(self, d_in, d_sae, eps=0.001):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.eps = eps

        # Normal weights: trained by reconstruction only
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Threshold: trained by both reconstruction and sparsity
        self.log_theta = nn.Parameter(torch.ones(d_sae) * (-6.9))  # exp(-6.9) ~ 0.001

    def forward(self, x):
        # x: (B, d_in)
        x = x - self.b_dec  # pre-encoder bias

        # Encoder pre-activations
        pre = F.relu(x @ self.W_enc + self.b_enc)  # (B, d_sae)

        # Threshold (positive)
        theta = torch.exp(self.log_theta)  # (d_sae,)

        # JumpReLU: features
        f = JumpReLU_STE.apply(pre, theta, self.eps)  # (B, d_sae)

        # Decoder
        x_hat = f @ self.W_dec + self.b_dec  # (B, d_in)

        return x_hat, f, pre, theta


def jumprelu_loss(model, x, lambda_sparsity):
    x_hat, f, pre, theta = model(x)

    # Reconstruction loss: flows to W_enc, W_dec, b_enc, b_dec, theta
    recon_loss = ((x - x_hat) ** 2).sum(dim=-1).mean()

    # Sparsity loss: L0 penalty via Heaviside STE
    # Only affects theta (grad to pre is zeroed in Heaviside_STE)
    l0 = Heaviside_STE.apply(pre, theta, model.eps).sum(dim=-1)  # (B,)
    sparsity_loss = lambda_sparsity * l0.mean()

    return recon_loss + sparsity_loss, recon_loss.item(), sparsity_loss.item()
```

---

## 14. Analogy: The Toll Bridge

| Symbol | Bridge analogy |
|--------|---------------|
| $\theta_i$ | Bridge height limit |
| $\pi_i(x)$ | Car height for sample $x$ |
| $H(\pi_i - \theta_i)$ | Does the car pass? ($1$ = yes, $0$ = no) |
| $\lambda$ | Toll per car (sparsity penalty) |
| $I_i$ | Economic value of letting one car through (reconstruction benefit) |
| $p_i(\theta_i)$ | Number of cars per inch at exactly the height limit |
| $\varepsilon$ | Width of the toll booth window (how precisely we measure "exactly at the limit") |
| $W_{\text{enc}}, W_{\text{dec}}$ | Road engineers (build the road, don't care about tolls) |
| $\theta_i$ | Toll operator (adjusts height limit to balance traffic and revenue) |

The toll operator asks: for cars at the height limit, is the economic value of letting them through greater than the toll? If yes, lower the bridge. If no, raise it. The road engineers just build the best road they can.

