# slab cone SAE

## idea  

each feature fires only when the input passes THREE hard gates at once:  
- slab lower gate: pre-activation `pi` must be ABOVE a learnable threshold  
- slab upper gate: pre-activation `pi` must be BELOW a learnable threshold  
- cone gate: angle between input and encoder direction must be BELOW a learnable threshold  

the slab forces magnitude specialization (a feature owns a band of activation strengths, not "everything above threshold") and the cone forces direction specialization (a feature owns a cone around its encoder row). a feature's receptive field is the intersection: a truncated cone segment.  

`features = pi * g_lower * g_upper * g_cone`  

## backprop (JumpReLU's calibrated STE, generalized to any gate)  

every gate is `gate = H(±(s - theta))`, `s` = the gate's score (`pi` for the slabs, angle for the cone), `theta` = its learnable threshold. the hard gate has zero true derivative wrt theta, so a straight-through estimator whose batch average equals the gradient of the EXPECTED loss:  

`grad_theta = sign * (1/eps) * rect((s - theta)/eps) * grad_output`  

- `sign = -1` for LOWER bounds: raising theta kills boundary features (recon worse, sparsity better)  
- `sign = +1` for UPPER bounds: raising theta revives boundary features (recon better, sparsity worse)  
- `eps` = rect window width (KDE bandwidth); only samples within `eps/2` of the threshold contribute  

`grad_score = None` → BLOCKED, ALWAYS. neither recon nor the sparsity penalty may move the encoder through a gate. if sparsity could flow into the scores, the encoder would dodge the penalty by shrinking pre-activations, inflating them past the upper slab, or rotating directions out of the cone. recon reaches the encoder ONLY through the explicit magnitude path (`features = pi * gates`). the encoder learns to reconstruct; the thetas alone handle the sparsity tradeoff.  

**why ±1/eps and not JumpReLU's -theta/eps:** in JumpReLU the STE node IS the magnitude product (`f = z * H`), so `grad_output` arrives without the magnitude and `-theta/eps` is needed to cancel the `1/theta` hidden in the recon chain rule. here each gate is a SEPARATE multiplicative node, so the product rule makes `grad_output` already carry `pi` AND the other gates' values; the correct coefficient is just ±1/eps. this is also what makes the cone gate work: a cone-boundary sample's firing magnitude is its own `pi` (decoupled from the angle), and the product structure supplies it automatically.  

code: `synapse/probes/sae/SlabConeSAE.py`, deleted 2026-09-24 (last version in commit `1c1497a`).  

# experiment look.  

seems like 0.1 set sparsity just pushes all activations to 0  

```
SAE               recon_pct     active 
jumprelu               15.6      579.5      
slabcone              100.0        0.0      
topk k_64              33.5       64.0      
topk k_96              29.1       96.0      
topk k_128             26.3      128.0
```
