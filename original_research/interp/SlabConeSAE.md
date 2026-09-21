# slab cone SAE

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
