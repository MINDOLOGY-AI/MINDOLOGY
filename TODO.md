port the openmindfab CLI here. unify a serving function.  

- READ THE CODE

- Fix Up Labeling Prompt

SAE retrain, all 16 layers at 1B tokens (`evoke/OlMo2_1b/interp/sae_resid_topk.py`, already configured: 1B data split, lr decay, seed 21):  
- L8 trial done (`results/OlMo2_1b/sae_resid_topk_1bTok/compare_L8/`): recon 19.4% -> 18.3%, CE +3.5% -> +2.8%, rare tail 31% -> 26%. decide if worth it  
- L8 alone took 6.5h on weighty; try 8 SAEs per LM pass instead of 4 if it fits 32GB  
- speedups: bf16 SAE matmuls  
- then eval + picks + labels (~$85), then delete the 100M run (`sae_resid_topk`, `olmo2_1b_interp_dataset_100Mrun`)  

sparse TopK SAE (`synapse/probes/sae/TopKSAE.py`):  
- keep TopK output as `idx (N, 64)` + `vals (N, 64)` instead of a dense (N, 32768) vector with 64 nonzeros  
- decode = sum of the 64 winning decoder rows: `F.embedding_bag(idx, W_dec, per_sample_weights=vals, mode="sum")` instead of `f @ W_dec` (decoder fwd+bwd ~3.3 TFLOP/step -> ~0; AuxK's 512 dead picks the same way)  
- callers needing dense f (picks, density) scatter it back  
- test: loss + all grads match the dense version; benchmark steps/s on weighty (expect ~25-35% faster)  
- interpviz sends `idx` + `vals` per token to the UI  

