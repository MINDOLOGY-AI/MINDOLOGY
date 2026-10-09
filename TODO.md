port the openmindfab CLI here. unify a serving function.  

- READ THE CODE

- Fix Up Labeling Prompt

SAE retrain, all 16 layers at 1B tokens (`evoke/OlMo2_1b/interp/sae_resid_topk.py`, already configured: 1B data split, lr decay, seed 21):  
- L8 trial done (`results/OlMo2_1b/sae_resid_topk_1bTok/compare_L8/`): recon 19.4% -> 18.3%, CE +3.5% -> +2.8%, rare tail 31% -> 26%. decide if worth it  
- L8 alone took 6.5h on weighty; try 8 SAEs per LM pass instead of 4 if it fits 32GB  
- speedups: bf16 SAE matmuls  
- then eval + picks + labels (~$85), then delete the 100M run (`sae_resid_topk`, `olmo2_1b_interp_dataset_100Mrun`)  

limited decoder (`TopKSAE.py`, done, matches the dense version to float rounding): benchmark steps/s on weighty before the 1B retrain (expect ~25-35% faster)  

SAE search (`TopKSAE.py`, L8 short runs vs the current recipe on the same eval):  
- AuxK stronger + earlier: dead threshold 10M tokens -> lower (e.g. 1M / 250k) so rare-but-alive features get aux gradient too; aux_coeff 1/32 -> higher  
- near-miss gradients: forward keeps top 64, extra loss on the top 256 reconstruction so features that nearly fired get pulled toward the residual (cf. Gao et al. Multi-TopK: L(k) + L(4k)/8)  

