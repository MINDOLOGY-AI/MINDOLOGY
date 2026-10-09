port the openmindfab CLI here. unify a serving function.  

- READ THE CODE

- Fix Up Labeling Prompt

Qwen3.5-4B SAEs on weighty (`evoke/Qwen3_5_4b/interp/sae_resid_topk.py`, 8 full-attention layers x 500M tokens):  
- `git pull`, rename the text dir: `mv data/datasteps/txt/olmo2_1b_interp_dataset data/datasteps/txt/interp_dataset`  
- tokenize: `python -m datasteps.interp_dataset.tokenize_interp_dataset` (~1.2B tokens)  
- short trial first: peak memory + steps/s for groups of 4 at 4096 tokens/step (the limited decoder's speedup shows here too)  
- full run (~a day), then picks + labels (8 x 40960 = 327,680 features, ~$52)  

SAE search (`TopKSAE.py`, L8 short runs vs the current recipe on the same eval):  
- AuxK stronger + earlier: dead threshold 10M tokens -> lower (e.g. 1M / 250k) so rare-but-alive features get aux gradient too; aux_coeff 1/32 -> higher  
- near-miss gradients: forward keeps top 64, extra loss on the top 256 reconstruction so features that nearly fired get pulled toward the residual (cf. Gao et al. Multi-TopK: L(k) + L(4k)/8)  

