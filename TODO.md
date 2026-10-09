port the openmindfab CLI here. unify a serving function.  

- READ THE CODE

- Fix Up Labeling Prompt

Qwen3.5-4B SAEs, full run on a rented H100 80GB (`evoke/Qwen3_5_4b/interp/sae_resid_topk.py`, all 32 layers, one epoch ~1.5B tokens, groups of 8, ~55h):  
- runpod H100 SXM pod: 150GB volume disk at /workspace (peak use ~100GB) + ~50GB container disk, ssh enabled  
- clone the repo, `pip install -r requirements.txt`, `python -m evoke.Qwen3_5_4b.run.download`  
- copy HF_TOKEN from the laptop's ~/.bashrc to the pod's (account codespaceDrifter, already accepted starcoderdata's gate)  
- check `lucadiliello/bookcorpusopen` rows are natural-text books (field `text`) before downloading  
- `python -m datasteps.interp_dataset.download_all`, then `python -m datasteps.interp_dataset.tokenize_interp_dataset` (2B tokens: train / test / eval)  
- `python -m evoke.Qwen3_5_4b.interp.sae_resid_topk gpu` in tmux; first check peak memory + steps/s  
- copy back weights + results + `eval.bin`, then `... sae_resid_topk label` on the laptop (1.31M features, ~$210)  
- next experiment against this baseline: bf16 SAE matmuls (one layer)  

SAE search (`TopKSAE.py`, L8 short runs vs the current recipe on the same eval):  
- AuxK stronger + earlier: dead threshold 10M tokens -> lower (e.g. 1M / 250k) so rare-but-alive features get aux gradient too; aux_coeff 1/32 -> higher  
- near-miss gradients: forward keeps top 64, extra loss on the top 256 reconstruction so features that nearly fired get pulled toward the residual (cf. Gao et al. Multi-TopK: L(k) + L(4k)/8)  

