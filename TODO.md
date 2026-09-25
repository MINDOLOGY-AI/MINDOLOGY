port the openmindfab CLI here. unify a serving function.  

SAE retrain "properly" (`evoke/OlMo2_1b/interp/sae_resid_topk.py`):  
- re-split interp dataset: ~1.0B train / ~140M eval (now 799M / 341M; same shuffled mix, only 100M of train was used)  
- TRAIN_TOKENS 100M -> 1B (current L8 expl_var still rising at 100M: 0.741 @50M -> 0.750 @100M)  
- add LR decay: constant, then linear to 0 over the last 20% of steps (simple_train has no scheduler now)  
- try 8 SAEs per LM pass instead of 4 if it fits 32GB (~50h -> ~35-40h on weighty)  
- then rerun eval + picks + relabel (~$85, ~3h), compare density rare tail vs current (31% of features < ideal/10, Gemma Scope 10%), then delete the 100M run  

