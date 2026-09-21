# low sample continual earning  

being able to learn information from low data like in context learning but in weights without catastrphoic forgetting is the most important capability gap between llms and humans  

(by the way, context rot is fundementally the same problem, if model can learn important context into weights it won't forget information evreytime it compresses)  

if the llm has sth in it's context it didnt know before, from a simple fact to a complex skill, when the context is wiped, it will forget it.  

for example if i tell it some user preference like "i like eating apples" or some coding project information it has no way to remember it  


this seems weird, cause llms can solve advanced open math problems but they cannot permanently remember simple facts from a single sentence without significantly destroying their own abilities (catastrphoic forgetting)  

i will try to solve this, why? first obviously main line quest primy will need it, but honestly to just get good at ml and hopefully get some results and get hired at a company i like , mainly anthropic  

here are some steps:  

1: get llama 3 4b working in synapse  
2: bvnch. normal benchmarks (for catatrophic forgetting measurement)  
3: make my own structured continual learning benchmarks  
4: implement frontier research in this area, understand and bench all of them  
5: understand how the brain works generally and continuously learns (neuroscience week, put in the interp_journey too)  
6: apply mech interp methods to see how they ICLs
7: try my own ideas, try to publish a paper  

# some thoughts and directions  

both parts of the problems are individually solved but not together  
low sample is solved with in context learning  
continual learning is sort of solved with just general pretraining  
but together it's not solved? 

"low sample continual learning: making llms sleep" is kinda a great title  



# workflows  

we can build a symbolic workflow.  

seperating each data into [new_knowledge] [question] [correct answer].  
we can let the model self generate different questions and correct answers based on the new_knowledge.  
then let the model self test [question] without [new_knowledge] to see if it even needs to learn this or it already knows.  
if it doesn't, we do our continual learning methods and retest.  

# Dual-Mask Gradient Updates for Continual Learning

Catastrophic forgetting occurs when neural networks lose previously learned knowledge upon learning new tasks. We propose a simple method that combines two complementary masking strategies to select which parameters to update.

## Method

Standard EWC penalizes changes to important parameters but still updates everything. Top-k gradient sparsity limits updates but doesn't know which parameters matter for old tasks.

We take the intersection: only update parameters that are **both** high-gradient for the new task **and** low-Fisher for old tasks.

```
mask = (fisher < θ_fisher) & topk(|∇L|, k)
∇L_masked = ∇L * mask
```

This explicitly routes new learning through parameters that won't damage old knowledge.

## Why It Works

| Method | What it does | Problem |
|--------|--------------|---------|
| EWC | Soft penalty on important params | Still updates everything |
| Top-k | Sparse updates | Doesn't know what to protect |
| **Ours** | Sparse updates avoiding important params | — |

## Results

TODO

## Usage

```bash
python train.py --method dual_mask --dataset split_cifar100 --k 0.1 --fisher_threshold 0.5
```

## Citation

```
TODO
```

