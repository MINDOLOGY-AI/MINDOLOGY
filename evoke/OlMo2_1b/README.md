# interp  
this is the model we seek to understand completely!  
we will try every interp method, from anthropic and original ones, on this model. and see if the features they find are corresponding.  

## sae
first run our 4 SAEs on a layer 8 residue layer to compare them.  
normal, topK, jumpRELU, slabCone  

or honestly also run:  
slab, slab with set 30 degree cone, slab with set 90 degree cone  
since we are writing the slabconeSAE paper i really want to make sure whats good whats not.  

pick the best one!  

## interp data mix  

we want 1b tokens which is on the lower end since this is a small model   

the dataset to gather activations should 1: reflect original training corpus 2: cover all concepts we try to find 3: not over bias any conceptual area   

we try to keep sae dataset english only because olmo is mostly trained on english  


we choose to the mix:  

- DCLM baseline (web)  
30%  

- starcoder  
20%.  
(python 4% C++ 4% html 4% css 4% js 4%)  

- wikipedia  
10%   

- OpenWebMATH  
real world math blogs and qna  
15%  

- arxiv  
academic papers  
togethercomputer/RedPajama-Data-1T (the one olmo uses)  
10%  

- allenai/tulu-3-sft-mixture  
15%, posttraining talk like a bot.      
