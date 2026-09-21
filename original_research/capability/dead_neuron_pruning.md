over parameterization helps model escape local minima (more coordinates to not get stuck and move parameter space to another place). however after training maybe a lot of neurons are dead. barely firing.  

therefore we can prune them by counting firing frequencies and straight up deleting them, making the model smaller, changing the connections accordingly of connected neurons.  

this reduces size of the model while quantization reduces the size of each synapse.  

this is also very useful for SAEs.  
