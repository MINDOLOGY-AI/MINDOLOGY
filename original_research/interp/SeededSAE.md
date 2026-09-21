since the purpose of the SAE is to unconver concepts the models are actually using.  
we should see what the model is even naturally using to seed some SAE decoder directions.  

for example we pass the beginning word embeddings into the MLPs, and for each output of the MLP it's a actual direction that the model uses.  

we pass each MLP output through the following attention OV circuit and that's another for that layer.  
we just propogate throguh the model like this and for each layer we will get (dictionary_size) number of seeded features.  

maybe this can be a way to initialize SAE features rather than a SAE itself.  
