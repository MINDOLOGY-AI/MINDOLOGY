# mind   
All our research code will be here!  

synapse for algorithms, training, probes, interp, and peripherals.  
datasteps for downloading datasets, splitting into train,test,val, tokenizing, remixing  
evoke is where you tie them together and write the experiment glue code  

data is for storing datasets  
weight is for storing weights. make sure both follow the data strcuture as the scripts they are made from.  
experiment is where you put all your new algorithms / training code. import from synapse but do not merge into it.   
results is where you write a concise .md results in it's README defined format.  

all theories are using the thoughtgraph program, a seperate system  

# code quality must be high  

all code here are absolutely sacred. well written, read and understood. in contrast to maybe some other vibecoded projects where i might only skim the code. all code should be written with reusability in mind.    

# goal: single model focused interp  

cross validate all interpretablity methods on a single model i.e. OlMo2_1b.  

build a database of labeling for each part of the model across generality from different methods.  
i.e. labeling MLP neurons, labeling a activation direction magntidue pair  

one major problem in current mech interp research is you try different methods, SAE, CLT, NLA, activation patching, etc. but you never compare them. you never build towards anything. in this single model focused thing, we are building towards the complete understanding of ONE model with all these methods.  

we will build a querieble vector database of like say we click an activation, we see it's direct labelling via k++, we see it's SAE decomposition and labeling, etc.  

all visually routed through a general 
