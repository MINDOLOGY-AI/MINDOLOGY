MLP Feature Map
interpret what MLP does by passing input activation feature combinations through and seeing output feature combinations   

assuming activation interpretability is somewhat solved with WCC or SAE, we want to actually interpret the weights cause the model is fundementally the weights.  

we simply pass the decoded input features in to the original MLP (do not change it in anyway, keep the nonlinearity) and get the output, decompose the output into features what the output layer's WCC or SAE, and see the top k activation values of output features  
we can then try different combinations of input features, like activating 2 or 3 input features and add their decoded activation and see the output  
we can also see if different ratios, like activation value 2 for feature A and 1 for feature B, and how that changes the output feature activations  

weight pertubation experiment:  
viewing MLP as a feature mapping table, we will modify the weights to change one specific input feature output feature relationship, say from Apple -> red to Apple -> blue, and see if that changes what the model outputs.  
this is very cool because it should work across prompts. and it actually changes the model rather than for example an external SAE attached to the model  

we might have to change multiple relationships to get an effect. for example first zero out all the apple -> red related features then turn up all the apple -> blue  related features.  
we would also have to target all mlp across layers that stores this relationship not just a single MLP. this is a wrong assumption attention V or O doesn't store information. but hopefully MLP effects can dominate enough to change outputs.  

to actually change the weights, we have to retrain that specific MLP layer. we need to preserve as much as possible of all other relationships and try to specifically target the relationships we want to change.  
so the loss should be based on prediction error of a range of original input feature combinations as input and edited output feature combinations as output.   
note that all pairings are in the data even unedited ones that have nothing to do with the mapping we want to change. note that this should have many ranges of activation values like 1.0, 1.1, 1.2 rather than just 0,1,2 like for out visual table.  

# problems and future directions
one problem is a mapping could be cross mlp layers rather than in a single layer. for example layer A could be apple -> some sense of color. and  then layer B be Apple + some sense of color -> some warm color. and layer C be Apple + some arm color -> red.  note that the residue stream could impact this because each input in the addition of all previous mlp and attention outputs. note that in more complicated mappings this could be even more multilayer.  
but for  now we keep both the interpretability and pertubation single layer  

i don't know whether to sae the residue or the mlp_out directly

also the MLP is structually similar to an SAE we should just try to interpret MLP neuron wise first.  

# concrete plans  
train and label SAE on all layers of the pre norm residue before mlp and the mlp  output. so this is like the residue stream vs what the mlp adds to the residue stream.  
make MLP feature map from these. for each feature we try activation value 1 and gather top 5 output features and activation values. we try all combinations of 2 at activation value 2  and gather top 5 output features and activation values  
we handpick some examples to try to pertubate it by finding all related layers with mappings and changing the values we want. (we test a perplexity on normal training data before and after to make sure we didn't break the model) then we give related handpicked prompts to see if pertubation worked  

