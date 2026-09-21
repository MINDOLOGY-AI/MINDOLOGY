# we can just pass features into an MLP and see what comes out.  
# for nonlinearity regions: let's use the like i.e. that feature's encoder row's direction times it's average activation (we need to measure this) since that's the supposedly region this feature fires most often. and we just freeze the relu and pass in the decoder column which is the feature  
# we then see the activation that comes out is mostly aligned with what feature. maybe we just use the same feature like basis?  
