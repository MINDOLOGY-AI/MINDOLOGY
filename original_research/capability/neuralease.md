# neuralease  

we input the transformer output embedding directly into next token input without decoding it into a token  

this makes "true continuous thinking" scale with time step. previously at each token output it is forced to be discretized. meaning each transformer layer can never see it's own ouputs or later layer's outputs in contious form.  

# related papers  

meta Coconut  

