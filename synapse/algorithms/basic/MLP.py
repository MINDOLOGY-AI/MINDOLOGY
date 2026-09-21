import torch
import torch.nn as nn

# linear with relu. last layer no relu.  
# arbiatry hidden dims (don't all have to be the same)  
class MLP (nn.Module):
    # pass dims in from input to output
    def __init__(self, dims: tuple[int, ...]):
        super().__init__()
        layers = []
        dims_len = len(dims)
        assert dims_len>=2
        for i in range (dims_len - 2):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
            ])
        # last layer no relu
        layers.append(nn.Linear(dims[dims_len-2], dims[dims_len-1]))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.blocks(x)
        return x
