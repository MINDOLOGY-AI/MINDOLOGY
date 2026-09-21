a lot of paper implementations are going to be here, maybe their algorithms won't be that good, it is fine don't delete them. primy is the selection not this.   


# to do 

# save saving


# evals

# async loading  

useful for large data like images not really useful for text  

```
# Async GPU prefetch pattern
# Overlaps CPU→GPU transfer with GPU compute
# Only worth it if GPU compute dominates, not IO-bound

device = torch.device("cuda")
loader = DataLoader(dataset, pin_memory=True, num_workers=4)

# Pre-fetch first batch, sync before use
data_iter = iter(loader)
x, y = next(data_iter)
x = x.to(device, non_blocking=True)
y = y.to(device, non_blocking=True)
torch.cuda.synchronize()

for batch in loader:
    # GPU trains current while next transfers async in background
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    
    # Start async transfer of NEXT batch
    x, y = next(data_iter)
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    # No sync here — next iteration's compute waits implicitly

# KEY: pin_memory=True + non_blocking=True + separate CUDA stream
# Workers prefetch to CPU RAM, main loop async copies to GPU VRAM
# num_workers=0 kills the overlap, always use >0
```
# complex train. schedulers.  

# curriculum train

