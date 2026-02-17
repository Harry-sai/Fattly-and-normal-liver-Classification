import torch

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.2f} GB")

