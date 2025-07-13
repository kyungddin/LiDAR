import torch
checkpoint = torch.load("/home/nsl/PointPillars/pretrained/epoch_160.pth", map_location="cpu")
print(type(checkpoint))
if isinstance(checkpoint, dict):
    print(checkpoint.keys())
else:
    print("checkpoint is not a dict, it's probably a state_dict directly.")