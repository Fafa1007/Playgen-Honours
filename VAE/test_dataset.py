from load_mario_dataset import MarioFramesDataset

ds = MarioFramesDataset(
    "/Users/phillipliu/Documents/UCT/Honours/Thesis/Code/capture_output",
    image_size=128,
    return_actions=True
)
img, act = ds[0]
print(img.shape, img.min().item(), img.max().item())
print(type(act), act)
