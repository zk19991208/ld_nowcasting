import torch
from torch import nn

a = torch.randn(7, 1, 16, 16)
b = torch.randn(7, 1, 16, 16)

x = torch.cat([a, b], dim=1)

pixel_unshuffle = nn.PixelUnshuffle(4)
pixel_shuffle = nn.PixelShuffle(4)

y = pixel_unshuffle(x)

c = y[:, :16, ...]

z = pixel_shuffle(y)

d = pixel_shuffle(c)
