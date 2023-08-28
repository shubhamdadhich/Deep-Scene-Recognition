import sys
sys.path.append('..')

import torch
from torch import nn

from proj6_unit_tests.test_base import verify
from proj6_unit_tests.test_stats_helper import test_mean_and_variance
from proj6_unit_tests.test_dl_utils import test_compute_loss
from image_loader import ImageLoader
from data_transforms import get_data_augmentation_transforms
from proj6_code.stats_helper import compute_mean_and_std
import PIL
import numpy as np

import pdb;pdb.set_trace()

def tensor_to_image(tensor):
    tensor = (tensor + min(tensor))
    tensor = (tensor/max(tensor))*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Example of target with class indices
# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(input, target)
# output.backward()
# Example of target with class probabilities
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5).softmax(dim=1)
# output = loss(input, target)
# output.backward()

# print(verify(test_compute_loss))

dataset = ImageLoader('../data/')

mean, std = compute_mean_and_std('../data')

img = dataset[0]

inp_size = (64, 64)
transform = get_data_augmentation_transforms(inp_size, mean, std)

tImg = transform(img[0])

new_img = tensor_to_image(tImg)