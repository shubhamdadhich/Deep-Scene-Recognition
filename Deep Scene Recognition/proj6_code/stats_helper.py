import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler
from image_loader import ImageLoader

def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then in [0,1] before computing mean
  and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################

  scalar = StandardScaler()

  dataloader = ImageLoader(dir_name)
  l = len(dataloader)
  # import pdb;pdb.set_trace()
  for i in range(len(dataloader)):
    img = dataloader[i]
    imgArr = np.array(img[0]).flatten()
    if max(imgArr) > 1:
      normImgArr = imgArr/255.0
    else:
      normImgArr = imgArr
    normImgArr = normImgArr.reshape(-1, 1)
    scalar.partial_fit(normImgArr)

  mean = scalar.mean_
  std = scalar.scale_
  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
