# calculate mean and std deviation

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# imageFilesDir = Path('/home/aslab/code/segmentation/data/PASCAL_VOC/VOC2012/JPEGImages/')
# imageFilesDir = Path('/home/aslab/code/yang_code/semantic_segmentation/data/Cityscapes/leftImg8bit/train/')
# imageFilesDir = Path('./data/voc/VOC2010/JPEGImages/')
# imageFilesDir = Path('./data/ADEChallengeData2016/images/training/')
imageFilesDir = Path('./data/mapillary/training/images/')
# files = list(imageFilesDir.rglob('*.png'))
files = list(imageFilesDir.rglob('*.jpg'))

# Since the std can't be calculated by simply finding it for each image and averaging like
# the mean can be, to get the std we first calculate the overall mean in a first run then
# run it again to get the std.

mean = np.array([0., 0., 0.])
stdTemp = np.array([0., 0., 0.])
std = np.array([0., 0., 0.])

numSamples = len(files)

for i in tqdm(range(numSamples)):
    im = cv2.imread(str(files[i]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.

    for j in range(3):
        mean[j] += np.mean(im[:, :, j])

mean = (mean / numSamples)

for i in tqdm(range(numSamples)):
    im = cv2.imread(str(files[i]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.
    for j in range(3):
        stdTemp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / (im.shape[0] * im.shape[1])

std = np.sqrt(stdTemp / numSamples)

print(mean.tolist())
print(std.tolist())

mean = mean * 255
std = std * 255

print(mean.tolist())
print(std.tolist())

# pascal context
# [0.4573452038706311, 0.4373267620507893, 0.4041014619999162]
# [0.27484295476790993, 0.2715177408131066, 0.2848014435958842]
# [116.62302698701093, 111.51832432295127, 103.04587280997863]
# [70.08495346581704, 69.23702390734218, 72.62436811695048]

# ade
# [0.4889702432110162, 0.4654837562508165, 0.42939523933218693]
# [0.2588368568690922, 0.2558111562941972, 0.27494394400665045]
# [124.68741201880913, 118.6983578439582, 109.49578602970767]
# [66.0033985016185, 65.23184485502028, 70.11070572169587]

# mapillary
# [0.4190161318279305, 0.45849286996723954, 0.4699152577823381]
# [0.2640499390934314, 0.27521748457328526, 0.3030281681501086]
# [106.84911361612228, 116.91568184164608, 119.82839073449622]
# [67.33273446882501, 70.18045856618774, 77.27218287827769]