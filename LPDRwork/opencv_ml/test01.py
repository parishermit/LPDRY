from pathlib import Path
import os

import skimage.io as io
import skimage.color as color
import skimage.morphology as morphology
import skimage.feature as feature
import skimage.measure as measure
from PIL import Image
import cv2 as cv

from matplotlib import pyplot as plt


# img = Image.open('../plate1.jpg')  # 打开图片
#
# img = img.convert("RGB")  # 4通道转化为rgb三通道
# img.save('../images/1.png')

img = io.imread('../plate1.jpg')

io.imshow(img)
# 转化为灰度
img2 = color.rgb2gray(img)
io.imshow(img2)
# Canny边缘检测并膨胀
img3 = feature.canny(img2, sigma=3)
io.imshow(img3)
img4 = morphology.dilation(img3)
io.imshow(img4)

label_img = measure.label(img4)
regions = measure.regionprops(label_img)
fig, ax = plt.subplots()
ax.imshow(img, cmap=plt.cm.gray)


# 标记+筛选
def in_bboxes(bbox, bboxes):
    for bb in bboxes:
        minr0, minc0, maxr0, maxc0 = bb
        minr1, minc1, maxr1, maxc1 = bbox
        if minr1 >= minr0 and maxr1 <= maxr0 and minc1 >= minc0 and maxc1 <= maxc0:
            return True
    return False


bboxes = []
for props in regions:
    y0, x0 = props.centroid
    minr, minc, maxr, maxc = props.bbox

    if maxc - minc > img4.shape[1] / 7 or maxr - minr < img4.shape[0] / 3:
        continue

    bbox = [minr, minc, maxr, maxc]
    if in_bboxes(bbox, bboxes):
        continue

    if abs(y0 - img4.shape[0] / 2) > img4.shape[0] / 4:
        continue

    bboxes.append(bbox)

    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-r', linewidth=2)

# 提取单个字符
bboxes = sorted(bboxes, key=lambda x: x[1])
chars = []
for bbox in bboxes:
    minr, minc, maxr, maxc = bbox
    ch = img2[minr:maxr, minc:maxc]
    chars.append(ch)

    io.imshow(ch)
    plt.show()

    str1 = "../images"
    # print(str1)
    path = str1 + os.sep + str(len(os.listdir(str1))) + '.jpg'
    ch *= 255
    cv.imwrite(path, ch)




