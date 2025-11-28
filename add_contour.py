import os
import cv2
import numpy as np
import random
from tqdm import tqdm
random.seed(1000)

source_img_floder = "/Users/qianyuhang/Projects/Datasets/LAKERED_DATASET/validation/images"
mask_img_floder = "/Users/qianyuhang/Projects/Datasets/LAKERED_DATASET/validation/masks"

img_output_path = "/Users/qianyuhang/Projects/Datasets/LAKERED_DATASET/validation/images_annotated"
contour_output_path = "/Users/qianyuhang/Projects/Datasets/LAKERED_DATASET/validation/contours"

#Set outline transparency
alpha = 0.3


# create the output folder (if not exist)
if not os.path.exists(img_output_path):
    os.makedirs(img_output_path, exist_ok=True)
if not os.path.exists(contour_output_path):
    os.makedirs(contour_output_path, exist_ok=True)
source_img_list = os.listdir(source_img_floder)

for source_img_name in tqdm(source_img_list):
    source_img_path = source_img_floder + '/' + source_img_name
    mask_img_name = source_img_name.split('.')[0] + ".png"
    mask_img_path = mask_img_floder + '/' + mask_img_name

    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
    bgr_img = cv2.imread(source_img_path)

    #print(mask_img.shape)
    # cv2.imshow("input img", mask_img)
    # cv2.waitKey(1000)

    reversed_img = 255 - mask_img
    # cv2.imshow("rev img", reversed_img)
    # cv2.waitKey(1000)
    # shut down all opencv windows

    kernel = np.ones((7, 7), np.uint8)

    #dilate img
    dilated = cv2.dilate(reversed_img, kernel, iterations=1)

    #erode img
    eroded= cv2.erode(reversed_img, kernel, iterations=1)

    contour = dilated - eroded
    # cv2.imshow("contour", contour)
    # cv2.waitKey(1000)

    colored_contour = np.zeros_like(bgr_img)
    # generate random HSV color with high saturation and brightness
    # H: Hue (0-179)
    # S: Saturation (set minimum as 180 to ensure high saturation)
    # V: Brightness (set minimum as 200 to ensure high brightness)
    h = random.randint(0, 179)
    s = random.randint(180, 255)
    v = random.randint(180, 255)

    #hsv to bgr
    hsv_color = np.uint8([[[h, s, v]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    #print(f"Random color: BGR={bgr_color}, HSV=({h}, {s}, {v})")

    #replace white color with random color
    colored_contour[contour > 0] = bgr_color

    # operate semi-transparency fusing only in the contour location
    contour_mask = contour > 0

    # conduct semi-transparency fusing using numpy
    bgr_img_float = bgr_img.astype(float)
    colored_contour_float = colored_contour.astype(float)

    # apply alpha blending: result = alpha * foreground + (1-alpha) * background
    bgr_img_float[contour_mask] = alpha * colored_contour_float[contour_mask] + (1-alpha) * bgr_img_float[contour_mask] 
    # convert back to uint8
    bgr_img = np.clip(bgr_img_float, 0, 255).astype(np.uint8)

    # cv2.imshow("colored contour", colored_contour)
    # cv2.waitKey(1000)
    # cv2.imshow("contoured_img", bgr_img)
    # cv2.waitKey(0)


    bgr_img_name = source_img_name
    contour_name = mask_img_name

    cv2.imwrite(contour_output_path + '/' + contour_name, contour)
    cv2.imwrite(img_output_path + '/' + bgr_img_name, bgr_img)

# cv2.destroyAllWindows()