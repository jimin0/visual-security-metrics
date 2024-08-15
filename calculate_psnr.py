import cv2
import numpy as np
import math


def psnr(img1, img2):
    mse = np.mean(
        (np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2
    )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / (np.sqrt(mse)))


def resize_to_reference(img1, img2):
    # 원본 이미지 크기로 리사이즈
    # 보간법 : INTER_AREA
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    return img2_resized


# 이미지 경로 설정
# plain_image_path = "/Users/jiminking/Documents/김지민/projects/myproject/gradu/data/Celebrity Faces Dataset/Angelina Jolie/002_8f8da10e.jpg"
# encrypted_image_path = "/Users/jiminking/Documents/김지민/projects/myproject/gradu/data/black_data/Angelina Jolie/002_8f8da10e.jpg.png"

plain_image_path = "/Users/jiminking/Documents/김지민/projects/myproject/gradu/data/PEID/refimg/tower.bmp"
encrypted_image_path = "/Users/jiminking/Documents/김지민/projects/myproject/gradu/data/PEID/encimg/tower_09_5.bmp"
# 이미지 로드
original = cv2.imread(plain_image_path)
contrast = cv2.imread(encrypted_image_path)

# 암호화된 이미지를 원본 이미지 크기로 리사이즈
contrast = resize_to_reference(original, contrast)

# PSNR 계산
psnr_value = psnr(original, contrast)
print(f"PSNR value is {psnr_value}")

# OpenCV 내장 함수로 PSNR 계산 (비교를 위해)
psnr_opencv = cv2.PSNR(original, contrast)
print(f"PSNR OpenCV value is {psnr_opencv}")
