import cv2
import numpy as np
from scipy import signal


def compute_gradient_magnitude(img, sigma):
    """
    수식 1, 2: 원본 이미지와 암호화된 이미지의 gradient magnitude 계산
    """
    hx, hy = gaussian_partial_derivative_filter(sigma)

    gx = signal.convolve2d(img, hx, mode="same", boundary="symm")
    gy = signal.convolve2d(img, hy, mode="same", boundary="symm")

    G = np.sqrt(gx**2 + gy**2)
    return G


def gaussian_partial_derivative_filter(sigma, size=None):
    """
    수식 3: 가우시안 편미분 필터 생성
    """
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)

    x, y = np.meshgrid(
        np.arange(-size // 2 + 1, size // 2 + 1),
        np.arange(-size // 2 + 1, size // 2 + 1),
    )

    constant = -1 / (2 * np.pi * sigma**4)
    exp_part = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    hx = constant * x * exp_part
    hy = constant * y * exp_part

    return hx, hy


def compute_spatial_contrast_map(img, sigma, M):
    """
    수식 4, 5 : GM 평균 계산
    - param img : 입력 이미지
    - param simga : 가우시안 필터 스케일 파라미터
    - param M : GM 맵의 최대 차수  - 논문에서 1~8까지 사용했고, M=3이 best였다함.
    => retrun C : spatial contrast map
    """
    G_maps = []
    current_map = img

    for _ in range(M):
        current_map = compute_gradient_magnitude(current_map, sigma)
        G_maps.append(current_map)

    # GM map 평균
    C = np.mean(G_maps, axis=0)
    return C


# 이미지 로드
image_path = "/Users/jiminking/Documents/김지민/projects/myproject/gradu/data/black_data/Angelina Jolie/001_fe3347c0.jpg.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found")


sigma = 1.0
M = 3 # papaer parameter settings : M = 3 기본값

G_P = compute_gradient_magnitude(img, sigma)
print("========================")
C_P = compute_spatial_contrast_map(img, sigma, M)


print(G_P)
