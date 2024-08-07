import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift


# 1. Feature Extraction


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
    - param M : GM 맵의 최대 차수  - 논문에서 1~8까지 사용했고, M=3이 best였다함.(기본값)
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


def log_gabor_filter(shape, sigma_s, mu_o, sigma_o):
    """
    수식 6: Log-Gabor 필터 생성
    """
    rows, cols = shape
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, cols), np.linspace(-0.5, 0.5, rows))
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    r_s = 0.5  # 중심 주파수
    r[r == 0] = np.finfo(float).eps  # 로그 계산에서 0을 피하기 위해 작은 값을 추가
    log_gabor_radial = np.exp(
        -((np.log(r / r_s)) ** 2) / (2 * (np.log(sigma_s / r_s)) ** 2)
    )
    log_gabor_angular = np.exp(-((theta - mu_o) ** 2) / (2 * sigma_o**2))
    log_gabor = log_gabor_radial * log_gabor_angular

    return log_gabor


def apply_log_gabor(img, sigma_s, mu_o, sigma_o):
    """
    수식 7: Log-Gabor 필터 적용 및 amplitude_map 계산
    """
    log_gabor = log_gabor_filter(img.shape, sigma_s, mu_o, sigma_o)
    img_fft = fftshift(fft2(img))
    filtered_img = img_fft * log_gabor
    filtered_img_ifft = ifft2(ifftshift(filtered_img))

    amplitude_map = np.abs(filtered_img_ifft)
    return amplitude_map


def apply_log_gabor_filter_only(img, sigma_s, mu_o, sigma_o):
    """
    Log-Gabor 필터만 적용한 결과를 반환합니다.
    """
    log_gabor = log_gabor_filter(img.shape, sigma_s, mu_o, sigma_o)
    img_fft = fftshift(fft2(img))
    filtered_img = img_fft * log_gabor
    filtered_img_ifft = ifft2(ifftshift(filtered_img))
    return np.real(filtered_img_ifft)


def compute_texture_map(img, S, O):
    """
    수식 8, 9: 텍스처 맵 계산
    """
    texture_map = np.zeros_like(img, dtype=float)

    for s in range(S):
        for o in range(O):
            sigma_s = 0.35 + 0.1 * s  # 예시 값, 필요에 따라 조정
            mu_o = o * np.pi / O
            sigma_o = 0.5  # 예시 값, 필요에 따라 조정
            amplitude_map = apply_log_gabor(img, sigma_s, mu_o, sigma_o)
            texture_map += amplitude_map

    texture_map /= S * O
    return texture_map


"""
시각화 
"""
# 이미지 로드
image_path = "/Users/jiminking/Documents/김지민/projects/myproject/gradu/data/black_data/Angelina Jolie/001_fe3347c0.jpg.png"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found")

# 파라미터 설정
sigma = 1.0
M = 3  # GM 맵의 최대 차수
S = 4  # 스케일 수
O = 8  # 방향 수

G_P = compute_gradient_magnitude(img, sigma)  # GM
C_P = compute_spatial_contrast_map(img, sigma, M)  # GM 평균

# log gabor 만
log_gabor_filtered = apply_log_gabor_filter_only(img, 0.6, np.pi / 4, 0.6)

# 최종 texture feature map
texture_map_combined = compute_texture_map(img, S, O)

# 시각화
plt.figure(figsize=(20, 10))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Gradient Magnitude G_P")
plt.imshow(G_P, cmap="gray")
plt.colorbar()
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Spatial Contrast Map C_P")
plt.imshow(C_P, cmap="gray")
plt.colorbar()
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Log-Gabor Filtered")
plt.imshow(log_gabor_filtered, cmap="gray")
plt.colorbar()
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("Texture Map (amplitude map)")
plt.imshow(texture_map_combined, cmap="gray")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()
