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


# 3. Similarity Measurement


def compute_contrast_similarity_map(C_P, C_E, R1=1e-12):
    """
    수식 10: Spatial Contrast Similarity Map 계산
    - param C_P: 원본 이미지 GM 평균 맵
    - param C_E: 암호화된 이미지의 GM 평균 맵
    - param R1: 분모 0 방지를 위한 상수
    - return: Spatial Contrast Similarity Map
    """
    S_C = (2 * C_P * C_E + R1) / (C_P**2 + C_E**2 + R1)
    return S_C


def compute_texture_similarity_map(T_P, T_E, R2=1e-12):
    """
    수식 11: Texture Similarity Map 계산
    - param T_P: 원본 이미지 Texture map
    - param T_E: 암호화된 이미지  Texture map
    - param R2: 분모 0 방지를 위한 상수
    - return: Texture Similarity Map
    """
    S_T = (2 * T_P * T_E + R2) / (T_P**2 + T_E**2 + R2)
    return S_T


def compute_visual_security_map(S_C, S_T):
    """
    수식 12: Visual Security Map 계산
    - param S_C: Spatial Contrast Similarity Map
    - param S_T: Texture Similarity Map
    - return: Visual Security Map
    """

    alpha = np.mean(S_C)  # sc 평균값
    beta = np.mean(S_T)  # st 평균값
    S_VS = (S_C**alpha) * (S_T**beta)
    return S_VS


# 3. Image Importance-Based Pooling
def compute_weighting_map(C_P, C_E):
    """
    수식 13: Weighting Map 계산
    - param C_P: 원본 이미지 GM 평균 맵
    - param C_E: 암호화된 이미지의 GM 평균 맵
    - return: Weighting Map
    """
    W = np.maximum(np.abs(C_P), np.abs(C_E))
    return W


def compute_visual_security_score(W, S_VS):
    """
    수식 14: Visual Security Score 계산
    - param W: Weighting Map
    - param S_VS: Visual Security Map
    - return: Visual Security Score
    """
    VS = np.sum(W * S_VS) / np.sum(W)
    return VS


def downsample_image(img, t):  # T를 위한
    """
    이미지 다운샘플링
    - param img: 입력 이미지
    - param t: 다운샘플링 횟수
    - return: 다운샘플링된 이미지
    """
    return img[:: 2**t, :: 2**t]


def compute_iibvsi(plain_img, encrypted_img, T=2, sigma=1.0, M=3, S=4, O=8):
    """
    수식 15: IIBVSI 계산
    - param plain_img: 원본 이미지
    - param encrypted_img: 암호화된 이미지
    - param T: 다중 해상도 레벨 수
    - param sigma: 가우시안 필터 스케일 파라미터
    - param M: GM 맵의 최대 차수
    - param S: 스케일 수
    - param O: 방향 수
    - return: IIBVSI Score
    """
    VS_scores = []

    for t in range(T + 1):
        # Downsample images
        plain_img_ds = downsample_image(plain_img, t)
        encrypted_img_ds = downsample_image(encrypted_img, t)

        # Compute the gradient magnitudes and spatial contrast maps
        G_P = compute_gradient_magnitude(plain_img_ds, sigma)
        C_P = compute_spatial_contrast_map(plain_img_ds, sigma, M)
        G_E = compute_gradient_magnitude(encrypted_img_ds, sigma)
        C_E = compute_spatial_contrast_map(encrypted_img_ds, sigma, M)

        # Compute the texture maps
        T_P = compute_texture_map(plain_img_ds, S, O)
        T_E = compute_texture_map(encrypted_img_ds, S, O)

        # Compute the similarity maps
        S_C = compute_contrast_similarity_map(C_P, C_E)
        S_T = compute_texture_similarity_map(T_P, T_E)
        S_VS = compute_visual_security_map(S_C, S_T)

        # Compute the weighting map and visual security score
        W = compute_weighting_map(C_P, C_E)
        VS = compute_visual_security_score(W, S_VS)
        VS_scores.append(VS)

    # Compute the final IIBVSI score
    IIBVSI_score = np.mean(VS_scores)
    return IIBVSI_score


"""
시각화 
"""
# 이미지 로드
plain_image_path = "/Users/jiminking/Documents/김지민/projects/myproject/gradu/data/PEID/refimg/tower.bmp"
encrypted_image_path = "/Users/jiminking/Documents/김지민/projects/myproject/gradu/data/PEID/encimg/tower_10_5.bmp"

img_plain = cv2.imread(plain_image_path, cv2.IMREAD_GRAYSCALE)
img_encrypted = cv2.imread(encrypted_image_path, cv2.IMREAD_GRAYSCALE)

if img_plain is None:
    raise FileNotFoundError("Plain image not found")
if img_encrypted is None:
    raise FileNotFoundError("Encrypted image not found")

# IIBVSI 계산
iibvsi_score = compute_iibvsi(img_plain, img_encrypted)
print("IIBVSI Score:", iibvsi_score)

# 파라미터 설정
sigma = 1.0
M = 3  # GM 맵의 최대 차수
S = 4  # 스케일 수
O = 8  # 방향 수

# Compute the gradient magnitudes and spatial contrast maps
G_P = compute_gradient_magnitude(img_plain, sigma)  # GM
C_P = compute_spatial_contrast_map(img_plain, sigma, M)  # GM 평균
G_E = compute_gradient_magnitude(img_encrypted, sigma)  # GM
C_E = compute_spatial_contrast_map(img_encrypted, sigma, M)  # GM 평균

# Compute the texture maps
texture_map_combined_plain = compute_texture_map(img_plain, S, O)
texture_map_combined_encrypted = compute_texture_map(img_encrypted, S, O)

# Compute the similarity maps
S_C = compute_contrast_similarity_map(C_P, C_E)
S_T = compute_texture_similarity_map(
    texture_map_combined_plain, texture_map_combined_encrypted
)
S_VS = compute_visual_security_map(S_C, S_T)

# print("Visual Security Map S_VS")
# print(S_VS)

# 시각화
plt.figure(figsize=(20, 10))

plt.subplot(2, 3, 1)
plt.title("Plain Image")
plt.imshow(img_plain, cmap="gray")
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
plt.title("Encrypted Image")
plt.imshow(img_encrypted, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Contrast Similarity Map S_C")
plt.imshow(S_C, cmap="gray")
plt.colorbar()
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("Texture Similarity Map S_T")
plt.imshow(S_T, cmap="gray")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()
