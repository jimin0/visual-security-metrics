import numpy as np
import cv2
from iibvsi import IIBVSI

# 테스트용 이미지 로드 
# plain_img = cv2.imread('path_to_plain_image.png', cv2.IMREAD_GRAYSCALE)
# encrypted_img = cv2.imread('path_to_encrypted_image.png', cv2.IMREAD_GRAYSCALE)

np.random.seed(0)
plain_img = np.random.rand(512, 512)  # 원본 이미지
encrypted_img = np.random.rand(512, 512)  # 암호화된 이미지

print("start")

# 이미지가 제대로 로드되었는지 확인
if plain_img is None or encrypted_img is None:
    raise ValueError("이미지를 로드할 수 없습니다.")

# 이미지 크기가 같은지 확인
if plain_img.shape != encrypted_img.shape:
    raise ValueError("원본 이미지와 암호화된 이미지의 크기가 다릅니다.")

iibvsi = IIBVSI()
score = iibvsi.compute_iibvsi(plain_img, encrypted_img)

print(f"IIBVSI 점수: {score}")

# 결과 시각화 (선택사항)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(plain_img, cmap='gray')
plt.title('원본 이미지')
plt.subplot(132)
plt.imshow(encrypted_img, cmap='gray')
plt.title('암호화된 이미지')
plt.subplot(133)
plt.text(0.5, 0.5, f'IIBVSI: {score:.4f}', ha='center', va='center', fontsize=20)
plt.axis('off')
plt.tight_layout()
plt.show()