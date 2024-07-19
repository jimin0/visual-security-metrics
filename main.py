import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_metrics import ImageMetrics

plain_img_path = "/Users/jiminking/Documents/김지민/projects/myproject/gradu/data/IVC-SelectEncrypt/cimg6013.pgm"
enc_img_path = "/Users/jiminking/Documents/김지민/projects/myproject/gradu/data/IVC-SelectEncrypt/cimg6013-021-trad.pgm"

# 테스트용 이미지 로드
plain_img = cv2.imread(plain_img_path, cv2.IMREAD_GRAYSCALE)
encrypted_img = cv2.imread(enc_img_path, cv2.IMREAD_GRAYSCALE)

print("start")

if plain_img is None or encrypted_img is None:
    raise ValueError("이미지를 로드할 수 없습니다.")

if plain_img.shape != encrypted_img.shape:
    raise ValueError("원본 이미지와 암호화된 이미지의 크기가 다릅니다.")

# 이미지 데이터 타입을 float32로 변환
plain_img = plain_img.astype(np.float32)
encrypted_img = encrypted_img.astype(np.float32)

plain_img = plain_img / 255.0
encrypted_img = encrypted_img / 255.0

# 매트릭 계ㅅ산~
metrics = ImageMetrics()
results = metrics.calculate_all_metrics(plain_img, encrypted_img)

# 결과 출력
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# 결과 시각화
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(plain_img, cmap="gray")
plt.title("Original Image")
plt.subplot(132)
plt.imshow(encrypted_img, cmap="gray")
plt.title("Encrypted Image")
plt.subplot(133)
plt.axis("off")
plt.text(
    0.5,
    0.5,
    "\n".join([f"{k}: {v:.4f}" for k, v in results.items()]),
    ha="center",
    va="center",
    fontsize=12,
)
plt.title("Metrics")
plt.tight_layout()
plt.show()
