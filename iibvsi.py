import numpy as np
import cv2
import math
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt

class IIBVSI:
    def __init__(self, T=2, M=3, S=3, O=8, R1=1e-12, R2=1e-12, sigma=1, sigma_s=1, mu_o=0, sigma_o=1):        
        self.T = T  # 다운 샘플링 횟수 (논문 = 2)
        self.M = M  # high order GM 차수  (논문 M=3)
        self.S = S  # 공간 스케일
        self.O = O  # orientation index
        self.R1 = R1  # 분모 0 안되게 하기 위한 상수
        self.R2 = R2  # 분모 0 안되게 하기 위한 상수
        self.sigma = sigma  # 가우시안 필터의 표준 편차
        self.sigma_s = sigma_s  # 로그-가보 필터의 공간 스케일
        self.mu_o = mu_o  # 로그-가보 필터의 중심 주파수
        self.sigma_o = sigma_o  # 로그-가보 필터의 방향 표준 편차

    def compute_gradient_magnitude(self, img, sigma):
        """
        논문 수식 (1) (2) : 그냥 이미지 gm, 암호화 이미지 gm 
        """
        gx = gaussian_filter(img, sigma=sigma, order=[1, 0])
        gy = gaussian_filter(img, sigma=sigma, order=[0, 1])
        gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
        
        #print(f"Gradient Magnitude is {gradient_magnitude}")
        
        return gradient_magnitude

    def gaussian_partial_derivative(self, sigma):
        """
        논문 수식 (3) : 가우시안 편미분 필터
        """
        # 필터 크기 
        size = int(2 * np.ceil(3 * sigma) + 1)
        x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1),
                           np.arange(-size // 2 + 1, size // 2 + 1))
        
        # 수식 상수부분 : -1 / (2 * pi * sigma^4)
        constant = -1 / (2 * np.pi * sigma ** 4)
        hx = constant * x * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) # x 방향 필터
        hy = constant * y * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) # y 방향 필터
        
        return hx, hy
        
    def compute_high_order_gm(self, img, M, sigma):
        """
        논문 수식 (4) : compute high order gradient magnitude
        """
        gm_maps = []
        current_map = img
        for _ in range(M):
            current_map = self.compute_gradient_magnitude(current_map, sigma)
            gm_maps.append(current_map)
        return np.mean(gm_maps, axis=0)
    
    def log_gabor_filter(self, shape, sigma_s, mu_o, sigma_o):
        """
        논문 수식 (6) : log gabor filter 생성!
        """
        rows, cols = shape
        
        x, y = np.meshgrid(np.linspace(-0.5, 0.5, cols), np.linspace(-0.5, 0.5, rows))
        r = np.sqrt(x ** 2 + y ** 2)  # 반경 거리 
        theta = np.arctan2(y, x)  # 각도 

        # 중심 주파수 r_s 설정
        r_s = 0.5

        # 6번 수식
        log_gabor_front = np.exp(-((np.log(r / r_s)) ** 2) / (2 * (np.log(sigma_s / r_s)) ** 2))
        log_gabor_back = np.exp(-((theta - mu_o) ** 2) / (2 * sigma_o ** 2))
        log_gabor = log_gabor_front * log_gabor_back

        return log_gabor
                            
    def apply_log_gabor(self, img, sigma_s, mu_o, sigma_o):
        """
        논문 수식 (7) : apply the log gabor filter 
        """
        log_gabor = self.log_gabor_filter(img.shape, sigma_s, mu_o, sigma_o)
        img_fft = fft2(img)  # 푸리에 변환
        filtered_img = img_fft * log_gabor  # 필터 적용
        filtered_img_ifft = ifft2(filtered_img)  # 역 푸리에 변환
        
        # 7번 수식
        amplitude_map = np.sqrt(np.real(filtered_img_ifft) ** 2 + np.imag(filtered_img_ifft) ** 2)  
        return amplitude_map
        
    def generate_texture_map(self, img, S, O, sigma_s, sigma_o):
        """
        논문 수식 (8),(9) : generate texture map or P(image) & E(image)
        """
        texture_map = np.zeros_like(img)
        for s in range(1, S + 1):
            for o in range(1, O + 1):
                mu_o = o * np.pi / O  # 방향 인덱스 계산
                texture_map += self.apply_log_gabor(img, sigma_s * s, mu_o, sigma_o)
        texture_map /= (S * O)
        return texture_map

    
    def compute_contrast_similarity_map(self, contrast_plain, contrast_encrypted):
        """
        논문 수식 (10) : compute spatial contrast similarity map of P(image) & E(image)
        """
        R1 = self.R1 # postive constant 분모 0  방지 
        similarity_map = (2 * contrast_plain * contrast_encrypted + R1) / (contrast_plain ** 2 + contrast_encrypted ** 2 + R1)
        return similarity_map

    def compute_texture_similarity_map(self, texture_plain, texture_encrypted):
        """
        논문 수식 (11) : compute texture similarity map of P(image) & E(image)
        """
        R2 = self.R2 # r1과 마찬가지로 postive constant 분모 0  방지 
        similarity_map = (2 * texture_plain * texture_encrypted + R2) / (texture_plain ** 2 + texture_encrypted ** 2 + R2)
        return similarity_map
    
    def compute_visual_security_map(self, contrast_similarity_map, texture_similarity_map):
        """
        논문 수식 (12) : compute visual security map
        alpha : sc 평균값, beta : st 평균값
        """
        alpha = np.mean(contrast_similarity_map)
        beta = np.mean(texture_similarity_map)
        
        visual_security_map = (contrast_similarity_map ** alpha) * (texture_similarity_map ** beta)
        return visual_security_map
    
    def compute_weighting_map(self, contrast_plain, contrast_encrypted):
        """
        논문 수식 (13) :  The maximum values of these two spatial contrast maps are utilized as the weighting map
        """
        weighting_map = np.maximum(np.abs(contrast_plain), np.abs(contrast_encrypted))
        return weighting_map
    
    def compute_vs(self, weighting_map, visual_security_map):
        """
        논문 수식 (14) : the visual security score
        """
        vs_score = np.sum(weighting_map * visual_security_map) / np.sum(weighting_map)
        return vs_score
    
    def downsample_image(self, img, t):
        """
        이미지 다운샘플링
        """
        return img[::2**t, ::2**t]
    

    def compute_iibvsi(self, plain_img, encrypted_img):
        """
        논문 수식 (15) : IIBVSI 최종 계산
        plain_img: 원본 이미지
        encrypted_img: 암호화된 이미지
        return: IIBVSI 점수
        """
        vs_scores = []

        for t in range(self.T + 1):
            plain_img_ds = self.downsample_image(plain_img, t)
            encrypted_img_ds = self.downsample_image(encrypted_img, t)
            contrast_plain = self.compute_high_order_gm(plain_img_ds, self.M, self.sigma)
            contrast_encrypted = self.compute_high_order_gm(encrypted_img_ds, self.M, self.sigma)
            contrast_similarity = self.compute_contrast_similarity_map(contrast_plain, contrast_encrypted)
            texture_plain = self.generate_texture_map(plain_img_ds, self.S, self.O, self.sigma_s, self.sigma_o)
            texture_encrypted = self.generate_texture_map(encrypted_img_ds, self.S, self.O, self.sigma_s, self.sigma_o)
            texture_similarity = self.compute_texture_similarity_map(texture_plain, texture_encrypted)
            visual_security_map = self.compute_visual_security_map(contrast_similarity, texture_similarity)
            weighting_map = self.compute_weighting_map(contrast_plain, contrast_encrypted)

            vs_score = self.compute_vs(weighting_map, visual_security_map)
            vs_scores.append(vs_score)

        # 최종 IIBVSI 점수 계산 (수식 15)
        iibvsi_score = np.mean(vs_scores)
        return iibvsi_score


