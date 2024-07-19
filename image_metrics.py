import numpy as np
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import vifp
from iibvsi import IIBVSI


class ImageMetrics:
    def __init__(self):
        self.iibvsi = IIBVSI()

    def calculate_psnr(self, original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return float("inf")
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def calculate_ssim(self, original, compressed):
        return ssim(original, compressed, data_range=original.max() - original.min())

    def calculate_vif(self, original, compressed):
        return vifp(original, compressed)

    def calculate_iibvsi(self, original, compressed):
        return self.iibvsi.compute_iibvsi(original, compressed)

    def calculate_all_metrics(self, original, compressed):
        return {
            "PSNR": self.calculate_psnr(original, compressed),
            "SSIM": self.calculate_ssim(original, compressed),
            "VIF": self.calculate_vif(original, compressed),
            "IIBVSI": self.calculate_iibvsi(original, compressed),
        }
