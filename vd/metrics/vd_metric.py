import cv2
import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY
# import lpips


@METRIC_REGISTRY.register()
def calculate_vd_psnr(img, img2):
    # normalized_psnr = -10 * np.log10(np.mean(np.power(img - img2, 2)))
    img_np = (img * 255.0).round()
    img2_np = (img2 * 255.0).round()
    mse = np.mean((img_np - img2_np) ** 2)
    if mse == 0:
        return float('inf')
    # if normalized_psnr == 0:
    #     return float('inf')
    # return normalized_psnr
    return 20 * np.log10(255.0 / np.sqrt(mse))


@METRIC_REGISTRY.register()
def calculate_vd_ssim(img, img2):
    img_np = (img * 255.0).round()
    img2_np = (img2 * 255.0).round()
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img_np.astype(np.float64)
    img2 = img2_np.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

    # vd_ssim = compare_ssim(img, img2, multichannel=True)
    # return vd_ssim

