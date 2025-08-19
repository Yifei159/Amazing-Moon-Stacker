import cv2
import numpy as np

def median_stack(imgs):
    stack = np.stack(imgs, axis=0).astype(np.uint16)
    med = np.median(stack, axis=0).astype(np.uint16)
    return med

def unsharp_mask(img16, amount=0.5, sigma=1.2):
    img = img16.astype(np.float32)
    blur = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    sharp = np.clip(sharp, 0, 65535).astype(np.uint16)
    return sharp
# latest update: updated alignment logic 2025-08-20 00:28:14 AEST
