import cv2
import numpy as np

def to_gray(img16: np.ndarray, use_clahe: bool) -> np.ndarray:
    if img16.ndim == 3:
        gray = cv2.cvtColor(img16, cv2.COLOR_BGR2GRAY)
    else:
        gray = img16
    gray = gray.astype(np.float32) / 65535.0
    if use_clahe:
        g8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g8 = clahe.apply(g8)
        gray = g8.astype(np.float32) / 255.0
    return gray

def detect_moon_mask(gray_ref: np.ndarray) -> np.ndarray:
    g = (gray_ref * 255).astype(np.uint8)
    thr = max(200, int(np.percentile(g, 98)))
    _, bw = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bw)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)
        r = max(5, int(0.02 * max(mask.shape)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask = cv2.dilate(mask, kernel)
    else:
        mask[:] = 255
    return mask

def _warp_mode_from_str(mode: str) -> int:
    mode = (mode or "").strip().lower()
    if mode in ("translation", "translate"):
        return cv2.MOTION_TRANSLATION
    return cv2.MOTION_AFFINE  # default to affine

def align_to_ref(img: np.ndarray,
                 ref_gray: np.ndarray,
                 mask_ref: np.ndarray,
                 use_clahe: bool,
                 resize_for_speed: float,
                 warp_mode_str: str,
                 ecc_max_iters: int,
                 ecc_eps: float) -> np.ndarray:
    h, w = ref_gray.shape
    # Match dimensions
    if resize_for_speed != 1.0:
        target_size = (int(w), int(h))
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    else:
        img_resized = img.copy()

    gray = to_gray(img_resized, use_clahe)
    warp_mode = _warp_mode_from_str(warp_mode_str)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(ecc_max_iters), float(ecc_eps))

    try:
        # Newer API: last positional argument is gaussFiltSize
        _, warp_matrix = cv2.findTransformECC(ref_gray, gray, warp_matrix, warp_mode, criteria, mask_ref, 5)
    except TypeError:
        # Older API: no gaussFiltSize
        _, warp_matrix = cv2.findTransformECC(ref_gray, gray, warp_matrix, warp_mode, criteria, mask_ref)
    except cv2.error as e:
        print(f"[WARN] ECC alignment failed, falling back to phase correlation. Reason: {e}")
        shift = cv2.phaseCorrelate(np.float32(ref_gray), np.float32(gray))[0]
        warp_matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)

    aligned = cv2.warpAffine(
        img_resized, warp_matrix, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT
    )
    return aligned