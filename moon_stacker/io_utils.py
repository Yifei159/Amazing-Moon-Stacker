import os
import cv2
import numpy as np

VALID_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".PNG", ".TIF", ".TIFF")

def ensure_dirs(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

def list_images(input_dir: str):
    if not os.path.isdir(input_dir):
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    files = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if f.endswith(VALID_EXTS)
    ]
    if not files:
        raise RuntimeError(f"No images found in {input_dir} (supported: {', '.join(VALID_EXTS)})")
    return files

def read_images(input_dir: str):
    files = list_images(input_dir)
    imgs = []
    for p in files:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Cannot read: {p}, skipped.")
            continue
        # Convert to 16-bit BGR
        if img.dtype != np.uint16:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (cv2.normalize(img.astype(np.float32), None, 0, 65535, cv2.NORM_MINMAX)).astype(np.uint16)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs.append(img)
    if len(imgs) < 2:
        raise RuntimeError("At least 2 images are required for stacking.")
    return imgs

def save_outputs(img16, out_dir: str):
    ensure_dirs(out_dir)
    tif_path = os.path.join(out_dir, "stacked_16bit.tif")
    png_path = os.path.join(out_dir, "stacked_8bit.png")
    ok1 = cv2.imwrite(tif_path, img16)
    img8 = (np.clip(img16.astype(np.float32) / 257.0, 0, 255)).astype(np.uint8)
    ok2 = cv2.imwrite(png_path, img8)
    if not (ok1 and ok2):
        raise RuntimeError("Failed to save outputs.")
    print(f"[OK] Output saved:\n - {png_path}\n - {tif_path}")