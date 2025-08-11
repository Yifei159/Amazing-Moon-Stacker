import argparse
import os
import cv2
from .config import Config
from .io_utils import read_images, save_outputs, ensure_dirs
from .align import to_gray, detect_moon_mask, align_to_ref
from .stack import median_stack, unsharp_mask

def build_parser():
    p = argparse.ArgumentParser(
        prog="moon_stacker",
        description="Moon auto-alignment and stacking (ECC + median)"
    )
    p.add_argument("--input-dir", default=Config.input_dir, help="Input directory (default: moon_photos)")
    p.add_argument("--output-dir", default=Config.output_dir, help="Output directory (default: moon_output)")
    p.add_argument("--warp-mode", default=Config.warp_mode, choices=["affine", "translation"], help="Alignment model (default: affine)")
    p.add_argument("--ecc-max-iters", type=int, default=Config.ecc_max_iters, help="Max iterations for ECC (default: 300)")
    p.add_argument("--ecc-eps", type=float, default=Config.ecc_eps, help="Convergence threshold for ECC (default: 1e-7)")
    p.add_argument("--resize", type=float, default=Config.resize_for_speed, help="Resize factor for alignment speed-up (default: 1.0)")
    p.add_argument("--no-clahe", action="store_true", help="Disable CLAHE contrast enhancement")
    p.add_argument("--unsharp-amount", type=float, default=Config.unsharp_amount, help="Unsharp mask strength 0~1 (default: 0.5)")
    p.add_argument("--gauss-sigma", type=float, default=Config.gauss_sigma, help="Gaussian radius for unsharp mask (default: 1.2)")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)

    cfg = Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        warp_mode=args.warp_mode,
        ecc_max_iters=args.ecc_max_iters,
        ecc_eps=args.ecc_eps,
        resize_for_speed=args.resize,
        use_clahe=not args.no_clahe,
        unsharp_amount=args.unsharp_amount,
        gauss_sigma=args.gauss_sigma,
    )

    print(f"[INFO] Input directory: {cfg.input_dir}")
    print(f"[INFO] Output directory: {cfg.output_dir}")
    ensure_dirs(cfg.output_dir)

    imgs = read_images(cfg.input_dir)

    # Select middle frame as reference
    ref_idx = len(imgs) // 2
    ref = imgs[ref_idx]

    # Reference grayscale and mask
    if cfg.resize_for_speed != 1.0:
        ref = cv2.resize(ref, (int(ref.shape[1]), int(ref.shape[0])), interpolation=cv2.INTER_AREA)
    ref_gray = to_gray(ref, cfg.use_clahe)
    mask_ref = detect_moon_mask(ref_gray)

    # Alignment
    aligned_list = []
    for i, img in enumerate(imgs):
        print(f"[INFO] Aligning image {i+1}/{len(imgs)} …")
        aligned = align_to_ref(
            img=img,
            ref_gray=ref_gray,
            mask_ref=mask_ref,
            use_clahe=cfg.use_clahe,
            resize_for_speed=cfg.resize_for_speed,
            warp_mode_str=cfg.warp_mode,
            ecc_max_iters=cfg.ecc_max_iters,
            ecc_eps=cfg.ecc_eps,
        )
        aligned_list.append(aligned)

    print("[INFO] Performing median stacking …")
    stacked = median_stack(aligned_list)

    print("[INFO] Applying gentle unsharp mask and saving output …")
    stacked = unsharp_mask(stacked, amount=cfg.unsharp_amount, sigma=cfg.gauss_sigma)

    save_outputs(stacked, cfg.output_dir)
    print("[DONE] Processing completed!")

if __name__ == "__main__":
    main()