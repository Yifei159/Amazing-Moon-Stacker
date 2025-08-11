from dataclasses import dataclass

@dataclass
class Config:
    # path
    input_dir: str = "moon_photos"      # raw moon photo (support jpg/jpeg/png/tif)
    output_dir: str = "moon_output"     # output dir

    # align
    warp_mode: str = "affine"           # "affine" or "translation"
    ecc_max_iters: int = 300
    ecc_eps: float = 1e-7
    resize_for_speed: float = 1.0       # <1 will downscale for speed
    use_clahe: bool = True              # Apply CLAHE before alignment for more stability

    # Post-processing
    unsharp_amount: float = 0.5         # Gentle unsharp mask strength (0~1)
    gauss_sigma: float = 1.2            # Gaussian radius for unsharp mask