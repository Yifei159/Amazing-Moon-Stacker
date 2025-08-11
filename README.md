# Moon Stacker (Automatic Moon Alignment & Stacking)

Uses ECC alignment + median stacking + gentle unsharp mask to produce high-quality 16-bit TIF and 8-bit PNG outputs.

## Features
- **Automatic Alignment**: ECC (Affine / Translation) + lunar surface mask for robust registration
- **Noise Reduction**: Median stacking to suppress cloud streaks, hot pixels, and atmospheric turbulence
- **Lossless Workflow**: Full 16-bit processing with gentle unsharp mask enhancement
- **Command-line Options**: Choose between `affine` or `translation`, toggle CLAHE, and fine-tune iteration/threshold parameters

## Usage

To begin stacking, first put the photos you take into:

    Moon-Stacker/moon_photos

Install the packages (The test environment is Python 3.9):

    pip install -r requirements.txt

Then run:

    python -m moon_stacker.cli

