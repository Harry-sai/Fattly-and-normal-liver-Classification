# simple mask cleanup script

import numpy as np
import cv2
from scipy import ndimage as ndi
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# paths
INPUT_MASK_ROOT  = "data/predictions_more_imgs/normal"          # current masks
OUTPUT_MASK_ROOT = "Check/normal"    # cleaned masks
MASK_EXTENSIONS  = [".png", ".jpg", ".jpeg"]

# main cleanup step
def fix_liver_mask(mask: np.ndarray) -> np.ndarray:
    """
    Fix liver mask by:
    1. Enforcing binary
    2. Filling internal holes (vessels)
    3. Keeping largest connected component

    Returns uint8 mask with values {0, 255}
    """
    # 1. make it binary
    mask = (mask > 0).astype(np.uint8)

    # 2. fill holes
    mask_filled = ndi.binary_fill_holes(mask).astype(np.uint8)

    # 3. keep biggest part
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_filled, connectivity=8
    )

    if num_labels <= 1:
        return mask_filled * 255

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = (labels == largest_label).astype(np.uint8)

    return clean_mask * 255


# process all masks
def process_all_masks(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mask_files = [
        p for p in input_root.rglob("*")
        if p.suffix.lower() in MASK_EXTENSIONS
    ]

    print(f"Found {len(mask_files)} masks to process")

    for mask_path in tqdm(mask_files):
        rel_path = mask_path.relative_to(input_root)
        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        mask = np.array(Image.open(mask_path).convert("L"))
        fixed_mask = fix_liver_mask(mask)

        Image.fromarray(fixed_mask).save(out_path)


# run
if __name__ == "__main__":
    process_all_masks(
        input_root=INPUT_MASK_ROOT,
        output_root=OUTPUT_MASK_ROOT
    )
