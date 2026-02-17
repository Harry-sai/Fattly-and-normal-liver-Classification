import cv2
import numpy as np
import os

input_folder = "data/images/fatty_liver"          # your original images folder
output_folder = "data2/images/fatty_liver"    # where normalized images will be saved

os.makedirs(output_folder, exist_ok=True)

def normalize_and_save(input_path, output_path):
    # load grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # normalize to [0,1]
    min_val = img.min()
    max_val = img.max()
    norm = (img - min_val) / (max_val - min_val + 1e-8)

    # convert back to 8-bit PNG (0–255) for saving
    norm_uint8 = (norm * 255).astype(np.uint8)

    cv2.imwrite(output_path, norm_uint8)

# process all PNGs
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        inp = os.path.join(input_folder, filename)
        out = os.path.join(output_folder, filename)
        normalize_and_save(inp, out)
        print("Saved:", out)
