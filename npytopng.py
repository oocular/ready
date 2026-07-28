import pathlib
import numpy as np
from PIL import Image

npy_masks = pathlib.Path("datasets/archive/openEDS/openEDS/masks_npy")
masks_out = pathlib.Path("datasets/archive/openEDS/openEDS/masks")
masks_out.mkdir(exist_ok=True)

count = 0
for npy_file in sorted(npy_masks.iterdir()):
    if npy_file.suffix == ".npy":
        mask_array = np.load(npy_file)
        mask_array = np.squeeze(mask_array)
        mask_array = mask_array.astype(np.uint8)
        mask = Image.fromarray(mask_array)
        mask.save(masks_out / npy_file.with_suffix(".png").name)
        count += 1

print("Converted:", count, "masks")
