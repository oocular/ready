from pathlib import Path
import shutil
from PIL import Image


data_root = Path("datasets/s-natural/s-natural")

images_out = data_root / "images"
masks_out = data_root / "masks"

images_out.mkdir(exist_ok=True)
masks_out.mkdir(exist_ok=True)

image_count = 0
mask_count = 0
skipped_count = 0

for folder in sorted(data_root.iterdir()):
    if folder.is_dir() and folder.name.isdigit():
        synthetic_folder = folder / "synthetic"
        mask_folder = folder / "mask-withoutskin-noglasses"

        if not synthetic_folder.exists():
            print("No synthetic folder found:", folder.name)
            continue

        if not mask_folder.exists():
            print("No mask folder found:", folder.name)
            continue

        for image_file in sorted(synthetic_folder.iterdir()):
            if image_file.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                mask_file = mask_folder / image_file.name

                if not mask_file.exists():
                    print("No matching mask for:", image_file.name)
                    skipped_count += 1
                    continue

                image_size = Image.open(image_file).size
                mask_size = Image.open(mask_file).size

                if image_size != (640, 480):
                    print("Wrong image size:", image_file.name, image_size)
                    skipped_count += 1
                    continue

                if mask_size != (640, 480):
                    print("Wrong mask size:", mask_file.name, mask_size)
                    skipped_count += 1
                    continue

                new_name = f"{folder.name}_{image_file.name}"

                shutil.copy(image_file, images_out / new_name)
                shutil.copy(mask_file, masks_out / new_name)

                image_count += 1
                mask_count += 1

print("Images copied:", image_count)
print("Masks copied:", mask_count)
print("Skipped:", skipped_count)