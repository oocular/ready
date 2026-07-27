"""
augment_images: resize and augment image and mask pairs for segmentation training.
"""

from pathlib import Path
from PIL import Image
import argparse

from ready.preprocessing.utils import find_pairs
from ready.preprocessing.geometric import apply as geo_apply
from ready.preprocessing.transforms import apply as transforms_apply


def main():
    parser = argparse.ArgumentParser(description="Resize and augment image and mask pairs.")

    # paths
    parser.add_argument("--image-dir",  required=True, help="folder containing input images")
    parser.add_argument("--mask-dir",   required=True, help="folder containing input masks")
    parser.add_argument("--output-dir", required=True, help="folder to save outputs")

    # resizing
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--limit",  type=int, default=None, help="only process this many pairs")

    # geometric augmentations - applied to both the image and the mask
    for name in ["hflip", "vflip","rot45", "rot90", "rot180", "rot270"]:
        parser.add_argument(f"--{name}", action="store_true")

    # photometric augmentations - image only, with a controllable value
    for name, default in [("brightness-up", 1.5), ("brightness-down", 0.6),
                          ("contrast-up", 1.5),   ("blur", 2.0)]:
        parser.add_argument(f"--{name}",        action="store_true")
        parser.add_argument(f"--{name}-factor", type=float, default=default)

    parser.add_argument("--grayscale", action="store_true")

    # enhancement augmentations - image only
    
    parser.add_argument("--autocontrast",     action="store_true")
    parser.add_argument("--autocontrast-factor", type=float, default=0.0)  
    
    parser.add_argument("--equalize",     action="store_true")
    parser.add_argument("--gamma",        action="store_true")
    parser.add_argument("--gamma-factor", type=float, default=1.0)

    parser.add_argument("--clahe",        action="store_true")
    parser.add_argument("--clahe-clip-limit", type=float, default=1.5)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8)

    args = parser.parse_args()

    out_images = Path(args.output_dir) / "images"
    out_masks  = Path(args.output_dir) / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    pairs, skipped = find_pairs(args.image_dir, args.mask_dir)

    if args.limit:
        pairs = pairs[:args.limit]

    geometric_count   = 0
    photometric_count = 0

    for image_path, mask_path in pairs:
        stem  = image_path.stem
        image = Image.open(image_path).convert("RGB")
        mask  = Image.open(mask_path)

        # resize before augmenting
        # BILINEAR for images, NEAREST for masks to preserve class label values
        image = image.resize((args.width, args.height), Image.Resampling.BILINEAR)
        mask  = mask.resize((args.width, args.height),  Image.Resampling.NEAREST)

        # save the resized original
        image.save(out_images / f"{stem}.jpg")
        mask.save(out_masks   / f"{stem}.png")

        # geometric augmentations - mask transformed alongside image
        for name, aug_image, aug_mask in geo_apply(image, mask, args):
            aug_image.save(out_images / f"{stem}_{name}.jpg")
            aug_mask.save(out_masks   / f"{stem}_{name}.png")
            geometric_count += 1

        # photometric and enhancement - image only, original mask saved alongside
        for name, aug_image in transforms_apply(image, args):
            aug_image.save(out_images / f"{stem}_{name}.jpg")
            mask.save(out_masks       / f"{stem}_{name}.png")
            photometric_count += 1

    total = len(pairs) + geometric_count + photometric_count
    print(f"\nDone. Saved to: {args.output_dir}")
    print(f"  Originals:   {len(pairs)}")
    print(f"  Geometric:   {geometric_count}")
    print(f"  Photometric: {photometric_count}")
    print(f"  Total:       {total}")

    if skipped:
        print(f"\nSkipped {len(skipped)} pairs:")
        for filename, reason in skipped:
            print(f"  - {filename}: {reason}")


if __name__ == "__main__":
    main()
