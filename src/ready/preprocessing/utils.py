from pathlib import Path
from PIL import Image


def find_pairs(image_dir, mask_dir):
    """
    Find image-mask pairs that can be used.
    A pair is only accepted if both files exist and have the same size.
    """

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    pairs = []
    skipped = []

    # Look for png and tif masks
    mask_files = (
        list(mask_dir.rglob("*.png")) +
        list(mask_dir.rglob("*.tif")) +
        list(mask_dir.rglob("*.tiff"))
    )

    # Start from the masks and look for matching images
    for mask_path in sorted(mask_files):
        rel_path = mask_path.relative_to(mask_dir)

        image_path = image_dir / rel_path.with_suffix(".jpg")

        if not image_path.exists():
            image_path = image_dir / rel_path.with_suffix(".png")

        if not image_path.exists():
            image_path = image_dir / rel_path.with_suffix(".tif")

        if not image_path.exists():
            image_path = image_dir / rel_path.with_suffix(".tiff")

        if not image_path.exists():
            skipped.append((mask_path.name, "no matching image found"))
            continue

        # Check that the image and mask are the same size
        with Image.open(image_path) as image:
            image_size = image.size

        with Image.open(mask_path) as mask:
            mask_size = mask.size

        if image_size != mask_size:
            skipped.append((mask_path.name, f"image size {image_size} does not match mask size {mask_size}"))
            continue

        pairs.append((image_path, mask_path))

    return pairs, skipped


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python utils.py <image-dir> <mask-dir>")
        sys.exit(1)

    pairs, skipped = find_pairs(sys.argv[1], sys.argv[2])

    print("Valid pairs found:", len(pairs))
    print("Skipped files:", len(skipped))

    for filename, reason in skipped:
        print(f"- {filename}: {reason}")
