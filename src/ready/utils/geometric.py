from PIL import Image, ImageOps


def apply(image, mask, args):
    """
    Geometric augmentations applied to both image and mask together.
    Uses Image.Transpose for rotations to avoid any interpolation.
    """
    results = []

    if args.hflip:
        results.append(("hflip", ImageOps.mirror(image), ImageOps.mirror(mask)))

    if args.vflip:
        results.append(("vflip", ImageOps.flip(image), ImageOps.flip(mask)))

    if args.rot90:
        results.append(("rot90", image.transpose(Image.Transpose.ROTATE_90), mask.transpose(Image.Transpose.ROTATE_90)))

    if args.rot180:
        results.append(("rot180", image.transpose(Image.Transpose.ROTATE_180), mask.transpose(Image.Transpose.ROTATE_180)))

    if args.rot270:
        results.append(("rot270", image.transpose(Image.Transpose.ROTATE_270), mask.transpose(Image.Transpose.ROTATE_270)))

    return results