from PIL import Image, ImageOps


def apply_geometric(image, mask, args):
    """
    These changes are applied to the image and mask together.
    This keeps the mask lined up with the image.
    """

    results = []

    if args.hflip:
        new_image = ImageOps.mirror(image)
        new_mask = ImageOps.mirror(mask)
        results.append(("hflip", new_image, new_mask))

    if args.vflip:
        new_image = ImageOps.flip(image)
        new_mask = ImageOps.flip(mask)
        results.append(("vflip", new_image, new_mask))

    if args.rot90:
        new_image = image.transpose(Image.Transpose.ROTATE_90)
        new_mask = mask.transpose(Image.Transpose.ROTATE_90)
        results.append(("rot90", new_image, new_mask))

    if args.rot180:
        new_image = image.transpose(Image.Transpose.ROTATE_180)
        new_mask = mask.transpose(Image.Transpose.ROTATE_180)
        results.append(("rot180", new_image, new_mask))

    if args.rot270:
        new_image = image.transpose(Image.Transpose.ROTATE_270)
        new_mask = mask.transpose(Image.Transpose.ROTATE_270)
        results.append(("rot270", new_image, new_mask))

    return results
