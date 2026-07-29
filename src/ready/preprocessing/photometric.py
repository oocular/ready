from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np


def apply_clahe(image, clip_limit=1.5, tile_grid_size=8):
    """
    Apply CLAHE to improve local contrast in the image.
    """

    image_array = np.array(image)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size)
    )

    r, g, b = cv2.split(image_array)

    r = clahe.apply(r)
    g = clahe.apply(g)
    b = clahe.apply(b)

    new_image = cv2.merge([r, g, b])

    return Image.fromarray(new_image)


def apply_gamma(image, gamma_factor=1.0):
    """
    Change image brightness using gamma correction.
    """

    table = [int((i / 255.0) ** gamma_factor * 255) for i in range(256)]

    return image.point(table * 3)


def apply_photometric(image, args):
    """
    These changes only affect the image.
    The mask is not changed because it stores class labels.
    """

    results = []

    if args.brightness_up:
        new_image = ImageEnhance.Brightness(image).enhance(args.brightness_up_factor)
        results.append(("brightness_up", new_image))

    if args.brightness_down:
        new_image = ImageEnhance.Brightness(image).enhance(args.brightness_down_factor)
        results.append(("brightness_down", new_image))

    if args.contrast_up:
        new_image = ImageEnhance.Contrast(image).enhance(args.contrast_up_factor)
        results.append(("contrast_up", new_image))

    if args.blur:
        new_image = image.filter(ImageFilter.GaussianBlur(radius=args.blur_factor))
        results.append(("blur", new_image))

    if args.grayscale:
        new_image = ImageOps.grayscale(image).convert("RGB")
        results.append(("grayscale", new_image))

    if args.equalize:
        new_image = ImageOps.equalize(image)
        results.append(("equalize", new_image))

    if args.autocontrast:
        new_image = ImageOps.autocontrast(image, cutoff=args.autocontrast_factor)
        results.append(("autocontrast", new_image))

    if args.clahe:
        new_image = apply_clahe(
            image,
            clip_limit=args.clahe_clip_limit,
            tile_grid_size=args.clahe_tile_grid_size
        )
        results.append(("clahe", new_image))

    if args.gamma:
        new_image = apply_gamma(image, gamma_factor=args.gamma_factor)
        results.append(("gamma", new_image))

    return results
