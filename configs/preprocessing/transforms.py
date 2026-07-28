
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np

def apply_clahe(image, clip_limit=1.5, tile_grid_size=8):
    img_array = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    r , g, b= cv2.split(img_array)
    r_clahe = clahe.apply(r)
    g_clahe = clahe.apply(g)
    b_clahe = clahe.apply(b)
    merged_image = cv2.merge([r_clahe, g_clahe, b_clahe])
    return Image.fromarray(merged_image)

def apply_gamma(image, gamma_factor=1.0):
    table = [int((i / 255.0) ** gamma_factor * 255) for i in range(256)]
    return image.point(table*3)


def apply(image, args):
    """
    Photometric and enhancement augmentations applied to image only not applied to Mask as the pixel label values must stay intact.
    """
    results = []

    if args.brightness_up:
        results.append(("brightness_up", ImageEnhance.Brightness(image).enhance(args.brightness_up_factor)))

    if args.brightness_down:
        results.append(("brightness_down", ImageEnhance.Brightness(image).enhance(args.brightness_down_factor)))

    if args.contrast_up:
        results.append(("contrast_up", ImageEnhance.Contrast(image).enhance(args.contrast_up_factor)))

    if args.blur:
        results.append(("blur", image.filter(ImageFilter.GaussianBlur(radius=args.blur_factor))))

    if args.grayscale:
        results.append(("grayscale", ImageOps.grayscale(image).convert("RGB")))

    if args.equalize:
        results.append(("equalize", ImageOps.equalize(image)))

    if args.autocontrast:
        results.append(("autocontrast", ImageOps.autocontrast(image, cutoff=args.autocontrast_factor)))

    if args.clahe:
        results.append(("clahe", apply_clahe(image, clip_limit=args.clahe_clip_limit, tile_grid_size=args.clahe_tile_grid_size)))

    if args.gamma:
        results.append(("gamma", apply_gamma(image, gamma_factor=args.gamma_factor)))

    return results
