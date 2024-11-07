"""
https://www.kaggle.com/code/soumicksarker/openeds-demo
"""

import os

import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from PIL import Image

from src.ready.utils.utils import HOME_PATH, set_data_directory

MAIN_PATH = os.path.join(HOME_PATH, "Desktop/nystagmus-tracking/")


def test_data_path():
    """
    Test data path
    python -m pytest -v -s tests/test_data_paths.py::test_data_path
    """
    set_data_directory(main_path=MAIN_PATH, data_path="datasets/openEDS")
    logger.info(f"Current working directory: {os.getcwd()}")

    assert os.getcwd() == MAIN_PATH + "datasets/openEDS"

def test_mobious_dataset():
    """
    Test mobious dataset
    python -m pytest -v -s tests/test_data_paths.py::test_mobious_dataset

    """
    print("mobious")
    # set_data_directory("datasets/mobious/MOBIOUS") #TODO test main data
    set_data_directory(data_path="data/mobious/sample-frames/test640x400")

    raw = mimg.imread("images/1_1i_Ll_1.jpg")
    mask = mimg.imread("masks/1_1i_Ll_1.png")
    lnp = np.asarray(Image.open("masks/1_1i_Ll_1.png").convert("RGBA"))
    sclera = lnp[:, :, 0]
    iris = lnp[:, :, 1]
    pupil = lnp[:, :, 2]
    #TOPLOT bck = lnp[:, :, 3]

    logger.info(f"raw.shape: {raw.shape}")
    logger.info(f"lnp.shape: {lnp.shape}")

    assert raw.shape == (400, 640, 3)
    assert lnp.shape == (400, 640, 4)

    # Plots
    plt.subplot(2, 3, 1)
    plt.imshow(raw)
    plt.title("Raw image (.jpg)")

    plt.subplot(2, 3, 2)
    plt.imshow(mask)
    plt.title("Mask (.png)")

    plt.subplot(2, 3, 4)
    plt.imshow(sclera, cmap="Reds")
    plt.title("Sclera")

    plt.subplot(2, 3, 5)
    plt.imshow(iris, cmap="Greens")
    plt.title("Iris")

    plt.subplot(2, 3, 6)
    plt.imshow(pupil, cmap="Blues")
    plt.title("Pupil")

    plt.show()


def test_mobious_dataset_labels():
    """
    Test mobious dataset
    python -m pytest -v -s tests/test_data_paths.py::test_mobious_dataset_labels

    """
    print("mobious")
    # set_data_directory("datasets/mobious/MOBIOUS") #TODO test main data
    set_data_directory(data_path="data/mobious/sample-frames/test640x400")

    #TODO: TEST each of the following images
    # imagename="1_1i_Ll_1"
    # imagename="1_1i_Lr_1"
    # imagename="1_1i_Ll_2"
    # imagename="1_1i_Lr_2"
    imagename = "1_1i_Ls_1"

    raw = mimg.imread("images/" + imagename + ".jpg")
    l = np.load("labels/" + imagename + ".npy")
    print(type(l))
    print(l.shape)
    mask_sclera = l[:, :, 0]  # sclera
    mask_iris = l[:, :, 1]  # iris
    mask_pupil = l[:, :, 2]  # pupil
    #TOPLOT mask_bck = l[:, :, 3]  # pupil

    logger.info(f"raw.shape: {raw.shape}")
    logger.info(f"l.shape: {l.shape}")

    assert raw.shape == (400, 640, 3)
    assert l.shape == (400, 640, 4)


    #Plots
    plt.subplot(1, 3, 1)
    # plt.imshow(raw, "gray", interpolation="none")
    plt.imshow(raw)
    plt.imshow(mask_sclera, "jet", interpolation="none", alpha=0.5)
    plt.title("Sclera")

    plt.subplot(1, 3, 2)
    # plt.imshow(m, "gray", interpolation="none")
    plt.imshow(raw)
    plt.imshow(mask_iris, "jet", interpolation="none", alpha=0.5)
    plt.title("Iris")

    plt.subplot(1, 3, 3)
    # plt.imshow(m, "gray", interpolation="none")
    plt.imshow(raw)
    plt.imshow(mask_pupil, "jet", interpolation="none", alpha=0.5)
    plt.title("Pupil")

    plt.show()


def test_openEDS_dataset():
    """
    Test openEDS dataset
    python -m pytest -v -s tests/test_data_paths.py::test_openEDS_dataset

    """
    print("openEDS")
    # set_data_directory("datasets/mobious/MOBIOUS") #TODO test main data
    set_data_directory(data_path="data/openEDS/sample-frames/val3frames")

    raw = mimg.imread("images/000160.png")
    mask = np.load("labels/000160.npy")
    # lnp = np.asarray(Image.open("masks/000160.png").convert("RGBA"))
    sclera = mask > 0
    iris = mask > 1
    pupil = mask > 2

    logger.info(f"raw.shape: {raw.shape}")
    logger.info(f"mask.shape: {mask.shape}")

    assert raw.shape == (400, 640)
    assert mask.shape == (400, 640)

    # Plots
    plt.subplot(2, 3, 1)
    plt.imshow(raw, cmap="gray")
    plt.title("Raw image (.png)")

    plt.subplot(2, 3, 2)
    plt.imshow(mask)
    plt.title("Mask (.npy)")

    plt.subplot(2, 3, 4)
    plt.imshow(sclera, cmap="Reds")
    plt.title("Sclera")

    plt.subplot(2, 3, 5)
    plt.imshow(iris, cmap="Greens")
    plt.title("Iris")

    plt.subplot(2, 3, 6)
    plt.imshow(pupil, cmap="Blues")
    plt.title("Pupil")

    plt.show()


def test_png_openEDS():
    """
    test png image with Image open
    python -m pytest -v -s tests/test_data_paths.py::test_png_openEDS
    """
    set_data_directory(main_path=MAIN_PATH, data_path="datasets/openEDS")
    im = Image.open("openEDS/openEDS/S_0/0.png")

    logger.info(f"im.size: {im.size}")
    logger.info(f"im.mode: {im.mode}")
    logger.info(f"im.channels: {len(im.split())}")

    assert im.size == (640, 400)
    assert im.mode == "L"
    assert len(im.split()) == 1

    assert im.show() == None


def test_imread_openEDS():
    """
    test png with imread
    """
    set_data_directory(main_path=MAIN_PATH, data_path="datasets/openEDS")

    m = mimg.imread("openEDS/openEDS/S_0/0.png")
    plt.imshow(m)
    plt.show()


def test_np_load_openEDS():
    """
    test npy data
    """
    set_data_directory(main_path=MAIN_PATH, data_path="datasets/openEDS")

    t = np.load("openEDS/openEDS/S_0/0.npy")
    plt.imshow(t)  # , cmap = 'rainbow')
    plt.colorbar()
    plt.show()
    print(t.shape)
    print(t)


def test_masks_openEDS():
    """
    test masks overlaid with original png image
    """
    set_data_directory(main_path=MAIN_PATH, data_path="datasets/openEDS")

    m = mimg.imread("openEDS/openEDS/S_0/0.png")
    t = np.load("openEDS/openEDS/S_0/0.npy")
    print(type(t))
    print(t.shape)
    mask_sclera = t > 0  # sclera
    print(type(mask_sclera))
    mask_iris = t > 1  # iris
    mask_pupil = t > 2  # pupil

    plt.subplot(1, 3, 1)
    plt.imshow(m, "gray", interpolation="none")
    plt.imshow(mask_sclera, "jet", interpolation="none", alpha=0.5)
    plt.title("Sclera")

    plt.subplot(1, 3, 2)
    plt.imshow(m, "gray", interpolation="none")
    plt.imshow(mask_iris, "jet", interpolation="none", alpha=0.5)
    plt.title("Iris")

    plt.subplot(1, 3, 3)
    plt.imshow(m, "gray", interpolation="none")
    plt.imshow(mask_pupil, "jet", interpolation="none", alpha=0.5)
    plt.title("Pupil")

    plt.show()


def test_txt_openEDS():
    """
    test txt
    """
    set_data_directory(main_path=MAIN_PATH, data_path="datasets/openEDS")

    with open("bbox/bbox/S_0.txt", encoding="utf-8") as file_handle:
        text = file_handle.read().split("\n")
        print(type(text))
        print(text)


def test_data_path_rit_eyes():
    """
    Test data path
    python -m pytest -v -s tests/test_data_paths.py::test_data_path_rit_eyes
    """
    set_data_directory(main_path=MAIN_PATH, data_path="datasets/RIT-eyes")
    print(os.getcwd())


def test_tif_with_image_rit_eyes():
    """
    test png image with Image open
    python -m pytest -v -s tests/test_data_paths.py::test_tif_with_image_rit_eyes
    """
    set_data_directory(main_path=MAIN_PATH, data_path="datasets/RIT-eyes")
    im = Image.open("12/synthetic/0000.tif")


    logger.info(f"im.size: {im.size}")
    logger.info(f"im.mode: {im.mode}")
    logger.info(f"im.channels: {len(im.split())}")


    assert im.size == (640, 480)
    assert im.mode == "RGB"
    assert len(im.split()) == 3

    assert im.show() == None

    m = mimg.imread("12/synthetic/0000.tif")
    plt.imshow(m)
    plt.show()


def test_tif_with_matplotlib_rit_eyes():
    """
    test png image with Image open
    python -m pytest -v -s tests/test_data_paths.py::test_tif_with_matplotlib
    """
    set_data_directory(main_path=MAIN_PATH, data_path="datasets/RIT-eyes")
    image = mimg.imread("12/synthetic/0000.tif")
    plt.imshow(image)
    plt.show()

    mask = mimg.imread("12/mask-withskin/0000.tif")
    plt.imshow(mask)
    plt.show()


def test_load_pickle_file_rit_eyes():
    """
    # TODO:
    test pickle data
    import pickle
    python -m pytest -v -s tests/test_data_paths.py::test_load_pickle_file
    """
    set_data_directory(main_path=MAIN_PATH, data_path="datasets/RIT-eyes")
    # t = np.load("12/12-natural.p", allow_pickle=False)
    # plt.imshow(t) #, cmap = 'rainbow')
    # plt.colorbar()
    # plt.show()
    # print(t.shape)
    # print(t)
