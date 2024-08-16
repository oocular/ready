"""
https://www.kaggle.com/code/soumicksarker/openeds-demo
"""

import os

import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ready.utils.utils import set_data_directory


def test_data_path():
    """
    Test data path
    """
    set_data_directory("datasets/openEDS")
    print(os.getcwd())


def test_png():
    """
    test png image with Image open
    """
    set_data_directory("datasets/openEDS")
    im = Image.open("openEDS/openEDS/S_0/0.png")
    print(f'Shape {im.size}')
    print(f'Mode {im.mode}')
    print(f'Channels: {len(im.split())}')
    im.show()


def test_imread():
    """
    test png with imread
    """
    set_data_directory("datasets/openEDS")
    m = mimg.imread("openEDS/openEDS/S_0/0.png")
    plt.imshow(m)
    plt.show()


def test_np_load():
    """
    test npy data
    """
    set_data_directory("datasets/openEDS")
    t = np.load("openEDS/openEDS/S_0/0.npy")
    plt.imshow(t) #, cmap = 'rainbow')
    plt.colorbar()
    plt.show()
    print(t.shape)
    print(t)


def test_masks():
    """
    test masks overlaid with original png image
    """
    set_data_directory("datasets/openEDS")
    m = mimg.imread("openEDS/openEDS/S_0/0.png")
    t = np.load("openEDS/openEDS/S_0/0.npy")
    print(type(t))
    mask_sclera = t > 0  # sclera
    print(type(mask_sclera))
    mask_iris = t > 1  # iris
    mask_pupil = t > 2  # pupil

    plt.subplot(1, 3, 1)
    plt.imshow(m, "gray", interpolation="none")
    plt.imshow(mask_sclera, "jet", interpolation="none", alpha=0.7)
    plt.title("Sclera")

    plt.subplot(1, 3, 2)
    plt.imshow(m, "gray", interpolation="none")
    plt.imshow(mask_iris, "jet", interpolation="none", alpha=0.7)
    plt.title("Iris")

    plt.subplot(1, 3, 3)
    plt.imshow(m, "gray", interpolation="none")
    plt.imshow(mask_pupil, "jet", interpolation="none", alpha=0.7)
    plt.title("Pupil")

    plt.show()


def test_txt():
    """
    test txt
    """
    set_data_directory("datasets/openEDS")
    with open("bbox/bbox/S_0.txt", encoding="utf-8") as file_handle:
        text = file_handle.read().split("\n")
        print(type(text))
        print(text)



def test_data_path_rit_eyes():
    """
    Test data path
    python -m pytest -v -s tests/test_data_paths.py::test_data_path_rit_eyes
    """
    set_data_directory("datasets/RIT-eyes")
    print(os.getcwd())

def test_tif_with_Image():
    """
    test png image with Image open
    """
    set_data_directory("datasets/RIT-eyes")
    im = Image.open("12/synthetic/0000.tif")
    print(f'Shape {im.size}')
    print(f'Mode {im.mode}')
    print(f'Channels: {len(im.split())}')
    im.show()

    m = mimg.imread("12/synthetic/0000.tif")
    plt.imshow(m)
    plt.show()

def test_tif_with_matplotlib():
    """
    test png image with Image open
    python -m pytest -v -s tests/test_data_paths.py::test_tif_with_matplotlib
    """
    set_data_directory("datasets/RIT-eyes")
    image = mimg.imread("12/synthetic/0000.tif")
    plt.imshow(image)
    plt.show()

    mask = mimg.imread("12/mask-withskin/0000.tif")
    plt.imshow(mask)
    plt.show()


def test_load_pickle_file():
    """
    # TODO:
    test pickle data
    import pickle
    python -m pytest -v -s tests/test_data_paths.py::test_load_pickle_file
    """
    set_data_directory("datasets/RIT-eyes")
    #t = np.load("12/12-natural.p", allow_pickle=False)
    #plt.imshow(t) #, cmap = 'rainbow')
    #plt.colorbar()
    #plt.show()
    #print(t.shape)
    #print(t)


