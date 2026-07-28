import pathlib
import numpy as np
from PIL import Image
import shutil

from pathlib import Path
import shutil

# get all files in the OpenEDS dataset, including images and masks, and return them as a list
def get_files(data_root):
    data_root = Path(data_root)
    files = []

    for subject_folder in sorted(data_root.iterdir()):
        if subject_folder.is_dir() and subject_folder.name.startswith("1"):
            for subfolder in subject_folder.iterdir():
                if subfolder.is_dir():
                    for file in subfolder.iterdir():
                        files.append(file)
    return files


# sort the OpenEDS files into separate folders for images and masks, and rename them to include the subject folder name 
def sort_openeds_files(files, data_root):
    data_root = Path(data_root)

    images_folder = data_root / "synthetic1"
    masks_folder = data_root / "mask-withoutskin-noglasses1"

    images_folder.mkdir(exist_ok=True)
    masks_folder.mkdir(exist_ok=True)

    image_count = 0
    mask_count = 0
 # loops through the list of files and copies them to the appropriate folder based on their file extension. It also renames the files to include the subject folder name as a prefix.
    for file in files:
        new_name = f"{file.parent.parent.name}_{file.name}"

        if file.parent.name == "synthetic":
            shutil.copy(file, images_folder / new_name)
            image_count += 1

        elif file.parent.name == "mask-withoutskin-noglasses": #and not file.stem.startswith("label_"):
            shutil.copy(file, masks_folder / new_name)
            mask_count += 1

    print("Images copied:", image_count)
    print("Masks copied:", mask_count)

# this is the main entry point of the script, which gets all files in the OpenEDS dataset and sorts them into separate folders for images and masks
if __name__ == "__main__":
    #ata_root = "datasets/s-natural/s-natural"
    data_root = "\\\\rdp.arc.ucl.ac.uk\\ritd-ag-project-rd02iw-mxoch87\\datasets\\s-natural\\s-natural"

    files = get_files(data_root)
    sort_openeds_files(files, data_root)


