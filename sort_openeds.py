from pathlib import Path
import shutil

# get all files in the OpenEDS dataset, including images and masks, and return them as a list
def get_files(data_root):
    data_root = Path(data_root)
    files = []

    for subject_folder in sorted(data_root.iterdir()):
        if subject_folder.is_dir() and subject_folder.name.startswith("S_"): #only process folders with names starting with "S_"
            for file in subject_folder.iterdir():
                files.append(file)

    return files

# sort the OpenEDS files into separate folders for images and masks, and rename them to include the subject folder name 
def sort_openeds_files(files, data_root):
    data_root = Path(data_root)

    images_folder = data_root / "images"
    masks_folder = data_root / "masks_npy"

    images_folder.mkdir(exist_ok=True)
    masks_folder.mkdir(exist_ok=True)

    image_count = 0
    mask_count = 0
 # loops through the list of files and copies them to the appropriate folder based on their file extension. It also renames the files to include the subject folder name as a prefix.
    for file in files:
        new_name = f"{file.parent.name}_{file.name}"

        if file.suffix == ".png":
            shutil.copy(file, images_folder / new_name)
            image_count += 1

        elif file.suffix == ".npy" and not file.stem.startswith("label_"):
            shutil.copy(file, masks_folder / new_name)
            mask_count += 1

    print("Images copied:", image_count)
    print("Masks copied:", mask_count)

# this is the main entry point of the script, which gets all files in the OpenEDS dataset and sorts them into separate folders for images and masks
if __name__ == "__main__":
    data_root = "datasets/archive/openEDS/openEDS"

    files = get_files(data_root)
    sort_openeds_files(files, data_root)