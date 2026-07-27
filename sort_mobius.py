import pathlib


def get_mobious_filenames(data_path_string: str = None):
    data_path = pathlib.Path(data_path_string)

    list_of_imagefiles = []
    print(data_path)
    for name in data_path.iterdir():
        list_of_imagefiles.append(name)
    return list_of_imagefiles

# compare the list of images and masks, and move any unmatched images to a new folder called "unmatched_images" in the data_root_string directory
def compare_and_move(images_lists, masks_lists, data_root_string):
    

    # Create a set of mask filenames for faster lookup
    mask_filenames_set = {mask_file.stem for mask_file in masks_lists}

    # Create a new directory to move unmatched images
    unmatched_dir = pathlib.Path(data_root_string) / "unmatched_images"
    unmatched_dir.mkdir(exist_ok=True)

    print(f"Unmatched images will be moved to: {unmatched_dir}")

    unmatched_count = 0

    for image_file in images_lists:
        print(image_file)
        if image_file.stem not in mask_filenames_set:
            # Move the unmatched image to the new directory
            new_location = unmatched_dir / image_file.name
            image_file.rename(new_location)
            print(f"Moved {image_file} to {new_location}")
            unmatched_count += 1

    print(f"\nTotal images checked: {len(images_lists)}")
    print(f"Total masks found:    {len(masks_lists)}")
    print(f"Unmatched images:     {unmatched_count}")


if __name__ == "__main__":

    data_root_string = "datasets/MOBIUS"

    data_path_string = f"{data_root_string}/Images"
    image_files = get_mobious_filenames(data_path_string)

    data_path_string2 = f"{data_root_string}/Masks"
    masks_files = get_mobious_filenames(data_path_string2)

    compare_and_move(image_files, masks_files, data_root_string)
