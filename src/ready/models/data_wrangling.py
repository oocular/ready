import pathlib



def get_mobious_filenames(data_path_string: str = None):
    data_path = pathlib.Path(data_path_string) #converts string input to path object for easy access to directory

    list_of_imagefiles = []
    print(data_path)
    for name in data_path.iterdir():
        list_of_imagefiles.append(name)
    return list_of_imagefiles


def compare_and_move(images_lists, masks_lists, data_root_string):
    """
    Compare the image and mask files and move the ones that do not have a corresponding mask to a new folder.
    """

    # Create a set of mask filenames for faster lookup
    mask_filenames_set = {mask_file.stem for mask_file in masks_lists} #comparing names of image and mask files without their extensions to find unatched ones

    # Create a new directory to move unmatched images if it does not exist
    unmatched_dir = pathlib.Path(data_root_string) / "unmatched_images"
    unmatched_dir.mkdir(exist_ok=True)

    print(f"Unmatched images will be moved to: {unmatched_dir}")

    #if image has no  corresponding mask, move it to the new directory
    for image_file in images_lists:
        print(image_file)
        if image_file.stem not in mask_filenames_set:
            # Move the unmatched image to the new directory
            new_location = unmatched_dir / image_file.name
            image_file.rename(new_location)
            print(f"Moved {image_file} to {new_location}")

            
if __name__ == "__main__":
    data_root_string = "C:/Users/Kofi Quarshie/Downloads/ready/datasets/mobious/MOBIOUS/train100per_1144"

    data_path_string = f"{data_root_string}/images"
    image_files = get_mobious_filenames(data_path_string)

    data_path_string2 = f"{data_root_string}/masks"
    masks_files = get_mobious_filenames(data_path_string2)

    compare_and_move(image_files, masks_files, data_root_string)