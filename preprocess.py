import os
import shutil

from torch.utils.data import DataLoader


############################# Specific INRIA dataset ###############################
def modify_filename_INRIA(filename):
    # function to modify the filename for INRIA dataset
    basename, extension = os.path.splitext(filename)
    if extension == ".right":
        superBasename, _ = os.path.splitext(basename)
        new_filename = superBasename + "R.jpg"
        return new_filename
    if extension == ".jpg":
        new_filename = basename + "L.jpg"
        return new_filename


def copy_files_INRIA(source_folder, destination_folderR, destination_folderL):
    # function to copy files ending with L/R from source folder to destination folder
    l, r = 0, 0
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folderR):
        os.makedirs(destination_folderR)

    if not os.path.exists(destination_folderL):
        os.makedirs(destination_folderL)

    # Get the list of files in the source folder
    files = os.listdir(source_folder)

    # Copy each file to the destination folder
    for file in files:
        source_path = os.path.join(source_folder, file)
        # if modified_filename ends with R.jpg, then copy the file to the destination folder
        if file.endswith("L.jpg"):
            l += 1
            destination_path = os.path.join(destination_folderL, file)
            shutil.copy2(source_path, destination_path)
            # modified_filename = modify_filename(file)
            # os.rename(os.path.join(destination_folderL, file), os.path.join(destination_folderL, modified_filename))

        elif file.endswith("R.jpg"):
            # copy to diff. folder
            r += 1
            destination_path = os.path.join(destination_folderR, file)
            shutil.copy2(source_path, destination_path)
            # modified_filename = modify_filename(file)
            # os.rename(os.path.join(destination_folderR, file), os.path.join(destination_folderR, modified_filename))

        else:
            print(file)

    print("Files copied successfully.")
    print("Total files copied = ", l + r)
    print("Left files copied = ", l)
    print("Right files copied = ", r)
    if l != r:
        print("Error: Left and Right files are not equal")
