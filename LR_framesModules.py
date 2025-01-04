# importing libraries
import shutil
from tqdm import tqdm
from torchvision import transforms

from torch.utils.data import DataLoader
import os


# importing user defined functions
from preprocess import make_folder, filenames
from evaluation import *


def train_test_split(
    source_folder_path,
    train_ratio=None,
    num_train_files=None,
    num_test_files=None,
    out_folder_path=None,
):
    # both train_ratio and num_train_files cannot be none
    if train_ratio is None and num_train_files is None:
        raise ValueError("Both train_ratio and num_train_files cannot be None")

    if train_ratio is not None and num_train_files is not None:
        raise ValueError(
            "Both train_ratio and num_train_files cannot be provided at the same time"
        )

    if source_folder_path == None:
        raise ValueError("LR_frames_path cannot be None")

    _, _, base_folder = filenames(source_folder_path)

    if out_folder_path != None:
        base_folder = out_folder_path

    train_folder = make_folder(
        folder_name="train", folder_path=base_folder, remove_if_exists=True
    )
    test_folder = make_folder(
        folder_name="test", folder_path=base_folder, remove_if_exists=True
    )

    # Get the list of files in the source folder
    files = sorted(os.listdir(source_folder_path))

    # Calculate the number of files for training and testing
    num_files = len(files)

    if train_ratio is not None:
        num_train_files = int(num_files * train_ratio)
        num_test_files = num_files - num_train_files

    if num_train_files is not None:
        num_test_files = num_files - num_train_files

    # Copy the train files to the train folder
    for i in range(num_train_files):
        file = files[i]
        source_path = os.path.join(source_folder_path, file)
        shutil.copy2(source_path, train_folder)

    # Copy the test files to the test folder
    for i in range(num_train_files, num_files):
        file = files[i]
        source_path = os.path.join(source_folder_path, file)
        shutil.copy2(source_path, test_folder)

    # Print the number of files and the number of files copied to each folder
    print("Total files: ", num_files)
    print("Train files: ", num_train_files)
    print("Test files: ", num_test_files)
    print("Files copied to train folder: ", num_train_files)
    print("Files copied to test folder: ", num_test_files)


def generate_L0_R0_R1(
    source_Lpath, source_Rpath, start_num=0, end_num=-1, out_folder_path=None
):
    """Function to copy and sort the images in the source path to the destination path
    Also, returns the dataloaders for the test images"""

    if source_Rpath == None or source_Lpath == None:
        raise ValueError("source_Rpath or source_Lpath cannot be None")

    _, _, base_folder = filenames(source_Rpath)

    if out_folder_path != None:
        base_folder = out_folder_path

    path_R0 = make_folder(
        folder_name="R0frames", folder_path=base_folder, remove_if_exists=True
    )
    path_L0 = make_folder(
        folder_name="L0frames", folder_path=base_folder, remove_if_exists=True
    )
    path_R1 = make_folder(
        folder_name="R1frames", folder_path=base_folder, remove_if_exists=True
    )

    # Get a list of all image files in the source path
    R0_files = sorted(os.listdir(source_Rpath))[start_num + 1 : end_num]
    R1_files = sorted(os.listdir(source_Rpath))[start_num : end_num - 1]
    L0_files = sorted(os.listdir(source_Lpath))[start_num + 1 : end_num]

    # Copy the sorted image files to the destination path
    for file in tqdm(R0_files, total=len(R0_files), desc="Copying R0 frames"):
        shutil.copy2(
            os.path.join(source_Rpath, file),
            path_R0,
        )
    print("Images copied to {} successfully!".format(path_R0))

    for file in tqdm(R1_files, total=len(R1_files), desc="Copying R1 frames"):
        shutil.copy2(
            os.path.join(source_Rpath, file),
            path_R1,
        )
    print("Images copied to {} successfully!".format(path_R1))

    for file in tqdm(L0_files, total=len(L0_files), desc="Copying L0 frames"):
        shutil.copy2(
            os.path.join(source_Lpath, file),
            path_L0,
        )
    print("Images copied to {} successfully!".format(path_L0))

    print("Images copied and sorted successfully in temp folder!")

    transform = transforms.ToTensor()

    print("\nR0 Samples:")
    testR0_sample = createTorchDataset(
        path_R0, transforms=transform, channel_last=False
    )
    print("\nL0 Samples:")
    testL0_sample = createTorchDataset(
        path_L0, transforms=transform, channel_last=False
    )
    print("\nR1 Samples:")
    testR1_sample = createTorchDataset(
        path_R1, transforms=transform, channel_last=False
    )

    test_R0 = DataLoader(testR0_sample, batch_size=1, shuffle=False)
    test_L0 = DataLoader(testL0_sample, batch_size=1, shuffle=False)
    test_R1 = DataLoader(testR1_sample, batch_size=1, shuffle=False)

    print("Data loaded successfully using pytorch dataloader!")

    return test_L0, test_R0, test_R1


def n_previous_frames(
    folder_path,
    folder_name=None,
    start_num=0,
    end_num=-1,
    out_folder_path=None,
    num_previous=1,
):
    """Function to copy and sort the images in the source path to the destination path
    Also, returns the dataloaders for the test images"""

    name, _, base_folder = filenames(folder_path)

    if out_folder_path != None:
        base_folder = out_folder_path
    if folder_name != None:
        name = folder_name

    path_dict = {}

    try:
        for num in range(num_previous + 1):
            path_dict["path" + str(num_previous)] = make_folder(
                folder_name=name[-1] + str(num) + "_frames",
                folder_path=base_folder,
                remove_if_exists=True,
            )

            frames = sorted(os.listdir(folder_path))[
                start_num + num_previous - num : end_num - num
            ]

            for file in tqdm(
                frames,
                total=len(frames),
                desc="Copying frames for " + name[-1] + str(num) + "_frames",
            ):
                shutil.copy2(
                    os.path.join(folder_path, file),
                    path_dict["path" + str(num_previous)],
                )
            print(
                "Images copied to {} successfully!".format(
                    path_dict["path" + str(num_previous)]
                )
            )
    except:
        raise ValueError("num_previous should be integer and/or > 0")

    return path_dict
