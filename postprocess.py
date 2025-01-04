import os
import shutil
import time
import torch
from PIL import Image
import torchvision
from torchvision import transforms
import pickle
from tqdm import tqdm
from preprocess import filenames, make_folder, path_to_imagelist
from patch import encoder_frame_from_patches


# define make a video with frames +48 and -48 frames from the lowest ssim index using ffmpeg
def copy_selected_frames(
    path_to_frames,
    idx,
    buffer=2,
    fps=None,
    seconds=False,
    out_folder_path=None,
):
    if path_to_frames == None:
        raise ValueError("path_to_frames cannot be None")

    _, _, base_folder = filenames(path_to_frames)

    if out_folder_path != None:
        base_folder = out_folder_path

    temp_folder_path = make_folder("TempFolder", base_folder, remove_if_exists=True)

    # copy the frames from the temp folder
    if seconds:
        if fps is None:
            raise ValueError(
                "FPS not provided!!Please provide the FPS (frames per second) value for the video."
            )
        else:
            for frame_num in range(idx - buffer * fps, idx + buffer * fps):
                formatted_number = "{:07}".format(frame_num)
                shutil.copy(
                    path_to_frames + "/out_" + str(formatted_number) + ".jpg",
                    temp_folder_path,
                )

    else:
        for frame_num in range(idx - buffer, idx + buffer):
            formatted_number = "{:07}".format(frame_num)
            shutil.copy(
                path_to_frames + "/out_" + str(formatted_number) + ".jpg",
                temp_folder_path,
            )

    print("Frames copied successfully!")
    print(
        "Frames copied at: ",
        temp_folder_path,
    )
