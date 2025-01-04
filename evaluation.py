## Import the required libraries
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
import os
import pickle
from tqdm import tqdm

# local packages
from preprocess import image_to_tensor, save_dict_to_file


def randomLRImage(half, testL, testR, testR1=None, model=None, random_index=1):

    ## function to generate a L0, R0, R1, and pred_test image for a random index given a model

    n = 0
    if model is not None:
        model.eval()
    if random_index == 0:
        raise (ValueError("Random index should be greater than 0"))
    if model is None:
        raise (ValueError("Model not found"))

    if testR1 is not None:
        for l0_test, r0_test, r1_test in zip(testL, testR, testR1):
            if half:
                print("Half precision")
                l0_test = l0_test.half().cuda()
            else:
                l0_test = l0_test.cuda()
            r0_test = r0_test.cuda()
            r1_test = r1_test.cuda()
            pred_test = model(l0_test, r1_test)
            if n == random_index:
                break
            n += 1
            del l0_test
            del r0_test
            del r1_test

    else:
        for l0_test, r0_test in zip(testL, testR):
            r0_test = r0_test.cuda()
            if half:
                l0_test = l0_test.half().cuda()
            else:
                l0_test = l0_test.cuda()
            if testR1 is not None:
                pred_test, _ = model(l0_test)
            pred_test = model(l0_test)
            if n == random_index - 1:
                r1_test = r0_test
            if n == random_index:
                break
            n += 1

    return l0_test, r0_test, r1_test, pred_test, n


def plot_L_R_pred(l0_test, r0_test, r1_test, pred_test, idx=1, assessment_plots=False):

    ### function to plot the L0, R0, R1, and pred_test images with/without assessment

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim = ssim.cuda()
    l = l0_test.cuda()
    r = r0_test.cuda()
    r1 = r1_test.cuda()
    pred_test = pred_test.cuda()
    x = 1
    # make subplots bigger
    if assessment_plots:
        x = 2
    plt.figure(figsize=(20, 8))
    plt.subplot(x, 3, 1)
    plt.title("Left Image (L) : " + str(idx))
    plt.imshow(l[0].permute(2, 1, 0).cpu().detach().numpy())
    plt.subplot(x, 3, 2)
    plt.title("Right Image (R)")
    plt.imshow(r[0].permute(2, 1, 0).cpu().detach().numpy())
    plt.subplot(x, 3, 3)
    plt.title("Predicted Right (R')")
    plt.imshow(pred_test[0].permute(2, 1, 0).cpu().detach().numpy())

    if assessment_plots:
        plt.subplot(x, 3, 4)
        diff = round(torch.sum((l[0] - r[0]) ** 2).item() / (384 * 160 * 3) * 100, 4)
        ssim_ = round(ssim(l, r).item(), 4)
        plt.title(
            "Pixel Diff % (L/R) : " + str(diff) + "\n" + "SSIM (L/R) : " + str(ssim_)
        )
        plt.imshow((l[0] - r[0]).permute(2, 1, 0).cpu().detach().numpy().clip(0, 1))
        plt.subplot(x, 3, 5)
        diff = round(torch.sum((r1[0] - r[0]) ** 2).item() / (384 * 160 * 3) * 100, 4)
        ssim_ = round(ssim(r1, r).item(), 4)
        plt.title(
            "Pixel Diff % (R-1/R) : "
            + str(diff)
            + "\n"
            + "SSIM (R-1/R) : "
            + str(ssim_)
        )
        plt.imshow((r1[0] - r[0]).permute(2, 1, 0).cpu().detach().numpy().clip(0, 1))
        plt.subplot(x, 3, 6)
        diff = round(
            torch.sum((pred_test[0] - r[0]) ** 2).item() / (384 * 160 * 3) * 100, 4
        )
        ssim_ = round(ssim(pred_test, r).item(), 4)
        plt.title(
            "Pixel Diff % (R'/R) : " + str(diff) + "\n" + "SSIM (R'/R) : " + str(ssim_)
        )
        plt.imshow(
            (pred_test[0] - r[0]).permute(2, 1, 0).cpu().detach().numpy().clip(0, 1)
        )
    plt.show()


def train_test_plot(train_loss, test_loss, title="Train vs Test Loss"):
    ### function to plot the train and test loss
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.xlabel("Epoch No. ->")
    plt.ylabel("Loss ->")
    plt.title(title)
    plt.legend()
    plt.show()


def avg_valuesPrint(
    Avg_lr_list,
    Avg_or_list,
    Avg_r1r_list,
    set_name="Test",
    metric="SSIM",
    print_values=True,
    return_values=False,
):
    """
    function to calculate the average values over diff. sets.
    """
    if print_values:

        print(f"\nAverage Values of {metric} over the {set_name} set")

        print(
            f"{metric} Avg over the {set_name} set LR: ",
            sum(Avg_lr_list) / len(Avg_lr_list),
        )
        print(
            f"{metric} Avg over the {set_name} set R'R: ",
            sum(Avg_or_list) / len(Avg_or_list),
        )
        print(
            f"{metric} Avg over the {set_name} set R-1 R: ",
            sum(Avg_r1r_list) / len(Avg_r1r_list),
        )
    if return_values:
        return (
            sum(Avg_lr_list) / len(Avg_lr_list),
            sum(Avg_or_list) / len(Avg_or_list),
            sum(Avg_r1r_list) / len(Avg_r1r_list),
        )


def ssimAll_over_time(
    test_L0,
    test_R0,
    test_R1,
    model,
    stacked=False,
    show_plot=True,
    print_values=True,
    fig_size=(15, 9),
    set_name="Test",
    half=False,
):
    """
    Function to compare the SSIM between L0 and R0, R1 and R0, and R_pred and R0
    """

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim = ssim.cuda()
    ssim_l_r = []
    ssim_pred_r = []
    ssim_r1_r = []
    model.eval()
    for l_test, r_test, r1_test in zip(test_L0, test_R0, test_R1):
        if half:
            l_test_cuda = l_test.half().cuda()
        else:
            l_test_cuda = l_test.cuda()
        r_test_cuda = r_test.cuda()
        r1_test_cuda = r1_test.cuda()
        if stacked:
            pred_test = model(l_test_cuda, r1_test_cuda)
        else:
            pred_test = model(l_test_cuda)
        ssim_l_r.append(ssim(l_test_cuda, r_test_cuda).item())
        ssim_pred_r.append(ssim(pred_test, r_test_cuda).item())
        ssim_r1_r.append(ssim(r1_test_cuda, r_test_cuda).item())
        del pred_test
        del l_test_cuda
        del r_test_cuda
        del r1_test_cuda

    # print(len(ssim_l_r), len(ssim_pred_r), len(ssim_r1_r))
    if show_plot:
        plt.figure(figsize=fig_size)
        plt.plot(ssim_l_r, label="L0 vs R0")
        plt.plot(ssim_pred_r, label="R_pred vs R0")
        plt.plot(ssim_r1_r, label="R1 vs R0")
        plt.xlabel("Frame No. ->")
        plt.ylabel("SSIM (More the better) ->")
        plt.legend()
        plt.title("SSIM Comparisons of different base lines")
        plt.show()

    if print_values:
        avg_valuesPrint(
            ssim_l_r, ssim_pred_r, ssim_r1_r, set_name=set_name, metric="SSIM"
        )

    return (ssim_l_r, ssim_pred_r, ssim_r1_r)


def pixelDiffAll_over_time(
    test_L0,
    test_R0,
    test_R1,
    model,
    stacked=False,
    show_plot=True,
    print_values=True,
    fig_size=(15, 9),
    set_name="Test",
):
    """
    Function to compare the pixel wise difference between L0 and R0, R1 and R0, and R_pred and R0
    """
    diff_l_r = []
    diff_pred_r = []
    diff_r1_r = []
    model.eval()
    for l_test, r_test, r1_test in zip(test_L0, test_R0, test_R1):
        l_test = l_test.cuda()
        r_test = r_test.cuda()
        r1_test = r1_test.cuda()
        if stacked:
            pred_test, _ = model(l_test, r1_test)
        else:
            pred_test, _ = model(l_test)
        diff_pred_r.append(
            round(
                torch.sum((pred_test[0] - r_test[0]) ** 2).item()
                / (384 * 160 * 3)
                * 100,
                4,
            )
        )
        diff_l_r.append(
            round(
                torch.sum((l_test[0] - r_test[0]) ** 2).item() / (384 * 160 * 3) * 100,
                4,
            )
        )
        diff_r1_r.append(
            round(
                torch.sum((r1_test[0] - r_test[0]) ** 2).item() / (384 * 160 * 3) * 100,
                4,
            )
        )

    if show_plot:
        # change plt figure size
        plt.figure(figsize=fig_size)
        plt.plot(diff_l_r, label="L0 vs R0")
        plt.plot(diff_pred_r, label="R_pred vs R0")
        plt.plot(diff_r1_r, label="R1 vs R0")
        plt.legend()
        plt.xlabel("Frame No. ->")
        plt.ylabel("Pixel Diff % (Less is better) ->")
        plt.title("Pixel wise % Diff. of different base lines")
        plt.show()

    if print_values:
        avg_valuesPrint(
            diff_l_r, diff_pred_r, diff_r1_r, set_name=set_name, metric="Pixel Diff %"
        )

    return (diff_l_r, diff_pred_r, diff_r1_r)


def PSNRAll_over_time(
    test_L0,
    test_R0,
    test_R1,
    model,
    stacked=False,
    show_plot=True,
    print_values=True,
    fig_size=(15, 9),
    set_name="Test",
    half=False,
):
    """
    Function to compare the SSIM between L0 and R0, R1 and R0, and R_pred and R0
    """

    psnr = PeakSignalNoiseRatio(data_range=1.0)
    psnr = psnr.cuda()
    psnr_l_r = []
    psnr_pred_r = []
    psnr_r1_r = []
    model.eval()
    for l_test, r_test, r1_test in zip(test_L0, test_R0, test_R1):
        if half:
            l_test_cuda = l_test.half().cuda()
        else:
            l_test_cuda = l_test.cuda()
        r_test_cuda = r_test.cuda()
        r1_test_cuda = r1_test.cuda()
        if stacked:
            pred_test = model(l_test_cuda, r1_test_cuda)
        else:
            pred_test = model(l_test_cuda)
        psnr_l_r.append(psnr(l_test_cuda, r_test_cuda).item())
        psnr_pred_r.append(psnr(pred_test, r_test_cuda).item())
        psnr_r1_r.append(psnr(r1_test_cuda, r_test_cuda).item())
        del pred_test
        del l_test_cuda
        del r_test_cuda
        del r1_test_cuda

    # print(len(ssim_l_r), len(ssim_pred_r), len(ssim_r1_r))
    if show_plot:
        plt.figure(figsize=fig_size)
        plt.plot(psnr_l_r, label="L0 vs R0")
        plt.plot(psnr_pred_r, label="R_pred vs R0")
        plt.plot(psnr_r1_r, label="R1 vs R0")
        plt.xlabel("Frame No. ->")
        plt.ylabel("PSNR (Less the better) ->")
        plt.legend()
        plt.title("PSNR Comparisons of different base lines")
        plt.show()

    if print_values:
        avg_valuesPrint(
            psnr_l_r, psnr_pred_r, psnr_r1_r, set_name=set_name, metric="PSNR"
        )

    return (psnr_l_r, psnr_pred_r, psnr_r1_r)


def ssim_over_time(test_L0, test_R0, model):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim = ssim.cuda()
    ssim_l_r = []
    ssim_r_pred = []
    for l_test, r_test in zip(test_L0, test_R0):
        l_test = l_test.cuda()
        r_test = r_test.cuda()
        pred_test, _ = model(l_test)
        ssim_l_r.append(ssim(l_test, r_test).item())
        ssim_r_pred.append(ssim(r_test, pred_test).item())

    plt.plot(ssim_l_r, label="L vs R")
    plt.plot(ssim_r_pred, label="R vs R'")
    plt.xlabel("Frame No. ->")
    plt.ylabel("SSIM ->")
    plt.legend()
    plt.title("SSIM between R and R' and L and R'")
    plt.show()


def pixelDiff_over_time(test_L0, test_R0, model):
    diff_l_r = []
    diff_r_pred = []
    for l_test, r_test in zip(test_L0, test_R0):
        l_test = l_test.cuda()
        r_test = r_test.cuda()
        pred_test, _ = model(l_test)
        diff_r_pred.append(
            round(
                torch.sum((r_test[0] - pred_test[0]) ** 2).item()
                / (384 * 160 * 3)
                * 100,
                4,
            )
        )
        diff_l_r.append(
            round(
                torch.sum((r_test[0] - l_test[0]) ** 2).item() / (384 * 160 * 3) * 100,
                4,
            )
        )

    plt.plot(diff_r_pred, label="R vs R'")
    plt.plot(diff_l_r, label="L vs R")
    plt.xlabel("Frame No. ->")
    plt.ylabel("Pixel Diff % ->")
    plt.legend()
    plt.title("Pixel wise % Difference between R and R' and L and R'")
    plt.show()
