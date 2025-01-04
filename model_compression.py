import torch.nn.utils.prune as prune
import torch
from model_eval import loadModel, model_size, saveModel
from Deep3D_model import Net_1080
from evaluation import ssimAll_over_time, avg_valuesPrint
import torch.nn as nn


def calculate_bitwise_dense(parameters_to_prune=None):
    if parameters_to_prune is None:
        raise ValueError("parameters_to_prune cannot be None")
    total_bits = 0
    for module, _ in parameters_to_prune:
        total_bits += (module.weight.data.nelement() * 8) / 1024**2
    return total_bits


def calculate_bitwise_sparse(parameters_to_prune=None):
    if parameters_to_prune is None:
        raise ValueError("parameters_to_prune cannot be None")
    total_bits_64 = 0
    total_bits_16 = 0
    total_bits_8 = 0
    sp_T_dict = {}
    custom_sparse_dict = {}
    custom_sparse_dict_quant = {}

    for i, (module, _) in enumerate(parameters_to_prune):
        idx = module.weight.data.nonzero().T
        values = module.weight.data[module.weight.data != 0]
        sp_T = torch.sparse_coo_tensor(idx, values, module.weight.data.size())

        sp_T = sp_T.coalesce()

        sp_T_dict[str(i)] = sp_T

        total_bits_64 += (
            sp_T.indices().nelement() * 8 + sp_T.values().nelement() * 4
        ) / 1024**2

        idx = idx.to(torch.int8)
        custom_sparse_dict["idx_" + str(i)] = idx
        custom_sparse_dict["values_" + str(i)] = values

        total_bits_16 += (
            sp_T.indices().nelement() * 1 + sp_T.values().nelement() * 4
        ) / 1024**2

        values = values.to(torch.float16)
        custom_sparse_dict_quant["idx_" + str(i)] = idx
        custom_sparse_dict_quant["values_" + str(i)] = values

        total_bits_8 += (
            sp_T.indices().nelement() * 1 + sp_T.values().nelement() * 2
        ) / 1024**2

    return (
        total_bits_64,
        total_bits_16,
        total_bits_8,
        sp_T_dict,
        custom_sparse_dict,
        custom_sparse_dict_quant,
    )


def update_parameters_to_prune(model_):
    parameters_to_prune = (
        # (model_.group1[0], "weight"),
        # (model_.group1[0], "bias"),
        (model_.group2[0], "weight"),
        (model_.group2[0], "bias"),
        (model_.group3[0], "weight"),
        (model_.group3[0], "bias"),
        (model_.group3[2], "weight"),
        (model_.group3[2], "bias"),
        (model_.group4[0], "weight"),
        (model_.group4[0], "bias"),
        (model_.group4[2], "weight"),
        (model_.group4[2], "bias"),
        (model_.group5[0], "weight"),
        (model_.group5[0], "bias"),
        (model_.group5[2], "weight"),
        (model_.group5[2], "bias"),
        (model_.group6[0], "weight"),
        (model_.group6[0], "bias"),
        (model_.fc8, "weight"),
        (model_.fc8, "bias"),
        # (model_.pred4[0], "weight"),
        # (model_.pred4[0], "bias"),
        (model_.pred4[1], "weight"),
        (model_.pred4[1], "bias"),
        # (model_.pred3[0], "weight"),
        # (model_.pred3[0], "bias"),
        (model_.pred3[1], "weight"),
        (model_.pred3[1], "bias"),
        # (model_.pred2[0], "weight"),
        # (model_.pred2[0], "bias"),
        (model_.pred2[1], "weight"),
        (model_.pred2[1], "bias"),
        # (model_.pred1[0], "weight"),
        # (model_.pred1[0], "bias"),
        (model_.pred1[1], "weight"),
        (model_.pred1[1], "bias"),
        (model_.deConv1[1], "weight"),
        (model_.deConv1[1], "bias"),
        (model_.deConv2[1], "weight"),
        (model_.deConv2[1], "bias"),
        (model_.deConv3[1], "weight"),
        (model_.deConv3[1], "bias"),
        (model_.deConv4[1], "weight"),
        (model_.deConv4[1], "bias"),
        (model_.deConv5[1], "weight"),
        (model_.upConv, "weight"),
        # (model_.up[1], "weight"),
    )
    return parameters_to_prune


###################################### Global Pruning ########################################
def global_prune(
    model, prune_amt=0.1, parameters_to_prune=None, remove_prune_info=True
):

    if parameters_to_prune is None:
        raise ValueError("parameters_to_prune cannot be None")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_amt,
    )

    if remove_prune_info:
        for mod, param in parameters_to_prune:
            prune.remove(mod, param)

    return model


def global_prune_list(
    prune_amt_list,
    model_pt_path,
    train_L0,
    train_R0,
    train_R1,
    model_temp_path="/home/pytorch/personal/Deep3D/SelfDeep3D/TempFolder/Model_after_pruning.pt",
    remove_prune_info=True,
):

    global_prune_eval = {}

    for prune_amt in prune_amt_list:

        print(f"\nPruning {prune_amt*100}%")
        torch.cuda.empty_cache()

        # Loading model

        model = loadModel(model_pt_path, model_class=Net_1080(), pth=False)
        print(f"Loaded model from {model_pt_path}")
        model.eval()

        # colleting parameters to prune
        parameters_to_prune = update_parameters_to_prune(model)

        # Pruning
        model = global_prune(
            model, prune_amt, parameters_to_prune, remove_prune_info=remove_prune_info
        )
        # save pruned model
        saveModel(model, model_temp_path)

        # load the pruned model
        model = loadModel(model_temp_path, model_class=Net_1080(), pth=False)
        model.eval()

        print("Calculating the required values..")
        global_prune_eval["memory_dense_" + str(int(prune_amt * 100))] = model_size(
            model_temp_path, print_value=False, return_value=True
        )

        global_prune_eval["bitwise_dense_" + str(int(prune_amt * 100))] = (
            calculate_bitwise_dense(parameters_to_prune)
        )

        (
            global_prune_eval["bitwise_sparse_64_" + str(int(prune_amt * 100))],
            global_prune_eval["bitwise_sparse_16_" + str(int(prune_amt * 100))],
            global_prune_eval["bitwise_sparse_8_" + str(int(prune_amt * 100))],
            spT_dict_64,
            spT_dict_16,
            spT_dict_8,
        ) = calculate_bitwise_sparse(parameters_to_prune)

        spT_dict_path_64 = (
            f"/home/pytorch/personal/Deep3D/SelfDeep3D/TempFolder/spT_dict64.pt"
        )

        torch.save(spT_dict_64, spT_dict_path_64)

        global_prune_eval["memory_sparse_64_" + str(int(prune_amt * 100))] = model_size(
            spT_dict_path_64, print_value=False, return_value=True
        )

        spT_dict_path_16 = (
            f"/home/pytorch/personal/Deep3D/SelfDeep3D/TempFolder/spT_dict16.pt"
        )

        torch.save(spT_dict_16, spT_dict_path_16)

        global_prune_eval["memory_sparse_16_" + str(int(prune_amt * 100))] = model_size(
            spT_dict_path_16, print_value=False, return_value=True
        )

        spT_dict_path_8 = (
            f"/home/pytorch/personal/Deep3D/SelfDeep3D/TempFolder/spT_dict8.pt"
        )

        torch.save(spT_dict_8, spT_dict_path_8)

        global_prune_eval["memory_sparse_8_" + str(int(prune_amt * 100))] = model_size(
            spT_dict_path_8, print_value=False, return_value=True
        )
        print("Required values calculated..")
        print("Calculating SSIM over frames..")

        ssim_l_r, ssim_pred_r, ssim_r1_r = ssimAll_over_time(
            train_L0,
            train_R0,
            train_R1,
            model,
            stacked=False,
            show_plot=False,
            print_values=False,
        )

        _, acc, _ = avg_valuesPrint(
            ssim_l_r, ssim_pred_r, ssim_r1_r, print_values=False, return_values=True
        )

        global_prune_eval["accuray_" + str(int(prune_amt * 100))] = acc

        print("SSIM calculated..")

    return global_prune_eval


###################################### Local Pruning ########################################


def local_prune(
    model,
    conv_prune_amt,
    rest_prune_amt,
    parameters_to_prune=None,
    remove_prune_info=True,
):

    if parameters_to_prune is None:
        raise ValueError("parameters_to_prune cannot be None")

    for module, param in parameters_to_prune:
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            prune.l1_unstructured(module, name=param, amount=conv_prune_amt)
        else:
            prune.l1_unstructured(module, name=param, amount=rest_prune_amt)

    if remove_prune_info:
        for mod, param in parameters_to_prune:
            prune.remove(mod, param)

    return model


def local_prune_list(
    Conv_prune_amt_list,
    rest_prune_amt_list,
    model_pt_path,
    train_L0,
    train_R0,
    train_R1,
    model_temp_path="/home/pytorch/personal/Deep3D/SelfDeep3D/TempFolder/Model_after_pruning.pt",
    remove_prune_info=True,
):

    local_prune_eval = {}

    if len(Conv_prune_amt_list) != len(rest_prune_amt_list):
        raise ValueError(
            "Length of Conv_prune_amt_list and rest_prune_amt_list should be same"
        )

    for i, prune_amt in enumerate(Conv_prune_amt_list):

        print("\n")
        print(f"Pruning Conv layers by {prune_amt * 100}%")
        print(f"Pruning rest of the layers by {rest_prune_amt_list[i] * 100}%")

        torch.cuda.empty_cache()

        # Loading model

        model = loadModel(model_pt_path, model_class=Net_1080(), pth=False)
        print(f"Loaded model from {model_pt_path}")
        model.eval()

        # colleting parameters to prune
        parameters_to_prune = update_parameters_to_prune(model)

        # Pruning
        model = local_prune(
            model,
            prune_amt,
            rest_prune_amt_list[i],
            parameters_to_prune,
            remove_prune_info,
        )
        # save pruned model
        saveModel(model, model_temp_path)

        # load the pruned model
        model = loadModel(model_temp_path, model_class=Net_1080(), pth=False)
        model.eval()

        print("Calculating the required values..")
        local_prune_eval[
            "memory_dense_"
            + str(int(prune_amt * 100))
            + str(int(rest_prune_amt_list[i] * 100))
        ] = model_size(model_temp_path, print_value=False, return_value=True)

        local_prune_eval[
            "bitwise_dense_"
            + str(int(prune_amt * 100))
            + str(int(rest_prune_amt_list[i] * 100))
        ] = calculate_bitwise_dense(parameters_to_prune)

        (
            local_prune_eval[
                "bitwise_sparse_64_"
                + str(int(prune_amt * 100))
                + str(int(rest_prune_amt_list[i] * 100))
            ],
            local_prune_eval[
                "bitwise_sparse_16_"
                + str(int(prune_amt * 100))
                + str(int(rest_prune_amt_list[i] * 100))
            ],
            local_prune_eval[
                "bitwise_sparse_8_"
                + str(int(prune_amt * 100))
                + str(int(rest_prune_amt_list[i] * 100))
            ],
            spT_dict_64,
            spT_dict_16,
            spT_dict_8,
        ) = calculate_bitwise_sparse(parameters_to_prune)

        spT_dict_path_64 = (
            f"/home/pytorch/personal/Deep3D/SelfDeep3D/TempFolder/spT_dict64.pt"
        )

        torch.save(spT_dict_64, spT_dict_path_64)

        local_prune_eval[
            "memory_sparse_64_"
            + str(int(prune_amt * 100))
            + str(int(rest_prune_amt_list[i] * 100))
        ] = model_size(spT_dict_path_64, print_value=False, return_value=True)

        spT_dict_path_16 = (
            f"/home/pytorch/personal/Deep3D/SelfDeep3D/TempFolder/spT_dict16.pt"
        )

        torch.save(spT_dict_16, spT_dict_path_16)

        local_prune_eval[
            "memory_sparse_16_"
            + str(int(prune_amt * 100))
            + str(int(rest_prune_amt_list[i] * 100))
        ] = model_size(spT_dict_path_16, print_value=False, return_value=True)

        spT_dict_path_8 = (
            f"/home/pytorch/personal/Deep3D/SelfDeep3D/TempFolder/spT_dict8.pt"
        )

        torch.save(spT_dict_8, spT_dict_path_8)

        local_prune_eval[
            "memory_sparse_8_"
            + str(int(prune_amt * 100))
            + str(int(rest_prune_amt_list[i] * 100))
        ] = model_size(spT_dict_path_8, print_value=False, return_value=True)
        print("Required values calculated..")
        print("Calculating SSIM over frames..")

        ssim_l_r, ssim_pred_r, ssim_r1_r = ssimAll_over_time(
            train_L0,
            train_R0,
            train_R1,
            model,
            stacked=False,
            show_plot=False,
            print_values=False,
        )

        _, acc, _ = avg_valuesPrint(
            ssim_l_r, ssim_pred_r, ssim_r1_r, print_values=False, return_values=True
        )

        local_prune_eval[
            "accuray_"
            + str(int(prune_amt * 100))
            + str(int(rest_prune_amt_list[i] * 100))
        ] = acc

        print("SSIM calculated..")

    return local_prune_eval


# # Print parameter names and requires_grad values
# def global_parameters_to_prune(model, weights = True, bias = False):
#     try_dict = {}
#     parameters_to_prune = []
#     for idx,(name, param) in enumerate(model.named_parameters()):

#         if  param.requires_grad and "weight" in name and weights:
#             if len(name.split(".")) == 3:
#                 try_dict["layer_weight_"+str(idx)] = getattr(model, name.split(".")[0])[int(name.split(".")[1])]
#                 parameters_to_prune.append(tuple([try_dict["layer_weight_"+str(idx)], 'weight']))

#             else:
#                 try_dict["layer_weight_"+str(idx)] = getattr(model, name.split(".")[0])
#                 parameters_to_prune.append(tuple([try_dict["layer_weight_"+str(idx)], 'weight']))

#         if param.requires_grad and "bias" in name and bias:
#             if len(name.split(".")) == 3:
#                 try_dict["layer_bias_"+str(idx)] = getattr(model, name.split(".")[0])[int(name.split(".")[1])]
#                 parameters_to_prune.append(tuple([try_dict["layer_bias_"+str(idx)], 'bias']))

#             else:
#                 try_dict["layer_bias_"+str(idx)] = getattr(model, name.split(".")[0])
#                 parameters_to_prune.append(tuple([try_dict["layer_bias_"+str(idx)], 'bias']))


#     parameters_to_prune = tuple(parameters_to_prune)
#     return parameters_to_prune, try_dict
# USE : parameters_to_prune, try_dict = global_parameters_to_prune(model, weights = True, bias = False)
