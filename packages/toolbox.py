import torch
from transformers import SwitchTransformersModel
import pandas as pd
import numpy as np

def print_model_size(model: torch.nn.Module):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print("Model size: {:.3f}MB".format(size_all_mb))


from auto_gptq.nn_modules import qlinear

qlinear_types = {
    torch.nn.Linear,
    qlinear.qlinear_cuda.QuantLinear,
    qlinear.qlinear_cuda_old.QuantLinear,
    qlinear.qlinear_exllama.QuantLinear,
    qlinear.qlinear_marlin.QuantLinear,
    qlinear.qlinear_qigen.QuantLinear,
    qlinear.qlinear_triton.QuantLinear,
}


def print_layer_size(model: torch.nn.Module):
    # Print the size of the projection layers
    proj_set = {}
    for module in model.named_modules():
        if any(isinstance(module[1], cls) for cls in qlinear_types):
            proj = module[0].split(".")[-1]
            proj_set[proj] = module[1]

    proj_set.keys()
    for attr, w in proj_set.items():
        if hasattr(model, "quantize_config"):
            print(
                f"{attr}: weight: {w.qweight.nbytes} bytes\n quant params: {w.qzeros.nbytes}, {w.scales.nbytes}, {w.g_idx.nbytes} bytes"
            )

        else:
            print(f"{attr}: {w.weight.nbytes} bytes")
    return proj_set


def get_layer_weights(
    model: torch.nn.Module,
    layer_list: list[int] = [],
    neglect_list: list[str] = ["lm_head"],
    flatten: bool = True
) -> pd.DataFrame:

    if issubclass(type(model), SwitchTransformersModel):
        return get_layer_weights_ST(model, [0, 1], neglect_list)

    # Get the weights of the projection layers
    # Do not collect the weights of the layers in the neglect_list
    proj_set = {}
    for name, param in model.named_parameters():
        if any(substring in name for substring in neglect_list):
            continue
        proj_name = name.split(".")[-2]
        proj_set[proj_name] = True

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList):
            layer_modulelist = module

    # if layer_modulelist is not defined, raise an error
    try:
        if len(layer_list) == 0:
            layer_list = list(range(len(layer_modulelist)))
        else:
            layer_modulelist = [layer_modulelist[i] for i in layer_list if i < len(layer_modulelist)]

    except NameError:
        raise ValueError("layer_modulelist is not defined. Please check the model structure.")

    df = pd.DataFrame(index=layer_list, columns=list(proj_set.keys()))
    for i in layer_list:
        for name, param in layer_modulelist[i].named_parameters():
            layername = name.split(".")[-2]
            if layername in proj_set.keys():
                w = param.detach().cpu().numpy()
                # If there's data in the dataframe, concatenate the data.
                try:
                    concat_data = np.concatenate(
                        (df[layername][i], w.flatten() if flatten else w), axis=0
                    )
                    df.loc[i, layername] = concat_data
                except:
                    df.loc[i, layername] = w.flatten() if flatten else w

    return df

def get_layer_weights_ST(
    model: SwitchTransformersModel,
    exportEncoder: bool = False,
    exportDecoder: bool = True,
    layer_list: list[int] = [0, 1],
    neglect_list: list[str] = ["q", "k", "v", "o", "classifier"],
) -> pd.DataFrame:
    proj_set = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            proj = name.split(".")[-1]
            if proj in neglect_list:
                continue
            proj_set[proj] = True

    df = pd.DataFrame(index=layer_list, columns=["mlp"])
    
    # If only one of the encoder or decoder is to be exported,
    # it can be exported with single dataframe. 
    if exportEncoder ^ exportDecoder:
        block = model.get_encoder().block if exportEncoder else model.get_decoder().block
    else:
        raise NotImplementedError("Both encoder and decoder export is not implemented.")
        

    for block_idx in layer_list:
        for name, module in block[block_idx].layer.named_modules():
            if "SelfAttention" in name.split('.')[-1]:
                attention = module
            elif "EncDecAttention" in name.split(".")[-1]:
                x_attention = module
            elif "mlp" in name.split(".")[-1]:
                mlp = module

        # Let's focus on the FF Layer.
        try:
            flattened_mlp = np.array([])
            for name, module in mlp.named_modules():
                if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in proj_set.keys():
                    flattened_mlp = np.concatenate(
                        (flattened_mlp,
                         module.weight.detach().cpu().numpy().flatten()),
                        axis=0,
                    )
            df.loc[block_idx, "mlp"] = flattened_mlp
        except NameError:
            raise ValueError("mlp is not defined. Please check the model structure.")

    return df