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
    try:
        del proj_set["lm_head"]
    except KeyError:
        pass
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
    model: torch.nn.Module, neglect_list: list[str] = ["lm_head"]
) -> pd.DataFrame:
    
    if issubclass(type(model), SwitchTransformersModel):
        return get_ST_layer_weights(model, [0, 1], neglect_list)
    
    # Get the weights of the projection layers
    proj_set = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            proj = name.split(".")[-1]
            if proj in neglect_list:
                continue
            proj_set[proj] = True
        elif isinstance(module, torch.nn.ModuleList):
            layerlist = module

    # if layerlist is not defined, raise an error
    try:
        layerlist
    except NameError:
        raise ValueError("layerlist is not defined. Please check the model structure.")

    df = pd.DataFrame(index=range(len(layerlist)), columns=list(proj_set.keys()))
    for i, layer in enumerate(layerlist):
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in proj_set.keys():
                w = module.weight.detach().numpy()
                df.loc[i, name.split(".")[-1]] = w

    return df

def get_ST_layer_weights(
    model: SwitchTransformersModel,
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

    decoder = model.get_decoder()
    block = decoder.block
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