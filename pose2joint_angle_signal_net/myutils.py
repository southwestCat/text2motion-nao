import torch

def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in tensor: {name}\nTensor value:\n{tensor}")
    

def check_for_nan_inf(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"NaN/Inf detected in tensor: {name}\nTensor value:\n{tensor}")
    

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_min = param.grad.min().item()
            grad_max = param.grad.max().item()
            print(f"Gradient of {name}: min={grad_min}, max={grad_max}")
            check_for_nan_inf(param.grad, f"Gradient of {name}")


def check_model_paramereters(model):
    for name, param in model.named_parameters():
        param_min = param.min().item()
        param_max = param.max().item()
        print(f"Parameter of {name}: min={param_min}, max={param_max}")
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(f"NaN/Inf detected in parameter: {name}")


def check_model_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(f"NaN/Inf detected in parameter: {name}")
