import torch
def build_network(cfg):
    name = cfg.model 
    if name == 'base':
        from .BaseModel.network import network
    elif name == 'swin3d':
        from .Swin.network import network
    else:
        raise ValueError(f"{name} is not a valid model!")

    return network(cfg[name])
