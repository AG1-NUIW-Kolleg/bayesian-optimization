from AD_Hill_System_HMC_Py import observe_blackbox_simulation
import jax
import torch


def HillType(x):
    x = x.numpy().squeeze()
    data = observe_blackbox_simulation(x)
    data = jax.device_get(data)
    print(data)
    return torch.tensor([[(data[2]-data[3])]]).to(torch.double)
