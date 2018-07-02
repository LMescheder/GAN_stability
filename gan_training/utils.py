
import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision


def save_images(imgs, outfile, nrow=8):
    imgs = imgs / 2 + 0.5     # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow)


def get_nsamples(data_loader, N):
    x = []
    y = []
    n = 0
    while n < N:
        x_next, y_next = next(iter(data_loader))
        x.append(x_next)
        y.append(y_next)
        n += x_next.size(0)
    x = torch.cat(x, dim=0)[:N]
    y = torch.cat(y, dim=0)[:N]
    return x, y


def update_average(model_tgt, model_src, beta):
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
