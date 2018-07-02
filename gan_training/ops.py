from torch import nn
import torch
from torch.nn import Parameter


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(
                torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        made_params = (
            hasattr(self.module, self.name + "_u")
            and hasattr(self.module, self.name + "_v")
            and hasattr(self.module, self.name + "_bar")
        )
        return made_params

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class CBatchNorm(nn.Module):
    def __init__(self, nfilter, nlabels):
        super().__init__()
        # Attributes
        self.nlabels = nlabels
        self.nfilter = nfilter
        # Submodules
        self.alpha_embedding = nn.Embedding(nlabels, nfilter)
        self.beta_embedding = nn.Embedding(nlabels, nfilter)
        self.bn = nn.BatchNorm2d(nfilter, affine=False)
        # Initialize
        nn.init.constant_(self.alpha_embedding.weight, 1.)
        nn.init.constant_(self.beta_embedding.weight, 0.)

    def forward(self, x, y):
        dim = len(x.size())
        batch_size = x.size(0)
        assert(dim >= 2)
        assert(x.size(1) == self.nfilter)

        s = [batch_size, self.nfilter] + [1] * (dim - 2)
        alpha = self.alpha_embedding(y)
        alpha = alpha.view(s)
        beta = self.beta_embedding(y)
        beta = beta.view(s)

        out = self.bn(x)
        out = alpha * out + beta

        return out


class CInstanceNorm(nn.Module):
    def __init__(self, nfilter, nlabels):
        super().__init__()
        # Attributes
        self.nlabels = nlabels
        self.nfilter = nfilter
        # Submodules
        self.alpha_embedding = nn.Embedding(nlabels, nfilter)
        self.beta_embedding = nn.Embedding(nlabels, nfilter)
        self.bn = nn.InstanceNorm2d(nfilter, affine=False)
        # Initialize
        nn.init.uniform(self.alpha_embedding.weight, -1., 1.)
        nn.init.constant_(self.beta_embedding.weight, 0.)

    def forward(self, x, y):
        dim = len(x.size())
        batch_size = x.size(0)
        assert(dim >= 2)
        assert(x.size(1) == self.nfilter)

        s = [batch_size, self.nfilter] + [1] * (dim - 2)
        alpha = self.alpha_embedding(y)
        alpha = alpha.view(s)
        beta = self.beta_embedding(y)
        beta = beta.view(s)

        out = self.bn(x)
        out = alpha * out + beta

        return out
