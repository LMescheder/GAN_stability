import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed


class Generator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        s0 = self.s0 = size // 64
        nf = self.nf = nfilter
        self.z_dim = z_dim

        # Submodules
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, 32*nf*s0*s0)

        self.resnet_0_0 = ResnetBlock(32*nf, 16*nf)
        self.resnet_1_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_2_0 = ResnetBlock(16*nf, 8*nf)
        self.resnet_3_0 = ResnetBlock(8*nf, 4*nf)
        self.resnet_4_0 = ResnetBlock(4*nf, 2*nf)
        self.resnet_5_0 = ResnetBlock(2*nf, 1*nf)
        self.conv_img = nn.Conv2d(nf, 3, 7, padding=3)

    def forward(self, z, y):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)

        yembed = self.embedding(y)
        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, 32*self.nf, self.s0, self.s0)

        out = self.resnet_0_0(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_1_0(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_2_0(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_3_0(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_4_0(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_5_0(out)

        out = F.interpolate(out, scale_factor=2)

        out = self.conv_img(actvn(out))
        out = torch.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = size // 64
        nf = self.nf = nfilter

        # Submodules
        self.conv_img = nn.Conv2d(3, 1*nf, 7, padding=3)

        self.resnet_0_0 = ResnetBlock(1*nf, 2*nf)
        self.resnet_1_0 = ResnetBlock(2*nf, 4*nf)
        self.resnet_2_0 = ResnetBlock(4*nf, 8*nf)
        self.resnet_3_0 = ResnetBlock(8*nf, 16*nf)
        self.resnet_4_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_5_0 = ResnetBlock(16*nf, 32*nf)

        self.fc = nn.Linear(32*nf*s0*s0, nlabels)

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.conv_img(x)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_0_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_2_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_4_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_5_0(out)

        out = out.view(batch_size, 32*self.nf*self.s0*self.s0)
        out = self.fc(actvn(out))

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        out = out[index, y]

        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out
