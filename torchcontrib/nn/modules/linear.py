import torch

import torch.nn as nn
from torch.nn import Module

import torch.nn.functional as functional
from .. import functional as F

class FiLM(Module):
    r"""Applies Feature-wise Linear Modulation to the incoming data as described
    in the paper `FiLM: Visual Reasoning with a General Conditioning Layer`_ .

     .. math::
        y_{n,c,*} = \gamma_{n, c} * x_{n,c,*} + \beta_{n,c},

    where :math:`\gamma_{n,c}` and :math:`\beta_{n,c}` are scalars and
    operations are broadcast over any additional dimensions of :math:`x`

     Shape:
        - Input: :math:`(N, C, *)` where :math:`*` means any number of additional
          dimensions
        - Gammas: :math:`(N, C)`
        - Betas: :math:`(N, C)`
        - Output: :math:`(N, C, *)`, same shape as the input

     Examples::
        >>> m = torchcontrib.nn.FiLM()
        >>> input = e
        >>> gamma = torch.randn(20)
        >>> beta = torch.randn(20)
        >>> output = m(input, gamma, beta)
        >>> output.size()
        torch.Size([128, 20, 4, 4])

     .. _`FiLM: Visual Reasoning with a General Conditioning Layer`:
        https://arxiv.org/abs/1709.07871
    """
    def forward(self, input, gamma, beta):
        return F.film(input, gamma, beta)

# partially from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
class SE(Module):
    r"""Applies Squeeze and Excitation to the incoming data as described
    in the paper `Squeeze-and-Excitation Networks`_.
    
    Args:
        in_ch (int): Number of channels in the input tensor
        r (int): Reduction ratio of the SE block. Default: 16

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)`, same shape as input

    Examples::
        >>> input = torch.randn(1, 3, 256, 256)
        >>> se = SE(in_ch=3, r=16)
        >>> out = se(input)
        >>> out.size()
        torch.Size([1, 3, 256, 256])
    """
    def __init__(self, in_ch, r=16):
        super(SE, self).__init__()
        
        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)
    
    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]),-1).mean(-1)
        x = functional.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        
        return x