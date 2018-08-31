import torch
from torch.nn import Module
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
