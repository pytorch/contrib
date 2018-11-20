import torch
import torch.nn as nn
import torch.nn.functional as F

class SSP(nn.Softplus):
    r"""Applies the element-wise function:

    .. math::
        \text{SSP}(x) = \text{Softplus}(x) - \text{Softplus}(0)

    The default SSP looks like ELU(alpha=log2).

    Args:
        beta: the :math:`\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::
        >>> m = torchcontrib.nn.SSP()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__(beta, threshold)
        self.sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()

    def forward(self, input):
        return F.softplus(input, self.beta, self.threshold) - self.sp0
