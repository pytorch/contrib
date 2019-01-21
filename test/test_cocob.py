import unittest
import re
import functools
from copy import deepcopy
import torch
from torch.autograd import Variable
from torch import sparse
from torch import optim
from torch import nn
import torchcontrib.optim as contriboptim
from common import TestCase, run_tests
from torch.utils import data
from test_optim import TestOptim

class TestCocob(TestOptim):

    # Test CocobBackprop

    def test_cocob_backprop(self):
        self._test_basic_cases(lambda weight, bias: contriboptim.CocobBackprop([weight, bias]))
        self._test_rosenbrock(lambda params: contriboptim.CocobBackprop(params))

if __name__ == '__main__':
    unittest.main()
