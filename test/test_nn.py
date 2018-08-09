import unittest
import torch
import torchcontrib
import torchcontrib.nn as contrib_nn
import torchcontrib.nn.functional as contrib_F
from torch.autograd import gradcheck, gradgradcheck
from common import run_tests, TestCase


class TestNN(TestCase):
    def assertGradAndGradgradChecks(self, apply_fn, inputs):
        # call assert function rather than returning a bool since it's nicer
        # if we get whether this failed on the gradcheck or the gradgradcheck.
        self.assertTrue(gradcheck(apply_fn, inputs))
        self.assertTrue(gradgradcheck(apply_fn, inputs))

    def test_film(self):
        m = contrib_nn.FiLM()
        input_1d = torch.randn(4, 10, 2, requires_grad=True)
        input_2d = torch.randn(4, 10, 2, 2, requires_grad=True)
        input_3d = torch.randn(4, 10, 2, 2, 2, requires_grad=True)
        ones = torch.ones(4, 10)
        zeros = torch.zeros(4, 10)
        half_ones_half_zeros = torch.cat([torch.ones(4, 5), torch.zeros(4, 5)], 1)
        half_ones_half_neg_ones = torch.cat([torch.ones(4, 5), torch.full((4, 5), -1)], 1)
        for inp in [input_1d, input_2d, input_3d]:
            self.assertGradAndGradgradChecks(lambda x: contrib_F.film(x, ones, ones), (inp,))
            output = m(inp, ones, zeros)
            self.assertEqual(contrib_F.film(inp, ones, zeros), output)
            self.assertEqual(inp, output)
            output = m(inp, zeros, ones)
            self.assertEqual(contrib_F.film(inp, zeros, ones), output)
            self.assertEqual(torch.ones_like(output), output)
            output = m(inp, -2 * ones, 3 * ones)
            self.assertEqual(contrib_F.film(inp, -2 * ones, 3 * ones), output)
            self.assertEqual((-2 * inp) + 3, output)
            output = m(inp, half_ones_half_zeros, half_ones_half_neg_ones)
            self.assertEqual(contrib_F.film(inp, half_ones_half_zeros, half_ones_half_neg_ones), output)
            self.assertEqual(output.sum(), inp[:, :5].sum())


if __name__ == '__main__':
    run_tests()
