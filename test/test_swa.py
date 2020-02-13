import re
import functools
from copy import deepcopy
import torch
from torch.autograd import Variable
from torch import sparse
from torch import optim
from torch import nn
import torchcontrib.optim as contriboptim
from .common import TestCase, run_tests
from torch.utils import data
from test_optim import rosenbrock, drosenbrock
from test_optim import TestOptim


class TestSWA(TestOptim):
    # Pavel: I slightly update the _test_... functions to (1) remove the
    # legacy-related parts and (2) add oprimizer.swap_swa_sgd() in the end of
    # optimization

    def _test_rosenbrock(self, constructor, automode=True):
        # automode shows wether we need to update SWA params manually

        params = torch.tensor([1.5, 1.5], requires_grad=True)
        optimizer = constructor([params])

        solution = torch.tensor([1., 1.])
        initial_dist = params.data.dist(solution)

        def eval():
            # SWA
            optimizer.zero_grad()
            loss = rosenbrock(params)
            loss.backward()
            # loss.backward() will give **slightly** different
            # gradients, than drosenbtock, because of a different ordering
            # of floating point operations. In most cases it doesn't matter,
            # but some optimizers are so sensitive that they can temporarily
            # diverge up to 1e-4, just to converge again. This makes the
            # comparison more stable.
            params.grad.data.copy_(drosenbrock(params.data))
            return loss

        for i in range(2000):
            optimizer.step(eval)
            if not automode:
                optimizer.update_swa()
        optimizer.swap_swa_sgd()

        self.assertLessEqual(params.data.dist(solution), initial_dist)

    def _test_state_dict(self, weight, bias, input, constructor):
        weight = weight.requires_grad_()
        bias = bias.requires_grad_()

        def fn_base(optimizer, weight, bias):
            optimizer.zero_grad()
            i = input_cuda if weight.is_cuda else input
            loss = (weight.mv(i) + bias).pow(2).sum()
            loss.backward()
            return loss

        optimizer = constructor(weight, bias)
        fn = functools.partial(fn_base, optimizer, weight, bias)

        # Prime the optimizer
        optimizer.update_swa()
        for i in range(20):
            optimizer.step(fn)
        # Clone the weights and construct new optimizer for them
        weight_c = weight.detach().clone().requires_grad_()
        bias_c = bias.detach().clone().requires_grad_()
        optimizer_c = constructor(weight_c, bias_c)
        fn_c = functools.partial(fn_base, optimizer_c, weight_c, bias_c)
        # Load state dict
        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_c.load_state_dict(state_dict_c)
        self.assertEqual(optimizer.optimizer.state_dict(), optimizer_c.optimizer.state_dict())
        # Run both optimizations in parallel
        for i in range(20):
            optimizer.optimizer.step(fn)
            optimizer_c.optimizer.step(fn_c)
            self.assertEqual(weight, weight_c)
            self.assertEqual(bias, bias_c)
            # check that averages also coincide
            optimizer.swap_swa_sgd()
            optimizer_c.swap_swa_sgd()
            self.assertEqual(weight, weight_c)
            self.assertEqual(bias, bias_c)
            optimizer.swap_swa_sgd()
            optimizer_c.swap_swa_sgd()
        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)

        # Check that state dict can be loaded even when we cast parameters
        # to a different type and move to a different device.
        if not torch.cuda.is_available():
            return

        input_cuda = Variable(input.data.float().cuda())
        weight_cuda = Variable(weight.data.float().cuda(), requires_grad=True)
        bias_cuda = Variable(bias.data.float().cuda(), requires_grad=True)
        optimizer_cuda = constructor(weight_cuda, bias_cuda)
        fn_cuda = functools.partial(fn_base, optimizer_cuda, weight_cuda, bias_cuda)

        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_cuda.load_state_dict(state_dict_c)

        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)

        for i in range(20):
            optimizer.step(fn)
            optimizer_cuda.step(fn_cuda)
            self.assertEqual(weight, weight_cuda)
            self.assertEqual(bias, bias_cuda)

    # Test SWA

    def test_swa(self):
        def sgd_constructor(params):
            sgd = optim.SGD(params, lr=1e-3)
            return contriboptim.SWA(
                sgd, swa_start=1000, swa_freq=1, swa_lr=1e-3)

        def sgd_manual_constructor(params):
            sgd = optim.SGD(params, lr=1e-3)
            return contriboptim.SWA(sgd)

        def sgd_momentum_constructor(params):
            sgd = optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=1e-4)
            return contriboptim.SWA(
                sgd, swa_start=1000, swa_freq=1, swa_lr=1e-3)

        def adam_constructor(params):
            adam = optim.Adam(params, lr=1e-2)
            return contriboptim.SWA(
                adam, swa_start=1000, swa_freq=1, swa_lr=1e-2)

        def adadelta_constructor(params):
            adadelta = optim.Adadelta(params)
            return contriboptim.SWA(
                adadelta, swa_start=1000, swa_freq=1)

        def adagrad_constructor(params):
            adagrad = optim.Adagrad(params, lr=1e-1)
            return contriboptim.SWA(
                adagrad, swa_start=1000, swa_freq=1, swa_lr=1e-2)

        def adamax_constructor(params):
            adamax = optim.Adamax(params, lr=1e-1)
            return contriboptim.SWA(
                adamax, swa_start=1000, swa_freq=1, swa_lr=1e-2)

        def rmsprop_constructor(params):
            rmsprop = optim.RMSprop(params, lr=1e-2)
            return contriboptim.SWA(
                rmsprop, swa_start=1000, swa_freq=1, swa_lr=1e-3)

        def rprop_constructor(params):
            rprop = optim.Rprop(params, lr=1e-2)
            return contriboptim.SWA(
                rprop, swa_start=1000, swa_freq=1, swa_lr=1e-3)

        def asgd_constructor(params):
            asgd = optim.ASGD(params, lr=1e-3)
            return contriboptim.SWA(
                asgd, swa_start=1000, swa_freq=1, swa_lr=1e-3)

        def lbfgs_constructor(params):
            lbfgs = optim.LBFGS(params, lr=5e-2, max_iter=5)
            return contriboptim.SWA(
                lbfgs, swa_start=1000, swa_freq=1, swa_lr=1e-3)

        auto_constructor_list = [sgd_constructor, sgd_momentum_constructor,
                                 adam_constructor, adadelta_constructor,
                                 adagrad_constructor, adamax_constructor,
                                 rmsprop_constructor, rprop_constructor,
                                 asgd_constructor, lbfgs_constructor]

        for i, constructor in enumerate(auto_constructor_list):
            self._test_rosenbrock(constructor)
            self._test_basic_cases(
                lambda weight, bias: constructor([weight, bias]),
                ignore_multidevice=(constructor == lbfgs_constructor)
            )
            if i < len(auto_constructor_list) - 1:
                self._test_basic_cases(
                    lambda weight, bias: constructor(
                        self._build_params_dict(weight, bias, lr=1e-2)))
                self._test_basic_cases(
                    lambda weight, bias: constructor(
                        self._build_params_dict_single(weight, bias, lr=1e-2)))

        self._test_rosenbrock(sgd_manual_constructor, automode=False)

    def _define_vars_loss_opt(self):
        x = Variable(torch.Tensor([5., 2.]), requires_grad=True)
        y = Variable(torch.Tensor([3., 7.]), requires_grad=True)

        def loss_fun(a, b):
            return torch.sum(a * b)**2

        opt = optim.SGD([{'params': [x]},
                        {'params': [y], 'lr': 1e-3}], lr=1e-2, momentum=0.9)
        return x, y, loss_fun, opt

    @staticmethod
    def _update_test_vars(i, swa_freq, swa_start, n_avg, x_sum, y_sum, x, y, upd_fun):
        if i % swa_freq == 0 and i > swa_start:
            upd_fun()
            n_avg += 1
            x_sum += x.data
            y_sum += y.data
        return n_avg, x_sum, y_sum

    def test_swa_auto(self):
        # Tests SWA in Auto mode: values of x and y after opt.swap_swa_sgd()
        # should be equal to the manually computed averages
        x, y, loss_fun, opt = self._define_vars_loss_opt()
        swa_start = 5
        swa_freq = 2
        opt = contriboptim.SWA(opt, swa_start=swa_start, swa_freq=swa_freq, swa_lr=0.001)

        x_sum = torch.zeros_like(x)
        y_sum = torch.zeros_like(y)
        n_avg = 0
        for i in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            n_avg, x_sum, y_sum = self._update_test_vars(
                i, swa_freq, swa_start, n_avg, x_sum, y_sum, x, y,
                upd_fun=lambda: None)

        opt.swap_swa_sgd()
        x_avg = x_sum / n_avg
        y_avg = y_sum / n_avg
        self.assertEqual(x_avg, x)
        self.assertEqual(y_avg, y)

    def test_swa_manual(self):
        # Tests SWA in manual mode: values of x and y after opt.swap_swa_sgd()
        # should be equal to the manually computed averages
        x, y, loss_fun, opt = self._define_vars_loss_opt()
        opt = contriboptim.SWA(opt)
        swa_start = 5
        swa_freq = 2

        x_sum = torch.zeros_like(x)
        y_sum = torch.zeros_like(y)
        n_avg = 0
        for i in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            n_avg, x_sum, y_sum = self._update_test_vars(
                i, swa_freq, swa_start, n_avg, x_sum, y_sum, x, y,
                upd_fun=opt.update_swa)

        opt.swap_swa_sgd()
        x_avg = x_sum / n_avg
        y_avg = y_sum / n_avg
        self.assertEqual(x_avg, x)
        self.assertEqual(y_avg, y)

    def test_swa_manual_group(self):
        # Tests SWA in manual mode with only y param group updated:
        # value of x should not change after opt.swap_swa_sgd() and y should
        # be equal to the manually computed average
        x, y, loss_fun, opt = self._define_vars_loss_opt()
        opt = contriboptim.SWA(opt)
        swa_start = 5
        swa_freq = 2

        y_sum = torch.zeros_like(y)
        n_avg = 0
        for i in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            n_avg, _, y_sum = self._update_test_vars(
                i, swa_freq, swa_start, n_avg, 0, y_sum, x, y,
                upd_fun=lambda: opt.update_swa_group(opt.param_groups[1]))

        x_before_swap = x.data.clone()

        with self.assertWarnsRegex(re.escape(r"SWA wasn't applied to param {}".format(x))):
            opt.swap_swa_sgd()

        y_avg = y_sum / n_avg
        self.assertEqual(y_avg, y)
        self.assertEqual(x_before_swap, x)

    def test_swa_auto_group_added_during_run(self):
        # Tests SWA in Auto mode with the second param group added after several
        # optimizations steps. The expected behavior is that the averaging for
        # the second param group starts at swa_start steps after it is added.
        # For the first group averaging should start swa_start steps after the
        # first step of the optimizer.

        x, y, loss_fun, _ = self._define_vars_loss_opt()
        opt = optim.SGD([x], lr=1e-3, momentum=0.9)
        swa_start = 5
        swa_freq = 2
        opt = contriboptim.SWA(opt, swa_start=swa_start, swa_freq=swa_freq, swa_lr=0.001)

        x_sum = torch.zeros_like(x)
        y_sum = torch.zeros_like(y)
        x_n_avg = 0
        y_n_avg = 0
        x_step = 0
        for i in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            x_step += 1
            if i % swa_freq == 0 and i > swa_start:
                x_n_avg += 1
                x_sum += x.data

        x_avg = x_sum / x_n_avg

        opt.add_param_group({'params': y, 'lr': 1e-4})

        for y_step in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            x_step += 1
            if y_step % swa_freq == 0 and y_step > swa_start:
                y_n_avg += 1
                y_sum += y.data
            if x_step % swa_freq == 0 and x_step > swa_start:
                x_n_avg += 1
                x_sum += x.data
                x_avg = x_sum / x_n_avg

        opt.swap_swa_sgd()
        x_avg = x_sum / x_n_avg
        y_avg = y_sum / y_n_avg
        self.assertEqual(x_avg, x)
        self.assertEqual(y_avg, y)

    def test_swa_lr(self):
        # Tests SWA learning rate: in auto mode after swa_start steps the
        # learning rate should be changed to swa_lr; in manual mode swa_lr
        # must be ignored

        # Auto mode
        x, y, loss_fun, opt = self._define_vars_loss_opt()
        swa_start = 5
        swa_freq = 2
        initial_lr = opt.param_groups[0]["lr"]
        swa_lr = initial_lr * 0.1
        opt = contriboptim.SWA(opt, swa_start=swa_start, swa_freq=swa_freq, swa_lr=swa_lr)

        for i in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            lr = opt.param_groups[0]["lr"]
            if i > swa_start:
                self.assertEqual(lr, swa_lr)
            else:
                self.assertEqual(lr, initial_lr)

        # Manual Mode
        x, y, loss, opt = self._define_vars_loss_opt()
        initial_lr = opt.param_groups[0]["lr"]
        swa_lr = initial_lr * 0.1
        with self.assertWarnsRegex("Some of swa_start, swa_freq is None"):
            opt = contriboptim.SWA(opt, swa_lr=swa_lr)

        for i in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            lr = opt.param_groups[0]["lr"]
            self.assertEqual(lr, initial_lr)

    def test_swa_auto_mode_detection(self):
        # Tests that SWA mode (auto or manual) is chosen correctly based on
        # parameters provided

        # Auto mode
        x, y, loss_fun, base_opt = self._define_vars_loss_opt()
        swa_start = 5
        swa_freq = 2
        swa_lr = 0.001

        opt = contriboptim.SWA(
            base_opt, swa_start=swa_start, swa_freq=swa_freq, swa_lr=swa_lr)
        self.assertEqual(opt._auto_mode, True)

        opt = contriboptim.SWA(base_opt, swa_start=swa_start, swa_freq=swa_freq)
        self.assertEqual(opt._auto_mode, True)

        with self.assertWarnsRegex("Some of swa_start, swa_freq is None"):
            opt = contriboptim.SWA(base_opt, swa_start=swa_start, swa_lr=swa_lr)
            self.assertEqual(opt._auto_mode, False)

        with self.assertWarnsRegex("Some of swa_start, swa_freq is None"):
            opt = contriboptim.SWA(base_opt, swa_freq=swa_freq, swa_lr=swa_lr)
            self.assertEqual(opt._auto_mode, False)

        with self.assertWarnsRegex("Some of swa_start, swa_freq is None"):
            opt = contriboptim.SWA(base_opt, swa_start=swa_start)
            self.assertEqual(opt._auto_mode, False)

        with self.assertWarnsRegex("Some of swa_start, swa_freq is None"):
            opt = contriboptim.SWA(base_opt, swa_freq=swa_freq)
            self.assertEqual(opt._auto_mode, False)

        with self.assertWarnsRegex("Some of swa_start, swa_freq is None"):
            opt = contriboptim.SWA(base_opt, swa_lr=swa_lr)
            self.assertEqual(opt._auto_mode, False)

    def test_swa_raises(self):
        # Tests that SWA raises errors for wrong parameter values

        x, y, loss_fun, opt = self._define_vars_loss_opt()

        with self.assertRaisesRegex(
                ValueError, "Invalid SWA learning rate: -0.0001"):
            opt = contriboptim.SWA(opt, swa_start=1, swa_freq=2, swa_lr=-1e-4)

        with self.assertRaisesRegex(
                ValueError, "Invalid swa_freq: 0"):
            opt = contriboptim.SWA(opt, swa_start=1, swa_freq=0, swa_lr=1e-4)

        with self.assertRaisesRegex(
                ValueError, "Invalid swa_start: -1"):
            opt = contriboptim.SWA(opt, swa_start=-1, swa_freq=0, swa_lr=1e-4)

    # bn_update test

    def _test_bn_update(self, data_tensor, dnn, device, label_tensor=None):

        class DatasetFromTensors(data.Dataset):
            def __init__(self, X, y=None):
                self.X = X
                self.y = y
                self.N = self.X.shape[0]

            def __getitem__(self, index):
                x = self.X[index]
                if self.y is None:
                    return x
                else:
                    y = self.y[index]
                    return x, y

            def __len__(self):
                return self.N

        with_y = label_tensor is not None
        ds = DatasetFromTensors(data_tensor, y=label_tensor)
        dl = data.DataLoader(ds, batch_size=5, shuffle=True)

        preactivation_sum = torch.zeros(dnn.n_features, device=device)
        preactivation_squared_sum = torch.zeros(dnn.n_features, device=device)
        total_num = 0
        for x in dl:
            if with_y:
                x, _ = x
            x = x.to(device)

            dnn(x)
            preactivations = dnn.compute_preactivation(x)
            if len(preactivations.shape) == 4:
                preactivations = preactivations.transpose(1, 3)
            preactivations = preactivations.reshape(-1, dnn.n_features)
            total_num += preactivations.shape[0]

            preactivation_sum += torch.sum(preactivations, dim=0)
            preactivation_squared_sum += torch.sum(preactivations**2, dim=0)

        preactivation_mean = preactivation_sum / total_num
        preactivation_var = preactivation_squared_sum / total_num
        preactivation_var = preactivation_var - preactivation_mean**2

        swa = contriboptim.SWA(optim.SGD(dnn.parameters(), lr=1e-3))
        swa.bn_update(dl, dnn, device=device)
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, prec=1e-1)

    def test_bn_update(self):
        def test(net_cls, x_shape, y_shape, device):
            x = torch.rand(x_shape, device=device)
            y = torch.rand(y_shape, device=device)

            dnn = net_cls().to(device)
            orig_momentum = dnn.bn.momentum
            dnn.train()
            self._test_bn_update(x, dnn, device)
            self._test_bn_update(x, dnn, device, label_tensor=y)
            self.assertTrue(dnn.training)

            # check that bn_update preserves eval mode
            dnn.eval()
            self._test_bn_update(x, dnn, device)
            self.assertFalse(dnn.training)

            # check that momentum is preserved
            self.assertEqual(dnn.bn.momentum, orig_momentum)

        # Test bn_update for fully-connected and convolutional networks with
        # BatchNorm1d and BatchNorm2d respectively
        objects = 100
        input_features = 5

        class DNN(nn.Module):
            def __init__(self):
                super(DNN, self).__init__()
                self.n_features = 100
                self.fc1 = nn.Linear(input_features, self.n_features)
                self.bn = nn.BatchNorm1d(self.n_features)

            def compute_preactivation(self, x):
                return self.fc1(x)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn(x)
                return x

        test(DNN, (objects, input_features), objects, 'cpu')
        if torch.cuda.is_available():
            test(DNN, (objects, input_features), objects, 'cuda')

        # Test bn_update for convolutional network and BatchNorm2d
        objects = 100
        channels = 3
        height, width = 5, 5

        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.n_features = 10
                self.conv1 = nn.Conv2d(channels, self.n_features, kernel_size=3, padding=1)
                self.bn = nn.BatchNorm2d(self.n_features, momentum=0.3)

            def compute_preactivation(self, x):
                return self.conv1(x)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn(x)
                return x

        test(CNN, (objects, channels, height, width), objects, 'cpu')
        if torch.cuda.is_available():
            test(CNN, (objects, channels, height, width), objects, 'cuda')


if __name__ == '__main__':
    run_tests()
