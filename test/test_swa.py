import functools
from copy import deepcopy
import torch
from torch.autograd import Variable
from torch import sparse
from torch import optim
import torchcontrib.optim as contriboptim
from common import TestCase, run_tests# TEST_WITH_UBSAN
from torch.utils import data


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def drosenbrock(tensor):
    x, y = tensor
    return torch.DoubleTensor((-400 * x * (y - x ** 2) - 2 * (1 - x), 200 * (y - x ** 2)))


def wrap_old_fn(old_fn, **config):
    def wrapper(closure, params, state):
        return old_fn(closure, params, config, state)
    return wrapper


class TestSWA(TestCase):
    # Pavel: I slightly update the _test_... functions to (1) remove the
    # legacy-related parts and (2) add oprimizer.swap_swa_sgd() in the end of
    # optimization

    def _test_rosenbrock(self, constructor, automode=True):
        # automode shows wether we need to update SWA params manually

        params = Variable(torch.Tensor([1.5, 1.5]), requires_grad=True)
        optimizer = constructor([params])

        solution = torch.Tensor([1, 1])
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

    def _test_rosenbrock_sparse(self, constructor, sparse_only=False):
        params_t = torch.Tensor([1.5, 1.5])

        params = Variable(params_t, requires_grad=True)
        optimizer = constructor([params])

        if not sparse_only:
            params_c = Variable(params_t.clone(), requires_grad=True)
            optimizer_c = constructor([params_c])

        solution = torch.Tensor([1, 1])
        initial_dist = params.data.dist(solution)

        def eval(params, sparse_grad, w):
            # Depending on w, provide only the x or y gradient
            optimizer.zero_grad()
            loss = rosenbrock(params)
            loss.backward()
            grad = drosenbrock(params.data)
            # NB: We torture test the optimizer by returning an
            # uncoalesced sparse tensor
            if w:
                i = torch.LongTensor([[0, 0]])
                x = grad[0]
                v = torch.DoubleTensor([x / 4., x - x / 4.])
            else:
                i = torch.LongTensor([[1, 1]])
                y = grad[1]
                v = torch.DoubleTensor([y - y / 4., y / 4.])
            x = sparse.DoubleTensor(i, v, torch.Size([2]))
            if sparse_grad:
                params.grad.data = x
            else:
                params.grad.data = x.to_dense()
            return loss

        for i in range(2000):
            # Do cyclic coordinate descent
            w = i % 2
            optimizer.step(functools.partial(eval, params, True, w))
            if not sparse_only:
                optimizer_c.step(functools.partial(eval, params_c, False, w))
                self.assertEqual(params.data, params_c.data)

        self.assertLessEqual(params.data.dist(solution), initial_dist)

    def _test_basic_cases_template(self, weight, bias, input, constructor):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)
        optimizer = constructor(weight, bias)

        # to check if the optimizer can be printed as a string
        optimizer.__repr__()

        def fn():
            optimizer.zero_grad()
            y = weight.mv(input)
            if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
                y = y.cuda(bias.get_device())
            loss = (y + bias).pow(2).sum()
            loss.backward()
            return loss

        initial_value = fn().item()
        for i in range(200):
            optimizer.step(fn)
        self.assertLess(fn().item(), initial_value)

    def _test_state_dict(self, weight, bias, input, constructor):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)

        def fn_base(optimizer, weight, bias):
            optimizer.zero_grad()
            i = input_cuda if weight.is_cuda else input
            loss = (weight.mv(i) + bias).pow(2).sum()
            loss.backward()
            return loss

        optimizer = constructor(weight, bias)
        fn = functools.partial(fn_base, optimizer, weight, bias)

        # Prime the optimizer
        for i in range(20):
            optimizer.step(fn)
        # Clone the weights and construct new optimizer for them
        weight_c = Variable(weight.data.clone(), requires_grad=True)
        bias_c = Variable(bias.data.clone(), requires_grad=True)
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

    def _test_basic_cases(self, constructor, ignore_multidevice=False):
        self._test_state_dict(
            torch.randn(10, 5),
            torch.randn(10),
            torch.randn(5),
            constructor
        )
        self._test_basic_cases_template(
            torch.randn(10, 5),
            torch.randn(10),
            torch.randn(5),
            constructor
        )
        # non-contiguous parameters
        self._test_basic_cases_template(
            torch.randn(10, 5, 2)[..., 0],
            torch.randn(10, 2)[..., 0],
            torch.randn(5),
            constructor
        )
        # CUDA
        if not torch.cuda.is_available():
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(),
            torch.randn(10).cuda(),
            torch.randn(5).cuda(),
            constructor
        )
        # Multi-GPU
        if not torch.cuda.device_count() > 1 or ignore_multidevice:
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(0),
            torch.randn(10).cuda(1),
            torch.randn(5).cuda(0),
            constructor
        )

    def _build_params_dict(self, weight, bias, **kwargs):
        return [dict(params=[weight]), dict(params=[bias], **kwargs)]

    def _build_params_dict_single(self, weight, bias, **kwargs):
        return [dict(params=bias, **kwargs)]

    #Test SWA
    
    def test_swa(self):
        print("here")
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

        #auto_constructor_list = [sgd_momentum_constructor]

        for i, constructor in enumerate(auto_constructor_list):
            print(constructor)
            # Pass
            self._test_rosenbrock(constructor)
            self._test_basic_cases(
                    lambda weight, bias: constructor([weight, bias]))
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

    def testSWAAuto(self):
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
            if i % swa_freq == 0 and i > swa_start:
                n_avg += 1
                x_sum += x.data
                y_sum += y.data

        opt.swap_swa_sgd()
        x_avg = x_sum / n_avg
        y_avg = y_sum / n_avg
        self.assertEqual(x_avg, x)
        self.assertEqual(y_avg, y)      

    def testSWAManual(self):
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
            if i % swa_freq == 0 and i > swa_start:
                opt.update_swa()
                n_avg += 1
                x_sum += x.data
                y_sum += y.data

        opt.swap_swa_sgd()
        x_avg = x_sum / n_avg
        y_avg = y_sum / n_avg
        self.assertEqual(x_avg, x)
        self.assertEqual(y_avg, y)      

    def testSWAManualGroup(self):
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
            if i % swa_freq == 0 and i > swa_start:
                opt.update_swa_group(opt.param_groups[1])
                n_avg += 1
                y_sum += y.data

        x_before_swap = x.data.clone()
        opt.swap_swa_sgd()
        y_avg = y_sum / n_avg
        self.assertEqual(y_avg, y)      
        self.assertEqual(x_before_swap, x)      

    def testSWAAutoGroupAddedDuringRun(self):
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

        opt.add_param_group({'params': y, 'lr':1e-4})

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
        
    def testSWALR(self):
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
        opt = contriboptim.SWA(opt, swa_lr=swa_lr)

        for i in range(1, 11):
            opt.zero_grad()
            loss = loss_fun(x, y)
            loss.backward()
            opt.step()
            lr = opt.param_groups[0]["lr"]
            self.assertEqual(lr, initial_lr)    

    def testAutoMode(self):
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

        opt = contriboptim.SWA(base_opt, swa_start=swa_start, swa_lr=swa_lr)
        self.assertEqual(opt._auto_mode, False)

        opt = contriboptim.SWA(base_opt, swa_freq=swa_freq, swa_lr=swa_lr)
        self.assertEqual(opt._auto_mode, False)

        opt = contriboptim.SWA(base_opt, swa_start=swa_start)
        self.assertEqual(opt._auto_mode, False)

        opt = contriboptim.SWA(base_opt, swa_freq=swa_freq)
        self.assertEqual(opt._auto_mode, False)

        opt = contriboptim.SWA(base_opt, swa_lr=swa_lr)
        self.assertEqual(opt._auto_mode, False)

    def testSWARaises(self):
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

    def testBNUpdate(self):
        # Tests bn_update function by creating a small dataset and network and
        # testing that the manually computed activation statistics are the same
        # as those computed by bn_update
        


if __name__ == '__main__':
    run_tests()
