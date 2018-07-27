from collections import defaultdict
from torch.optim import Optimizer
import torch
import warnings

class SWA(Optimizer):
    def __init__(self, optimizer, swa_start=None, swa_freq=None, swa_lr=None):
        r"""
        swa_freq = None => call swa_upd manually
        """
        self.auto_mode, (self.swa_start, self.swa_freq) = \
                self._check_params(self, swa_start, swa_freq)
        self.swa_lr = swa_lr

        self.optimizer = optimizer
        print('SWA')
        print('start', self.swa_start)
        print('freq', self.swa_freq)
        print('lr', self.swa_lr)

        self.step_counter = 0
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.state['opt_state'] = self.optimizer.state
        #self.state['n_avg'] = 0
        for group in self.param_groups:
            group['n_avg'] = 0

    @staticmethod
    def _check_params(self, swa_start, swa_freq):
        # TODO: not raise error if swa_lr None
        # TODO: raise error if negative swa_start, swa_freq
        params = [swa_start, swa_freq]
        params_none = [param is None for param in params]
        if not all(params_none) and any(params_none):
            warnings.warn(
                "Some of swa_start, swa_freq is None, ignoring other")
            # TODO: we can avoid swa_lr
        for param in params:
            if param is not None and not isinstance(param, int):
                param = int(param)
                warnings.warn("Casting swa_start, swa_freq to int")
        return not any(params_none), params

    def _reset_lr_to_swa(self):
        for param_group in self.param_groups:
            param_group['lr'] = self.swa_lr

    def update_swa(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'swa_buffer' not in param_state:
                    param_state['swa_buffer'] = torch.zeros_like(p.data)
                buf = param_state['swa_buffer']
                #virtual_decay = 1 / (self.state["n_avg"] + 1)
                virtual_decay = 1 / (group["n_avg"] + 1)
                diff = (p.data - buf) * virtual_decay
                buf.add_(diff)
        group["n_avg"] += 1

    def swap_swa_sgd(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                buf = param_state['swa_buffer']
                tmp = torch.empty_like(p.data)
                tmp.copy_(p.data)
                p.data.copy_(buf)
                buf.copy_(tmp)
                #TODO: is it an ok way of doing this?

    def step(self, closure=None):
        if self.auto_mode:
            swa_started = self.step_counter >= self.swa_start
            if swa_started and self.swa_lr is not None:
                self._reset_lr_to_swa()
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        if self.auto_mode:
            steps = self.step_counter
            if swa_started and steps % self.swa_freq == 0:
                self.update_swa()
        return loss

# BatchNorm utils

def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    #TODO: fix this docstring
    r"""
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not _check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(_reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
