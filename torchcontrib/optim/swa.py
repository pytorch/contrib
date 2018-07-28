from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import torch
import warnings

#TODO: " -> ' or ' -> "
class SWA(Optimizer):
    def __init__(self, optimizer, swa_start=None, swa_freq=None, swa_lr=None):
        r"""
        swa_freq = None => call swa_upd manually
        """
        self._auto_mode, (self.swa_start, self.swa_freq) = \
                self._check_params(self, swa_start, swa_freq)
        self.swa_lr = swa_lr

        if self._auto_mode:
            if swa_start < 0:
                raise ValueError("Invalid swa_start: {}".format(swa_start))
            if swa_freq < 1:
                raise ValueError("Invalid swa_freq: {}".format(swa_freq))
        else:
            if self.swa_lr is not None:
                warnings.warn(
                    "Some of swa_start, swa_freq is None, ignoring swa_lr")
            # If not in auto mode make all swa parameters None
            self.swa_lr = None
            self.swa_start = None
            self.swa_freq = None

        if self.swa_lr is not None and self.swa_lr < 0:
            raise ValueError("Invalid SWA learning rate: {}".format(swa_lr))

        self.optimizer = optimizer
        print('SWA')
        print('start', self.swa_start)
        print('freq', self.swa_freq)
        print('lr', self.swa_lr)

        # self.step_counter = 0
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        #self.state['opt_state'] = self.optimizer.state
        self.opt_state = self.optimizer.state
        #self.state['n_avg'] = 0
        for group in self.param_groups:
            group['n_avg'] = 0
            group['step_counter'] = 0

    @staticmethod
    def _check_params(self, swa_start, swa_freq):
            
        params = [swa_start, swa_freq]
        params_none = [param is None for param in params]
        if not all(params_none) and any(params_none):
            warnings.warn(
                "Some of swa_start, swa_freq is None, ignoring other")
            # TODO: we can avoid swa_lr
        for i, param in enumerate(params):
            if param is not None and not isinstance(param, int):
                params[i] = int(param)
                warnings.warn("Casting swa_start, swa_freq to int")
        return not any(params_none), params

    def _reset_lr_to_swa(self):
        if self.swa_lr is None:
            return
        for param_group in self.param_groups:
            if param_group['step_counter'] >= self.swa_start:
                param_group['lr'] = self.swa_lr

    def update_swa_group(self, group):
        for p in group['params']:
            param_state = self.state[p]
            if 'swa_buffer' not in param_state:
                param_state['swa_buffer'] = torch.zeros_like(p.data)
            buf = param_state['swa_buffer']
            virtual_decay = 1 / (group["n_avg"] + 1)
            diff = (p.data - buf) * virtual_decay
            buf.add_(diff)
        group["n_avg"] += 1

    def update_swa(self):
        for group in self.param_groups:
            self.update_swa_group(group)

    def swap_swa_sgd(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'swa_buffer' not in param_state:
                    # If swa wasn't applied we don't swap params
                    continue
                buf = param_state['swa_buffer']
                tmp = torch.empty_like(p.data)
                tmp.copy_(p.data)
                p.data.copy_(buf)
                buf.copy_(tmp)
                #TODO: is it an ok way of doing this?

    def step(self, closure=None):
        self._reset_lr_to_swa()
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            group["step_counter"] += 1
            steps = group["step_counter"]
            if self._auto_mode:
                if steps > self.swa_start and steps % self.swa_freq == 0:
                    self.update_swa_group(group)
        return loss

    def state_dict(self):
        opt_state_dict = self.optimizer.state_dict()
        swa_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
                                for k, v in self.state.items()}
        opt_state = opt_state_dict["state"]
        param_groups = opt_state_dict["param_groups"]
        return {"opt_state": opt_state, "swa_state": swa_state, 
                "param_groups": param_groups}

    def load_state_dict(self, state_dict):
        # Need to load the optimizer state explicitly
        swa_state_dict = {"state": state_dict["swa_state"], 
                          "param_groups": state_dict["param_groups"]}
        opt_state_dict = {"state": state_dict["opt_state"], 
                          "param_groups": state_dict["param_groups"]}
        super(SWA, self).load_state_dict(swa_state_dict)
        self.optimizer.load_state_dict(opt_state_dict)
        #self.param_groups = self.optimizer.param_groups
        self.opt_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group['n_avg'] = 0
        param_group['step_counter'] = 0
        self.optimizer.add_param_group(param_group)


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
