from torch.optim import Optimizer

class SWA(Optimizer):
    def __init__(self, optimizer, swa_start, swa_lr, swa_freq=None):
        r"""
        swa_freq = None => call swa_upd manually
        """
        self.optimizer = optimizer
        self.swa_start = swa_start
	elf.swa_freq = swa_freq
        self.step_counter = 0
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self._make_shadow_vars()

        self.n_avg = 0
        self.state['n_avg'] = n_avg

    def _make_shadow_vars():
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'swa_buffer' not in param_state:
                    param_state['swa_buffer'] = torch.zeros_like(p.data)

    def swa_upd(self):
        for group in self.param_groups:
            for p in group['params']:
                buf = param_state['swa_buffer']
                virtual_decay = 1 / (self.n_avg + 1)
                diff = (p.data - buf) * virtual_decay
                buf.add_(diff)
                self.n_avg += 1

    def swap_swa_sgd(self):
        for group in self.param_groups:
            for p in group['params']:
                buf = param_state['swa_buffer']
                tmp = p.data
                p.data = buf
                buf = tmp
                #TODO: is it an ok way of doing this?

    def step(self):
	loss = self.optimizer.step()
        self.step_counter += 1
        if self.swa_freq is not None:
            if self.step_counter % self.swa_freq == 0:
                self.swa_upd()
        return loss

