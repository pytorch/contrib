import torch
from torch import optim

class CocobBackprop(optim.Optimizer):
    """Implements Cocob-Backprop .
    It has been proposed in `Training Deep Networks without Learning Rates
    Through Coin Betting`__.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        alpha (float, optional): positive number to adjust betting fraction.
            Theoretical convergence gauarantee does not depend on choice of
            alpha (default: 100.0)
        eps (float, optional): positive epsilon (default: 1e-8)
            Theoretical convergence gauarantee does not depend on choice of
            alpha (default: 1e-8)
    __ https://arxiv.org/pdf/1705.07795.pdf
    """
    def __init__(self, params, alpha=100.0, eps=1e-8):
        self.alpha = alpha
        self.eps = eps
        defaults = dict(alpha=alpha, eps=eps)
        super(CocobBackprop, self).__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]
                param_shape = param.shape

                # Better bets for -ve gradient
                neg_grad = - grad

                if len(state) == 0:
                    # Happens only once at the begining of optimization start
                    # Set initial parameter weights and zero reward
                    state['initial_weight'] = param.data
                    state['reward'] = param.new_zeros(param_shape)

                    # Don't bet anything for first round
                    state['bet'] = param.new_zeros(param_shape)

                    # Initialize internal states useful for computing betting fraction
                    state['neg_grads_sum'] = param.new_zeros(param_shape)
                    state['grads_abs_sum'] = param.new_zeros(param_shape)
                    state['max_observed_scale'] = self.eps * param.new_ones(param_shape)

                # load states in variables
                initial_weight = state['initial_weight']
                reward = state['reward']
                bet = state['bet']
                neg_grads_sum = state['neg_grads_sum']
                grads_abs_sum = state['grads_abs_sum']
                max_observed_scale = state['max_observed_scale']

                # Update internal states useful for computing betting fraction
                max_observed_scale = torch.max(max_observed_scale, torch.abs(grad))
                grads_abs_sum += torch.abs(grad)
                neg_grads_sum += neg_grad

                # Based on how much the Better bets on -ve gradient prediction,
                # check how much the Better won (-ve if lost)
                win_amount = bet * neg_grad

                # Update better's reward. Negative reward is not allowed.
                reward = torch.max(reward + win_amount, torch.zeros_like(reward))

                # Better decides the bet fraction based on so-far observations
                bet_fraction = neg_grads_sum / (max_observed_scale * (torch.max(grads_abs_sum + max_observed_scale, self.alpha * max_observed_scale)))

                # Better makes the bet according to decided betting fraction.
                bet = bet_fraction * (max_observed_scale + reward)

                # Set parameter weights
                param.data = initial_weight + bet

                # save state back in memory
                state['neg_grads_sum'] = neg_grads_sum
                state['grads_abs_sum'] = grads_abs_sum
                state['max_observed_scale'] = max_observed_scale
                state['reward'] = reward
                state['bet'] = bet
                # For Cocob-Backprop bet_fraction need not be maintained in state. Only kept for visualization.
                state['bet_fraction'] = bet_fraction

        return loss
