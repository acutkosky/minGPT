
import math
import torch
import numpy as np
from torch.optim import Optimizer

def ONS_CB_interval(state, grad_correlation):
    initial_wealth = 1.0
    domain = 0.5

    if 'cb.wealth' not in state:
        state['cb.wealth'] = torch.tensor(initial_wealth)
    if 'cb.max_grad' not in state:
        state['cb.max_grad'] = torch.tensor(1e-5)
    if 'cb.w' not in state:
        state['cb.w'] = torch.tensor(0.0)
    if 'cb.beta' not in state:
        state['cb.beta'] = torch.tensor(0.0)
    if 'cb.sum_squared_z' not in state:
        state['cb.sum_squared_z'] = torch.tensor(1e-8)

    # loss is unused here

    clipped_old_w = torch.clamp(state['cb.w'], 0.0, 1.0)
    # set gradient to zero for coordinates in which the gradient is trying to push
    # us outside of the constraint set.
    if (grad_correlation *(state['cb.w'] - clipped_old_w)) >= 0.0:
        return clipped_old_w

    grad = grad_correlation# * ((grad_correlation *(state['cb.w'] - clipped_old_w)) < 0.0)

    truncated_grad = torch.clamp(grad, -state['cb.max_grad'], state['cb.max_grad'])
    state['cb.max_grad'].copy_(torch.max(torch.abs(grad), state['cb.max_grad']))
    grad = truncated_grad

    # compute gradient of beta-loss L(beta) = -log(1-beta *grad)
    z = grad/(1.0 - state['cb.beta'] * grad)


    ### do Online Newton Step (ONS) update on L(beta)
    state['cb.sum_squared_z'].add_(z**2)
    # ONS_eta = 2.0/(2.0 - np.log(3.0)) # magic constant related to logarithms and 0.5...
    ONS_eta = -0.5 * domain**2/(domain + np.log(1-domain))
    domain = domain/state['cb.max_grad']
    state['cb.beta'].copy_(torch.clamp(state['cb.beta'] - ONS_eta * z / state['cb.sum_squared_z'], 0.0, domain))

    state['cb.wealth'].add_(-grad * state['cb.w'])

    state['cb.w'].copy_(state['cb.beta'] * state['cb.wealth'])
    # print("w: ",self.w)
    # print("self.beta: ", self.beta)


    return torch.clamp(state['cb.w'], 0.0, 1.0)



class Acceleration_NonConvex_small(Optimizer):
    def __init__(self, params, lr=0.1, weight_decay=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(Acceleration_NonConvex_small, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Acceleration_NonConvex_small, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # # Perform stepweight decay
                # p.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad

                grad = grad + group['weight_decay'] * p
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                # amsgrad = group['amsgrad']

                state = self.state[p]


                # State initialization
                if len(state) == 0:
                    state['step'] = 1.0
                    state['sum_alpha'] = 1.0

                    state['previous_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Weighted average sum of (g - g_prev)^2
                    state['alpha_weighted_sum_squared_diff_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['optimistic_value'] = torch.clone(p)
            # loss currently unused here.

            # could also try np.linalg.norm(grad)**2
                alpha = state['step']
                next_alpha = alpha + 1.0

                squared_magnitude_diff_grad = (grad - state['previous_grad'])**2

                state['alpha_weighted_sum_squared_diff_grad'].add_((alpha**2) * squared_magnitude_diff_grad)

                weighted_grad_diff = next_alpha * grad + alpha * (grad - state['previous_grad'])

                optimistic_eta = group['lr'] / torch.sqrt(1e-8 + state['alpha_weighted_sum_squared_diff_grad'])

                state['optimistic_value'].add_(- optimistic_eta * weighted_grad_diff)

                gradient_correlation = torch.sum(state['sum_alpha'] * grad * state['previous_grad'] * alpha * optimistic_eta)

                prox_lr = ONS_CB_interval(state, -gradient_correlation)
                # assert torch.abs(gradient_correlation) < 1e5, "grad correaltion big: {}, alpha: {}, grad norm: {}, opteta: {}".format(gradient_correlation, alpha, torch.norm(state['previous_grad']),torch.norm(optimistic_eta))
                state['step'] += 1.0
                alpha = next_alpha
                state['sum_alpha'] += alpha
                tau = alpha/state['sum_alpha']

                p_update = tau * (state['optimistic_value'] - p) - (1 - tau) * prox_lr * grad * alpha * optimistic_eta
                # p_update = state['optimistic_value'] - p
                p.add_(p_update)

                state['previous_grad'].copy_(grad)
        return loss




class custom_adagrad(Optimizer):
    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr)
        super(custom_adagrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(custom_adagrad, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # # Perform stepweight decay
                # p.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                # amsgrad = group['amsgrad']

                state = self.state[p]


                # State initialization
                if len(state) == 0:
                    state['sum_squared_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['adagrad_val'] = torch.clone(p)
            # loss currently unused here.

            # could also try np.linalg.norm(grad)**2
                state['sum_squared_grad'].add_(grad**2)
                p_update = -grad * group['lr'] / torch.sqrt(state['sum_squared_grad'] + 0.0001)
                state['adagrad_val'].add_(p_update)
                p.add_(state['adagrad_val'] - p)

        return loss
