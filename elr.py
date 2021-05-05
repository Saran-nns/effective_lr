import torch
from torch.optim.optimizer import Optimizer, required

class AdamELR(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamELR, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamELR, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
 
                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                    # Static learning rate
                    state['lr'] = torch.zeros_like(p.data).fill_(group['lr'])
                    # Effective learning rate
                    state['elr'] = torch.zeros_like(p.data)
                    # Delta parameter
                    state['dp'] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient values along parameter group p
                    state['m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values along parameter group p
                    state['v'] = torch.zeros_like(p.data)
                    
                m,v = state['m'], state['v']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                m.mul_(beta1).add_(1 - beta1, grad)
                v.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Bias corrections
                m_debiased = m.div(1 - beta1 ** state['step'])
                v_debiased = v.div(1 - beta2 ** state['step'])

                # Update learning rate
                state['dp'] = -state['lr'] * m_debiased/(torch.sqrt(v_debiased) + group['eps'])
                state['elr'] = -state['dp']/(grad + group['eps'])
                p.data.add_(state['dp'])

        return loss
