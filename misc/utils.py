import numpy as np
import torch
from torch import nn
import torch.optim as optim
import ipdb


class ClassifyCriterion(nn.Module):
    def __init__(self):
        self.episod = 1e-8
        super(ClassifyCriterion, self).__init__()

    def forward(self, outputs, masks):
        batch_size = len(outputs)
        loss = []
        # ipdb.set_trace()
        for ii in range(batch_size):
            prob = 1 - outputs[ii]
            mask = masks[ii]

            pos_prod = 1 - torch.prod(torch.where(mask == 1, prob, prob.new_ones(prob.shape)), dim=0)
            pos_prod = torch.where(pos_prod > 0, pos_prod, pos_prod.new_ones(pos_prod.shape))
            pos_loss = -torch.log(pos_prod + self.episod)

            neg_prod = 1 - torch.prod(torch.where(mask == 0, prob, prob.new_ones(prob.shape)), dim=0)
            neg_loss = -torch.log(1 - neg_prod + self.episod)

            img_loss = pos_loss + neg_loss
            loss.append(img_loss)

        loss = torch.stack(loss, dim=0)

        return loss.sum(-1).mean()


def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon,
                             weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon,
                          weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))


class ReduceLROnPlateau(object):
    "Optim wrapper that implements rate."

    def __init__(self, optimizer, mode='max', factor=0.1, patience=10, verbose=False, threshold=0.0001,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience, threshold,
                                                              threshold_mode, cooldown, min_lr, eps, verbose)
        self.optimizer = optimizer
        self.current_lr = get_lr(optimizer)

    def step(self):
        "Update parameters and rate"
        self.optimizer.step()

    def scheduler_step(self, val):
        self.scheduler.step(val)
        self.current_lr = get_lr(self.optimizer)

    def state_dict(self):
        return {'current_lr': self.current_lr,
                'scheduler_state_dict': self.scheduler.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if 'current_lr' not in state_dict:
            # it's normal optimizer
            self.optimizer.load_state_dict(state_dict)
            set_lr(self.optimizer, self.current_lr)  # use the lr fromt the option
        else:
            # it's a schduler
            self.current_lr = state_dict['current_lr']
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            # current_lr is actually useless in this case

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
