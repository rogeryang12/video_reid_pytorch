import os
import torch
from torch.utils.data import DataLoader

from .logger import Logger


class Trainer(object):
    """
    Train a model
    """

    def __init__(self, args, dataset, model, optimizer, device=None):
        self.args = args
        self.log = Logger(args)
        self.loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = torch.nn.DataParallel(model).to(self.device).train()

    def train(self, cal_loss):
        """
        :param cal_loss:  function that accepts data and model, and returns loss and log
        :return:
        """
        step = 0
        while step < self.args.iters:
            for data in self.loader:
                self.optimizer.zero_grad()
                ret = cal_loss(data, self.model)
                if ret is None:
                    break
                loss, log = ret
                loss.backward()
                self.optimizer.step()
                step += 1
                self.log.save_log(step, log)

                if step % self.args.save_iter == 0:
                    state_dict = self.model.cpu().module.state_dict()
                    torch.save(state_dict, os.path.join(self.args.experiment_root, 'model{}.pkl'.format(step)))
                    self.model.to(self.device)

                if step % self.args.lr_decay_iter == 0:
                    for para_group in self.optimizer.param_groups:
                        para_group['lr'] = para_group['lr'] * self.args.gamma

                if step >= self.args.iters:
                    break
