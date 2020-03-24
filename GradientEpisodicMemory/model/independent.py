# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .common import MLP, ResNet18


class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.nets = torch.nn.ModuleList()
        self.opts = []

        self.is_cifar = (args.data_file == 'cifar100.pt')
        if self.is_cifar:
            self.nc_per_task = n_outputs / n_tasks
        self.n_outputs = n_outputs

        # setup network
        for _ in range(n_tasks):
            self.nets.append(
                    MLP([n_inputs] + [int(nh / n_tasks)] * nl + [n_outputs]))

        # setup optimizer
        for t in range(n_tasks):
            self.opts.append(torch.optim.SGD(self.nets[t].parameters(),
                                             lr=args.lr))

        # setup loss
        self.bce = torch.nn.CrossEntropyLoss()

        self.finetune = args.finetune
        self.gpu = args.cuda
        self.old_task = 0

    def forward(self, x, t):
        output = self.nets[t](x)
        return output

    def observe(self, x, t, y):
        # detect beginning of a new task
        if self.finetune and t > 0 and t != self.old_task:
            # initialize current network like the previous one
            for ppold, ppnew in zip(self.nets[self.old_task].parameters(),
                                    self.nets[t].parameters()):
                ppnew.data.copy_(ppold.data)
            self.old_task = t

        self.train()
        self.zero_grad()
        self.bce(self(x, t), y).backward()
        self.opts[t].step()
