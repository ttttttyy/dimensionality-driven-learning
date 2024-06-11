import numpy as np
import torch
from util import get_lids_random_batch
import os


class D2LCallback:
    def __init__(self, model, data_loader, device, epochs=120,
                 pace_type='d2l', init_epoch=5, epoch_win=5, lid_subset_size=1280,
                 lid_k=20, verbose=1, is_found_turning_point=False):
        super(D2LCallback, self).__init__()
        self.validation_data = None
        self.model = model
        self.turning_epoch = -1
        self.data_loader = data_loader
        self.device = device
        self.epochs = epochs
        self.pace_type = pace_type
        self.mean_lid = -1.
        self.lids = []
        self.p_lambda = 0.
        self.init_epoch = init_epoch
        self.epoch_win = epoch_win
        self.lid_subset_size = lid_subset_size
        self.lid_k = lid_k
        self.verbose = verbose
        self.alpha = 1.0
        self.is_found_turning_point = is_found_turning_point

    def on_epoch_begin(self, epoch, logs={}):
        lids_tem = get_lids_random_batch(self.model, self.data_loader, self.device, k=20, batch_size=128)
        lid = lids_tem.mean().item()
        self.p_lambda = epoch*1./self.epochs

        # deal with possible illegal lid value
        if lid > 0:
            self.lids.append(lid)
        else:
            self.lids.append(self.lids[-1])

        # find the turning point where to apply lid-paced learning strategy
        if self.found_turning_point(self.lids):
            self.update_learning_pace()
            self.is_found_turning_point = True

        if len(self.lids) > 5:
            print('lid = ..., ', self.lids[-5:])
        else:
            print('lid = ..., ', self.lids)

        if self.verbose > 0:
            print('--Epoch: %s, LID: %.2f, min LID: %.2f, lid window: %s, turning epoch: %s, lambda: %.2f' %
                  (epoch, lid, min(self.lids), self.epoch_win, self.turning_epoch, self.p_lambda))

        return

    def found_turning_point(self, lids):
        if len(lids) > self.init_epoch + self.epoch_win:
            if self.turning_epoch > -1:
                return True
            else:
                smooth_lids = lids[-self.epoch_win-1:-1]
                if lids[-1] - torch.mean(torch.tensor(smooth_lids)) > 2 * torch.std(torch.tensor(smooth_lids)):
                    self.turning_epoch = len(lids) - 2
                    # rollback model if you want, should be called before checkpoint callback
                    # otherwise need to save two models
                    return True
        return False

    def update_learning_pace(self):
        expansion = self.lids[-1] / min(self.lids)
        self.alpha = torch.exp(torch.tensor(-self.p_lambda * expansion)).item()
        # self.alpha = np.exp(-0.1*expansion)

        print('## Turning epoch: %s, lambda: %.2f, expansion: %.2f, alpha: %.2f' %
              (self.turning_epoch, self.p_lambda, expansion, self.alpha))

        # self.alpha = np.exp(-expansion)

