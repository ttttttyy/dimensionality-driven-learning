import logging
import os
import torch
import numpy as np
import torch.nn.functional as F
from lid import lid_mle
from lass import lass

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0/batch_size))
    return res


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              '\tglobal_step=' + str(global_step)
    for key, value in kwargs.items():
        display += '\t' + str(key) + '=%.5f' % value
    display += '\ttime=%.2fit/s' % (1. / time_elapse)
    return display


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def save_model(filename, model, optimizer, scheduler, epoch, **kwargs):
    # Torch Save State Dict
    state = {
        'epoch': epoch+1,
        'model': model.state_dict() if model is not None else None,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
    }
    for key, value in kwargs.items():
        state[key] = value
    torch.save(state, filename+'.pth')
    return


def load_model(filename, model, optimizer, scheduler, **kwargs):
    checkpoints = torch.load(filename + '.pth')
    if model is not None and checkpoints['model'] is not None:
        model.load_state_dict(checkpoints['model'])
    if optimizer is not None and checkpoints['optimizer'] is not None:
        optimizer.load_state_dict(checkpoints['optimizer'])
    if scheduler is not None and checkpoints['scheduler'] is not None:
        scheduler.load_state_dict(checkpoints['scheduler'])
    print("%s Loaded!" % (filename))
    return checkpoints


def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary_head" not in name)/1e6


def get_lids_random_batch(model, data_loader, device, k=20, batch_size=128, batch_num=10):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    """

    lids = []
    model.eval()
    
    def estimate(images):
        images = images.to(device, non_blocking = True)
        #get the output of the second-to-last layer of the network
        with torch.no_grad():
            _, X_act = model(images)
            
        lid_batch = lid_mle(X_act, X_act, k=k)
        return lid_batch

    
    for j, (images,labels) in enumerate(data_loader['train_dataset']):
        if j < batch_num:
            lid_batch = estimate(images)
            lids.extend(lid_batch)

    lids = torch.stack(lids, dim=0).type(torch.float32)
    return lids

def get_csr_random_batch(model, data_loader, device, batch_size=128, batch_num=4):
    model.eval()
    adv_ind_sum = 0
    for j, (images,labels) in enumerate(data_loader['test_dataset']):
        if j < batch_num:
            images = images.to(device, non_blocking = True)
            scale_factor = 255. / (torch.max(images) - torch.min(images))
            #scale_factor = 1
            csr_model = lass(model, device, a=0.25 / scale_factor, b=0.2 / scale_factor, r=0.3 / scale_factor, iter_max=100)
            X_adv, adv_ind = csr_model.find(images)
            adv_ind_sum += torch.sum(adv_ind)
            
    samples_num = batch_num * batch_size
    csr = adv_ind_sum * 1. / samples_num
    return csr
            