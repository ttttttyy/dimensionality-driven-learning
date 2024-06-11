import torch
from scipy.spatial.distance import cdist

def gmean(input_x, dim=0):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))

def get_lid_r(data, reference):
    b = data.shape[0]
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    r = torch.cdist(data, reference, p=2)
    a, idx = torch.sort(r, dim=1)
    return r, a, idx

def lid_mle(data, reference, k=20, get_idx=False, compute_mode='use_mm_for_euclid_dist_if_necessary'):
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    r = torch.cdist(reference, data, p=2, compute_mode=compute_mode)
    a, idx = torch.sort(r, dim=1)
    lids = -k / torch.sum(torch.log(a[:, 1:k+1] / a[:, k+1].view(-1,1)), dim=1)
    if get_idx:
        return idx, lids
    return lids

def lid_mom_est(data, reference, k, get_idx=False, compute_mode='use_mm_for_euclid_dist_if_necessary'):
    b = data.shape[0]
    k = min(k, b-2)
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)
    a, idx = torch.sort(r, dim=1)
    m = torch.mean(a[:, 1:k], dim=1)
    lids = m / (a[:, k] - m)
    if get_idx:
        return idx, lids
    return lids

def lid_mom_est_eps(data, reference, k, get_idx=False):
    b = data.shape[0]
    k = min(k, b-2)
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    r = torch.cdist(data, reference, p=2)
    a, idx = torch.sort(r, dim=1)
    m = torch.mean(a[:, 1:k], dim=1)
    lids = m / ((a[:, k] - m) + 1.e-4)
    if get_idx:
        return idx, lids
    return lids