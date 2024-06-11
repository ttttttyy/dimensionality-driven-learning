import os
import numpy as np
import matplotlib.pyplot as plt
from lid import lid_mle
import torch
import util

np.random.seed(1024)

MODELS = ['ce', 'fl', 'bl', 'bsl', 'bhl', 'd2l']
MODEL_LABELS = ['cross-entropy', 'forward', 'backward', 'boot-soft', 'boot-hard', 'D2L']
COLORS = ['r', 'y', 'c', 'm', 'g', 'b']
MARKERS = ['x', 'D', '<', '>', '^', 'o']


def lid_trend_through_training(exp_name, dataset, data_loader, device, model, optimizer, scheduler, model_name='d2l', noise_type='sym', noise_ratio=0.):
    """
    plot the lid trend for clean vs noisy samples through training.
    This can provide some information about manifold learning dynamics through training.
    """

    lids, train_accs, test_accs = None, None, None

    # get LID of raw inputs
    k = 20
    lids = []
    for j, (images,labels) in enumerate(data_loader['train_dataset']):
        images = images.to(device, non_blocking = True)
        lids.extend(lid_mle(images, images, k=k))
        
    lids = torch.stack(lids, dim=0).type(torch.float32)
    lid_X = lids.mean()
    print('LID of input X: ', lid_X)
    
    exp_path = os.path.join(exp_name, model_name)
    checkpoint_path = os.path.join(exp_path, 'checkpoints')
    checkpoint_path_file = os.path.join(checkpoint_path, model_name)
    checkpoint = util.load_model(filename=checkpoint_path_file,
                                 model=model,
                                 optimizer=optimizer,
                                 scheduler=scheduler)
    ENV = checkpoint['ENV']
    train_accs = ENV['train_history']
    train_accs.insert(0,0)
    test_accs = ENV['eval_history']
    test_accs.insert(0,0)
    lids = ENV['lid']
    lids.insert(0,lid_X)
    lids = torch.stack(lids, dim=0).type(torch.float32)

    plot(dataset, model_name, noise_ratio, lids, train_accs, test_accs)


def plot(dataset, model_name, noise_ratio, lids, train_accs, test_accs):
    """
    plot function
    """
    # plot
    fig = plt.figure()  # figsize=(7, 6)
    xnew = np.arange(0, len(lids), 5)

    lids = lids.cpu().numpy()
    train_accs = np.array(train_accs) / 100
    test_accs = np.array(test_accs) / 100
    print(train_accs)
    lids = lids[xnew]
    train_accs = train_accs[xnew]
    test_accs = test_accs[xnew]

    ax = fig.add_subplot(111)
    ax.plot(xnew, lids, c='r', marker='o', markersize=3, linewidth=2, label='LID score')

    ax2 = ax.twinx()
    ax2.plot(xnew, train_accs, c='b', marker='x', markersize=3, linewidth=2, label='Train acc')
    ax2.plot(xnew, test_accs, c='c', marker='^', markersize=3, linewidth=2, label='Test acc')

    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Subspace dimensionality (LID score)", fontsize=15)
    ax2.set_ylabel("Train/test accuracy", fontsize=15)
    # ax.set_title("%s with %s%% noisy labels" % (dataset.upper(), noise_ratio), fontsize=15)

    if dataset == 'mnist':
        ax.set_ylim((4, 22))  # for mnist
        ax2.set_ylim((0.2, 1.2))
    elif dataset == 'svhn':
        ax.set_ylim((7, 20)) # for svhn
        ax2.set_ylim((0.2, 1.2))
    elif dataset == 'cifar10':
        ax.set_ylim((2.5, 12.5))  # for cifar-10
        #ax.set_ylim((3.5, 20.5))
        ax2.set_ylim((0., 1.2))
    elif dataset == 'cifar100':
        ax.set_ylim((3, 12))  # for cifar-100
        ax2.set_ylim((0., 1.))

    legend = ax.legend(loc='upper left')
    plt.setp(legend.get_texts(), fontsize=15)
    legend2 = ax2.legend(loc='upper right')
    plt.setp(legend2.get_texts(), fontsize=15)
    fig.savefig("plots/lid_trend_%s_%s_%s.png" % (model_name, dataset, noise_ratio), dpi=300)
    plt.show()


def lid_trend_of_learning_models(exp_name, dataset, model, optimizer, scheduler, model_list=['ce'], noise_ratio=0):
    """
    The LID trend of different learning models throughout.
    """
    # plot initialization
    fig = plt.figure()  # figsize=(7, 6)
    ax = fig.add_subplot(111)

    for model_name in model_list:
        exp_path = os.path.join(exp_name, model_name)
        checkpoint_path = os.path.join(exp_path, 'checkpoints')
        checkpoint_path_file = os.path.join(checkpoint_path, model_name)
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     scheduler=scheduler)
        ENV = checkpoint['ENV']
        lids = ENV['lid']
        lids = torch.stack(lids, dim=0).type(torch.float32)
        lids = lids.cpu().numpy()
            # smooth for plot
        lids[lids < 0] = 0
        lids[lids > 10] = 10

        xnew = np.arange(0, len(lids), 5)
        lids = lids[xnew]

        # plot line
        idx = MODELS.index(model_name)
        ax.plot(xnew, lids, c=COLORS[idx], marker=MARKERS[idx], markersize=3, linewidth=2, label=MODEL_LABELS[idx])

    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Subspace dimensionality (LID score)", fontsize=15)
    # ax.set_title("%s with %s%% noisy labels" % (dataset.upper(), noise_ratio), fontsize=15)
    legend = plt.legend(loc='lower center', ncol=2)
    plt.setp(legend.get_texts(), fontsize=15)
    fig.savefig("plots/lid_trend_all_models_%s_%s_%s.png" % (exp_name, dataset, noise_ratio), dpi=300)
    plt.show()

def test_acc_trend_of_learning_models(exp_name, dataset, model, optimizer, scheduler, model_list=['ce'], noise_ratio=0):
    """
    The test_acc trend of different learning models throughout.
    """
    # plot initialization
    fig = plt.figure()  # figsize=(7, 6)
    ax = fig.add_subplot(111)

    for model_name in model_list:
        exp_path = os.path.join(exp_name, model_name)
        checkpoint_path = os.path.join(exp_path, 'checkpoints')
        checkpoint_path_file = os.path.join(checkpoint_path, model_name)
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     scheduler=scheduler)
        ENV = checkpoint['ENV']
        test_accs = ENV['eval_history']
        test_accs = np.array(test_accs) / 100

        xnew = np.arange(0, len(test_accs), 5)
        test_accs = test_accs[xnew]

        # plot line
        idx = MODELS.index(model_name)
        ax.plot(xnew, test_accs, c=COLORS[idx], marker=MARKERS[idx], markersize=3, linewidth=2, label=MODEL_LABELS[idx])

    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Test Accuracy", fontsize=15)
    # ax.set_title("%s with %s%% noisy labels" % (dataset.upper(), noise_ratio), fontsize=15)
    legend = plt.legend(loc='lower center', ncol=2)
    plt.setp(legend.get_texts(), fontsize=15)
    fig.savefig("plots/test_accs_trend_all_models_%s_%s_%s.png" % (exp_name, dataset, noise_ratio), dpi=300)
    plt.show()
    
def csr_trend_of_learning_models(exp_name, dataset, model, optimizer, scheduler, model_list=['ce'], noise_ratio=0):
    """
    The CSR trend of different learning models throughout.
    """
    # plot initialization
    fig = plt.figure()  # figsize=(7, 6)
    ax = fig.add_subplot(111)

    for model_name in model_list:
        exp_path = os.path.join(exp_name, model_name)
        checkpoint_path = os.path.join(exp_path, 'checkpoints')
        checkpoint_path_file = os.path.join(checkpoint_path, model_name)
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     scheduler=scheduler)
        ENV = checkpoint['ENV']
        csr = ENV['csr']
        csr = torch.stack(csr, dim=0).type(torch.float32)
        csr = csr.cpu().numpy()

        xnew = np.arange(0, len(csr), 5)
        csr = csr[xnew]

        # plot line
        idx = MODELS.index(model_name)
        ax.plot(xnew, csr, c=COLORS[idx], marker=MARKERS[idx], markersize=3, linewidth=2, label=MODEL_LABELS[idx])

    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("CRS", fontsize=15)
    # ax.set_title("%s with %s%% noisy labels" % (dataset.upper(), noise_ratio), fontsize=15)
    legend = plt.legend(loc='lower center', ncol=2)
    plt.setp(legend.get_texts(), fontsize=15)
    fig.savefig("plots/crs_trend_all_models_%s_%s_%s.png" % (exp_name, dataset, noise_ratio), dpi=300)
    plt.show()