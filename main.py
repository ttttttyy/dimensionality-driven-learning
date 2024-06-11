import torch
import argparse
import util
import os
import datetime
import random
import mlconfig
import loss
import models
import dataset
import shutil
from evaluator import Evaluator
from trainer import Trainer
from util import get_lids_random_batch,get_csr_random_batch
from callback_util import D2LCallback
import numpy as np
from plot import lid_trend_through_training, lid_trend_of_learning_models, test_acc_trend_of_learning_models, csr_trend_of_learning_models

# ArgParse
parser = argparse.ArgumentParser(description='Normalized Loss Functions for Deep Learning with Noisy Labels')
# Training
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--config_path', type=str, default='configs')
parser.add_argument('--version', type=str, default='ce')
parser.add_argument('--exp_name', type=str, default="run1")
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--asym', action='store_true', default=False)
parser.add_argument('--noise_rate', type=float, default=0.0)
parser.add_argument('--plot', action='store_true', default=False)
parser.add_argument('--plotall', action='store_true', default=False)
args = parser.parse_args()

# Set up
if args.exp_name == '' or args.exp_name is None:
    args.exp_name = 'exp_' + datetime.datetime.now()
exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)

logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")
for arg in vars(args):
    logger.info("%s: %s" % (arg, getattr(args, arg)))

random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    logger.info("Using CUDA!")
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

logger.info("PyTorch Version: %s" % (torch.__version__))
config_file = os.path.join(args.config_path, args.version) + '.yaml'
config = mlconfig.load(config_file)
if args.version == 'fl' or args.version == 'bl':
    config['criterion']['noise_rate']=args.noise_rate
if args.version != 'd2l':
    config.set_immutable()
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))
for key in config:
    logger.info("%s: %s" % (key, config[key]))


def train(starting_epoch, model, data_loader, optimizer, scheduler, criterion, trainer, evaluator, ENV, callback, mode):
    for epoch in range(starting_epoch, config.epochs):
        if args.version == 'd2l':
            if mode == 'stage2':
                config['criterion']['alpha'] = callback.alpha
                criterion=config.criterion()
                
        logger.info("="*20 + "Training" + "="*20)
            
        # Train
        ENV['global_step'] = trainer.train(epoch, ENV['global_step'], model, optimizer, criterion)
        scheduler.step()
        
        if args.version == 'd2l':
            callback.on_epoch_begin(epoch)
            if mode == 'stage1':
                if callback.is_found_turning_point == True:
                    break
                
        # Eval
        logger.info("="*20 + "Eval" + "="*20)
        evaluator.eval(epoch, ENV['global_step'], model, torch.nn.CrossEntropyLoss())
        payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)
        # LID
        lids = get_lids_random_batch(model, data_loader, device, k=20, batch_size=128)
        lid = lids.mean()
        logger.info('LID:%f'%(lid))
        # CSR
        csr = get_csr_random_batch(model, data_loader, device)
        logger.info('CSR:%f'%(csr))
        
        ENV['train_history'].append(trainer.acc_meters.avg*100)
        ENV['eval_history'].append(evaluator.acc_meters.avg*100)
        ENV['curren_acc'] = evaluator.acc_meters.avg*100
        ENV['best_acc'] = max(ENV['curren_acc'], ENV['best_acc'])
        ENV['lid'].append(lid)
        ENV['csr'].append(csr)
    

        # Reset Stats
        trainer._reset_stats()
        evaluator._reset_stats()

        # Save Model
        target_model = model.module if args.data_parallel else model
        util.save_model(ENV=ENV,
                        epoch=epoch,
                        model=target_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        filename=checkpoint_path_file)
        logger.info('Model Saved at %s', checkpoint_path_file)
        torch.cuda.empty_cache()
    return


def main():
    if config.dataset.name == 'DatasetGenerator':
        data_loader = config.dataset(seed=args.seed, noise_rate=args.noise_rate, asym=args.asym)
    else:
        data_loader = config.dataset()

    model = config.model()
    if isinstance(data_loader, dataset.Clothing1MDatasetLoader):
        model.fc = torch.nn.Linear(2048, 14)
    model = model.to(device)

    data_loader = data_loader.getDataLoader()
    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    #data_train = data_loader['train_dataset'].dataset
    #tensor_list = []
    #for j in range(len(data_train)):
        #tensor_list.append(data_train[j][0])
    
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 40, gamma = 0.1)
    if config.criterion.name == 'NLNL':
        criterion = config.criterion(train_loader=data_loader['train_dataset'])
    else:
        criterion = config.criterion()
    trainer = Trainer(data_loader['train_dataset'], logger, config)
    evaluator = Evaluator(data_loader['test_dataset'], logger, config)

    starting_epoch = 0
    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'current_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'lid':[],
           'csr':[]}

    if args.load_model:
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     scheduler=scheduler)
        starting_epoch = checkpoint['epoch']
        ENV = checkpoint['ENV']
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file))
    
    idx = -5 if args.asym else -4
    if args.plot:
        lid_trend_through_training(exp_name=args.exp_name, dataset=args.config_path[8:idx], data_loader=data_loader, device=device, model=model, optimizer=optimizer, scheduler=scheduler, model_name=args.version, noise_type='sym', noise_ratio=args.noise_rate)
    elif args.plotall:
        lid_trend_of_learning_models(exp_name=args.exp_name, dataset=args.config_path[8:idx], model=model, optimizer=optimizer, scheduler=scheduler, model_list=['ce', 'fl', 'bl', 'bsl', 'bhl', 'd2l'], noise_ratio=args.noise_rate)
        test_acc_trend_of_learning_models(exp_name=args.exp_name, dataset=args.config_path[8:idx], model=model, optimizer=optimizer, scheduler=scheduler, model_list=['ce', 'fl', 'bl', 'bsl', 'bhl', 'd2l'], noise_ratio=args.noise_rate)
        csr_trend_of_learning_models(exp_name=args.exp_name, dataset=args.config_path[8:idx], model=model, optimizer=optimizer, scheduler=scheduler, model_list=['ce', 'fl', 'bl', 'bsl', 'bhl', 'd2l'], noise_ratio=args.noise_rate)
    else:  
        d2l_callback = D2LCallback(model, data_loader, device)
        train(starting_epoch, model, data_loader, optimizer, scheduler, criterion, trainer, evaluator, ENV, d2l_callback, mode='stage1')
        if args.version == 'd2l':
            checkpoint = util.load_model(filename=checkpoint_path_file,
                                         model=model,
                                         optimizer=optimizer,
                                         scheduler=scheduler)
            starting_epoch = checkpoint['epoch']
            ENV = checkpoint['ENV']
            trainer.global_step = ENV['global_step']
            logger.info("File %s loaded!" % (checkpoint_path_file))
        
            train(starting_epoch, model, data_loader, optimizer, scheduler, criterion, trainer, evaluator, ENV, d2l_callback, mode='stage2')
    
    
    return


if __name__ == '__main__':
    main()
