import torch
import torch.nn as nn
import torch.optim as optim

import itertools as it

import os, sys
import argparse
import random 
import numpy

from copy import copy, deepcopy
from collections import defaultdict

from utils.episodeLoader import EpisodeLoader
from modules.ProtoMAML import ProtoMAML
from utils.batchManagers import MultiNLIBatchManager, IBMBatchManager, MRPCBatchManager, PDBBatchManager, SICKBatchManager

from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category = UserWarning)

# path of the trained state dict
MODELS_PATH = './state_dicts/'
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def path_to_dict(config):
    if config.resume_with == None:
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        model_name = 'ProtoMAML{}.pt'.format(current_time)
    else:
        model_name = config.resume_with
    
    return os.path.join(config.model_save_dir, model_name)

def get_accuracy(model, batchmanager, test_set=False, nr_of_batches=sys.maxsize):
    """compute dev or test accuracy on a certain task
    Parameters:
    model (MultiTaskBERT): the model to train
    batchmanager (MultiTaskTrainLoader, BatchManager): the multi task batchmanager OR a single batchManagers
    Returns:
    float: said accuracy"""

    model.eval()
    count, num = 0., 0

    iter = batchmanager.test_iter if test_set else batchmanager.dev_iter

    with torch.no_grad():
        for step, batch in enumerate(iter):
            data, targets = batch
            out = model(data)
            predicted = out.argmax(dim=1)
            count += (predicted == targets).sum().item()
            num += len(targets)

            if step == nr_of_batches:
                break

    model.train()
    return count / num

def load_model(config):
    """Load a model (either a new one or from disk)

    Parameters:
    config: argparse flags

    Returns:
    FineTunedBERT: the loaded model"""


    trainable_layers = [9, 10, 11]
    assert len(trainable_layers) == 0 or min(trainable_layers) >= 0 and max(trainable_layers) <= 11 # BERT has 12 layers!

    # n_classes = None cause we don't use it
    # use_classifier = False cause we don't use it
    model = ProtoMAML(device = config.device, trainable_layers = trainable_layers)

    # if we saved the state dictionary, load it.
    if config.resume_with != None:
        try :
            print(f"Loading `{path_to_dict(config)}`.")
            model.load_state_dict(torch.load(path_to_dict(config), map_location = config.device))
        except Exception:
            print(f"WARNING: the `--resume_with` flag was passed, but `{path_to_dict(config)}` was NOT found!")

    return model


def protomaml(config, sw, model_init, train_bms, val_bms, test_bms):
    
    model_episode = type(model_init)(device=model_init.device) 
    CLASSIFIER_DIMS = 768
    
    beta = config.beta
    alpha = config.alpha
    
    optimizer = AdamW(model_init.parameters(), lr=beta)
    criterion = torch.nn.CrossEntropyLoss()    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = config.warmup, num_training_steps = config.nr_epochs * config.nr_episodes)

    NUM_WORKERS = 3 
    train_episodes = iter(EpisodeLoader.create_dataloader(
        config.samples_per_support, train_bms, config.batch_size,
        num_workers = NUM_WORKERS
    ))

    global_step = 0

    def do_epoch(episode_loader, config, mode='train'):
        assert mode in ['train', 'val', 'test']
        train = mode == 'train'
        nonlocal global_step

        totals = {}
        ns = {}
        def log(value, bm, avg, extra="", name='loss'):
            tag = '{}/{}{}/{}'.format(mode, bm.name, extra, name)
            sw.add_scalar(tag, value, global_step)
            if hasattr(bm, 'parent'):
                log(value, bm.parent, avg, extra="_stX", name=name)
            if avg:
                totals[name] = value if name not in totals else totals[name] + value
                ns[name] = 1 if name not in ns else ns[name] + 1

        for i, batch in enumerate(it.islice(episode_loader, config.nr_episodes)):

            # external data structured used to accumulate gradients.
            accumulated_gradients = defaultdict(lambda : None)   
            
            for j, (support_iter, query_iter, bm) in enumerate(batch):

                print(f'batch {i}, task {j} : {bm.name}', flush = True)
                
                support_set = next(iter(support_iter))

                # [1] Clone model for this episode and calculate parameters for softmax.
                classes = bm.classes()
                model_init.deactivate_linear_layer()
                model_episode.deactivate_linear_layer()
                
                weights = deepcopy(model_init.state_dict())
                model_episode.load_state_dict(weights)
                
                model_init.generateParams(support_set, classes)
                model_episode.ffn_W = deepcopy(model_init.ffn_W)
                model_episode.ffn_b = deepcopy(model_init.ffn_b)
                model_episode.zero_grad()

                # [2] Adapt parameters on support set.
                task_optimizer = optim.SGD(model_episode.parameters(), lr=alpha)
                task_criterion = torch.nn.CrossEntropyLoss()
                for step, batch in enumerate([support_set] * config.k):
                    batch_inputs, batch_targets = batch

                    out = model_episode(batch_inputs)
                    loss = task_criterion(out, batch_targets)
                    
                    task_optimizer.zero_grad()
                    loss.backward()
                    task_optimizer.step()

                    if step == 0:
                        log(loss.item(), bm, not train, name='loss_1')
                    elif step == config.k-1:
                        log(loss.item(), bm, not train, name='loss_k')

                    if train:
                        global_step += 1
   
                if not config.skip_prototypes:
                    ffn_W = model_init.original_W + (model_episode.ffn_W - model_init.original_W).detach()
                    ffn_b = model_init.original_b + (model_episode.ffn_b - model_init.original_b).detach()
                    # First delete the nn.Parameter and replace with regular tensor,
                    # this will make gradients flow back to orignal model too.
                    del model_episode.ffn_W
                    del model_episode.ffn_b
                    model_episode.ffn_W = ffn_W
                    model_episode.ffn_b = ffn_b

                # [3] Evaluate adapted params on query set, calc grads.            
                for step, batch in enumerate(it.islice(query_iter, 1)):
                    batch_inputs, batch_targets = batch
                    out = model_episode(batch_inputs)

                    loss = task_criterion(out, batch_targets)

                    task_optimizer.zero_grad()
                    model_init.zero_grad()
                    loss.backward()

                    log(loss.item(), bm, not train, name='loss_q')

                    if train:
                        global_step += 1


                def accumulate_gradients(model_):
                    # accumulate the gradients
                    for n, p in model_.named_parameters():
                        if p.requires_grad and n not in ('ffn_W','ffn_b'):
                            if accumulated_gradients[n] is None:
                                accumulated_gradients[n] = p.grad.detach().clone()
                            else:
                                accumulated_gradients[n] += p.grad.detach().clone()

                accumulate_gradients(model_episode)
                if not config.skip_prototypes:
                    accumulate_gradients(model_init)
               
                # during validation/test we also measure performance on entire test set of task
                if not train:
                    acc = get_accuracy(model_episode, bm, True)
                    log(acc, bm, True, name='acc_test')


                # end of inner loop

            model_init.deactivate_linear_layer()
            if train: # and thus not validation/test
                optimizer.zero_grad()
                # load the accumulated gradients and optimize
                for n, p in model_init.named_parameters():
                    if p.requires_grad:
                        p.grad = accumulated_gradients[n]
                optimizer.step()
                scheduler.step()

        for key, total in totals.items():
            log(total/ns[key], bm, False, name=key+'/avg')

        return { key : total/ns[key] for key,total in totals.items() }
 
    val_episodes = iter(EpisodeLoader.create_dataloader(
        config.samples_per_support, val_bms, config.nr_val_trials
    ))

    val_config = deepcopy(config)
    val_config.nr_episodes = 1 # we do 1 episode with config.nr_val_trials (32) batches of the val task

    filename = path_to_dict(config)

    epochs_since = 0
    best_acc = 0
    for epoch in range(config.nr_epochs):
        
        # validate 
        print('validating...')
        results = do_epoch(val_episodes, val_config, mode='val')
        
        if results['acc_test'] > best_acc:
            best_acc = results['acc_test']
            torch.save(model_init.state_dict(), filename)
            print("New best acc found at {}, written model to {}".format(best_acc, filename))
            epochs_since = 0
        else:
            epochs_since += 1
            if epochs_since >= 6:
                print(f"no improvement for {epochs_since}-th time.")
                break

        # train
        print('training...')
        do_epoch(train_episodes, config)

    test_episodes = iter(EpisodeLoader.create_dataloader(
        config.samples_per_support, test_bms, 8*config.nr_val_trials # do a lot of trials for test to get stddev down.
    ))

    # test
    print('testing...')
    best_weights = torch.load(filename)
    model_init.load_state_dict(best_weights)
    results = do_epoch(test_episodes, val_config, mode='test')
    print('results:')
    print(results)
                    
 
###########

def logloc(comment='',dir_name='runs'):
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(dir_name, current_time + '_' + socket.gethostname() + comment)
    return log_dir   
        
if __name__ == "__main__":
     # Parse training configuration
    parser = argparse.ArgumentParser()

    # Training params
    parser.add_argument('--nr_episodes', type=int, help='Number of episodes in an epoch', default = 25)
    parser.add_argument('--nr_epochs', type=int, help='Number of epochs', default = 80)
    parser.add_argument('--batch_size', type=int, default="64", help="How many tasks in an episode over which gradients for M_init are accumulated")
    parser.add_argument('--k', type=int, default="3", help="How many times do we update weights prime")
    parser.add_argument('--random_seed', type=int, default="42", help="Random seed")
    parser.add_argument('--beta', type=float, help='Beta learning rate', default = 5e-5)
    parser.add_argument('--alpha', type=float, help='Alpha learning rate', default = 1e-3)
    parser.add_argument('--warmup', type=float, help='For how many episodes we do warmup on meta-optimization.', default = 100)
    parser.add_argument('--samples_per_support', type=int, help='Number of samples to draw from the support set.', default = 32)
    parser.add_argument('--skip_prototypes', action='store_true')

    # Misc
    parser.add_argument('--nr_val_trials', type=int, help='Over how many k-shots on validation task we average.', default = 32)
    #parser.add_argument('--loss_print_rate', type=int, default='250', help='Print loss every')
    parser.add_argument('--model_save_dir', type=str, default=MODELS_PATH, help='The directory in which to store the models.')
    parser.add_argument('--resume_with', type=str, default=None, help='Resume training with this state_dict stored in model_save_dir instead of restarting')
    parser.add_argument('--sw_log_dir', type=str, default='runs', help='The directory in which to create the default logdir.')
    parser.add_argument('--device', type=str, help='')
    
    config = parser.parse_args()
    print(config) #print config, so we have 'paper trail' where we can make double sure what paramaters caused a result.

    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    numpy.random.seed(config.random_seed)

    if not config.device:
        config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("Running on: {}".format(config.device))
    print("Encoding: {} (should probably be UTF-8)".format(sys.stdout.encoding), flush=True)

    model = load_model(config)

    batchmanager1 = MultiNLIBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager2 = IBMBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager3 = MRPCBatchManager(batch_size = config.samples_per_support, device = config.device)        
    batchmanager4 = PDBBatchManager(batch_size = config.samples_per_support, device = config.device)        
    batchmanager5 = SICKBatchManager(batch_size = config.samples_per_support, device = config.device)

    pdb_subtasks = list(batchmanager4.get_subtasks(2))
    mnli_subtasks = list(batchmanager1.get_subtasks(2))

    # Double the weighting of tasks that aren't represented twice (normal, binary-sub-tasks).
    batchmanager3.weight_factor *= 2 # (only original)
    for bm in pdb_subtasks:
        bm.weight_factor *= 2        # (only subtasks)

    # MultiNLI, MRPC, PDB for training.
    train_bms = [ batchmanager1, batchmanager3 ]
    train_bms.extend(mnli_subtasks)
    train_bms.extend(pdb_subtasks)

    # SICK for validation
    val_bms = [ batchmanager5 ] 
    
    # IBM for test
    test_bms = [ batchmanager2 ]

    logdir = logloc(dir_name=config.sw_log_dir)
    sw = SummaryWriter(log_dir=logdir)

    # Train the model
    protomaml(config, sw, model, train_bms, val_bms, test_bms)
    
