# Main TODO (decreasing order of urgence):
# * have more samples in D_train than in D_val |D_train| > |D_val|
# * figure out how to make gradients flow through the parameter generation OR repeat that operation to accumulate the gradients
# * implement second order (perhaps)

import torch
import torch.nn as nn
import torch.optim as optim

import itertools as it

import os, sys
import argparse
import random 

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

def path_to_dicts(config):
    return MODELS_PATH + 'ProtoMAML' + ".pt"

def load_model(config):
    """Load a model (either a new one or from disk)

    Parameters:
    config: argparse flags

    Returns:
    FineTunedBERT: the loaded model"""


    trainable_layers = [9, 10, 11]
    assert min(trainable_layers) >= 0 and max(trainable_layers) <= 11 # BERT has 12 layers!

    # n_classes = None cause we don't use it
    # use_classifier = False cause we don't use it
    model = ProtoMAML(device = config.device, trainable_layers = trainable_layers)

    """# if we saved the state dictionary, load it.
    if config.resume:
        try :
            model.load_state_dict(torch.load(path_to_dicts(config), map_location = config.device))
        except Exception:
            print(f"WARNING: the `--resume` flag was passed, but `{path_to_dicts(config)}` was NOT found!")
    else:
        if os.path.exists(path_to_dicts(config)):
            print(f"WARNING: `--resume` flag was NOT passed, but `{path_to_dicts(config)}` was found!")"""

    return model


def protomaml(config, sw, batch_managers, model_init, val_bms):

    CLASSIFIER_DIMS = 768
    
    # TODO learnable alpha, beta learning rates?
    beta = config.beta
    alpha = config.alpha
    
    optimizer = AdamW(model_init.parameters(), lr=beta)
    criterion = torch.nn.CrossEntropyLoss()    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 100, num_training_steps = config.nr_epochs * config.nr_episodes)

    NUM_WORKERS = 3 
    train_episodes = iter(EpisodeLoader.create_dataloader(
        config.samples_per_support, batch_managers, config.batch_size,
        num_workers = NUM_WORKERS
    ))

    global_step = 0

    def do_epoch(episode_loader, config, train=True):
        nonlocal global_step

        totals = {}
        avg_n = 0
        def log(loss, step, bm, avg):
            nonlocal avg_n
            tbname = '{}/{}/loss'.format('train' if train else 'val', bm.name)
            tag = tbname+'@{}'.format(step)
            sw.add_scalar(tag, loss, global_step)
            if hasattr(bm, 'parent'):
                log(loss, step, bm.parent, avg)
            if avg:
                totals[step] = loss if step not in totals else totals[step] + loss
                avg_n += 1

        for i, batch in enumerate (it.islice(episode_loader, config.nr_episodes)):
            optimizer.zero_grad()

            # external data structured used to accumulate gradients.
            accumulated_gradients = defaultdict(lambda : None)   
            
            for j, (support_iter, query_iter, bm) in enumerate(batch):

                print(f'batch {i}, task {j} : {bm.name}', flush = True)
                
                support_set = next(iter(support_iter))

                # [1] Calculate parameters for softmax.
                # TODO: make gradients flow inside this function. (Or create them again)
                classes = bm.classes()
                del model_init.FFN
                model_init.generateParams(support_set, classes)
           
                original_weights = deepcopy(model_init.state_dict())
                model_episode = copy(model_init)
                model_episode.revert_state(original_weights)
                
                # [2] Adapt task-specific parameters
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
                        log(loss.item(), '1', bm, not train)
                    elif step == config.k-1:
                        log(loss.item(), 'k', bm, not train)

                    global_step += 1
    
                # this will make gradients flow back to orignal model too.
                model_episode.FFN.weight = nn.Parameter(model_episode.prototypes + (model_episode.FFN.weight - model_episode.prototypes).detach_())
                model_episode.FFN.bias = nn.Parameter(model_episode.prototype_norms + (model_episode.FFN.bias - model_episode.prototype_norms).detach_())

                # [3] Evaluate adapted params on query set, calc grads.            
                for step, batch in enumerate(it.islice(query_iter, 1)):
                    batch_inputs, batch_targets = batch
                    out = model_episode(batch_inputs)

                    loss = task_criterion(out, batch_targets)

                    model_episode.zero_grad()
                    loss.backward()

                    log(loss.item(), 'q', bm, not train)
                    global_step += 1


                def accumulate_gradients(model, skip_ffn=True):
                    # accumulate the gradients
                    for n, p in model.named_parameters():
                        if p.requires_grad and not (skip_ffn and n in ('FFN.weight','FFN.bias')):
                            if accumulated_gradients[n] is None:
                                accumulated_gradients[n] = p.grad.data
                            else:
                                accumulated_gradients[n] += p.grad.data

                accumulate_gradients(model_episode, skip_ffn=False)
                accumulate_gradients(model_init)

                # end of inner loop

            if train: # and thus not validation
                # load the accumulated gradients and optimize
                for n, p in model_init.named_parameters():
                    if p.requires_grad:
                        p.grad.data = accumulated_gradients[n]
                optimizer.step()
                scheduler.step()
        
        for key, total in totals.items():
            log(total/avg_n, key+'/avg', bm, False)

        return { key : total/avg_n for key,total in totals.items() }
 
    val_episodes = iter(EpisodeLoader.create_dataloader(
        config.samples_per_support, val_bms, 16 # make parameter or define as constant
    ))

    val_config = deepcopy(config)
    val_config.nr_episodes = 1
    val_config.alpha *= 10
    #val_config.k = 

    best_loss = sys.maxsize
    for epoch in range(config.nr_epochs):
        
        # train
        print('training...')
        do_epoch(train_episodes, config)

        # validate
        print('validating...')

        results = do_epoch(val_episodes, 1, train=False)
        if results['q'] < best_loss:
            best_loss = results['q']
            torch.save(model_init.state_dict(), path_to_dicts(config))


    model.deactivate_linear_layer()
    return model.state_dict(), None
 
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
    parser.add_argument('--nr_epochs', type=int, help='Number of epochs', default = 160)
    parser.add_argument('--batch_size', type=int, default="16", help="How many tasks in an episode over which gradients for M_init are accumulated")
    parser.add_argument('--k', type=int, default="4", help="How many times do we update weights prime")
    parser.add_argument('--random_seed', type=int, default="42", help="Random seed")
    parser.add_argument('--resume', action='store_true', help='resume training instead of restarting')
    parser.add_argument('--beta', type=float, help='Beta learning rate', default = 5e-5)
    parser.add_argument('--alpha', type=float, help='Alpha learning rate', default = 5e-4)
    parser.add_argument('--samples_per_support', type=int, help='Number of samples to draw from the support set.', default = 32)


    # Misc
    #parser.add_argument('--loss_print_rate', type=int, default='250', help='Print loss every')
    parser.add_argument('--sw_log_dir', type=str, default='runs', help='The directory in which to create the default logdir.')
    parser.add_argument('--device', type=str, help='')
    
    config = parser.parse_args()

    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)

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

    train_bms = [ batchmanager2, batchmanager3 ]
    train_bms.extend(batchmanager1.get_subtasks(2))
    train_bms.extend(batchmanager4.get_subtasks(2))
    #TODO decide on final mix of tasks in training.

    val_bms = [ batchmanager5 ]

    logdir = logloc(dir_name=config.sw_log_dir)
    sw = SummaryWriter(log_dir=logdir)

    # Train the model
    state_dict, dev_acc = protomaml(config, sw, train_bms, model, val_bms)
    
    print(f"#*#*#*#*#*#*#*#*#*#*#\nFINAL BEST DEV ACC: {dev_acc}\n#*#*#*#*#*#*#*#*#*#*#", flush = True)

    #save model
    torch.save(state_dict, path_to_dicts(config))
