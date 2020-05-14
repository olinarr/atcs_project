# Main TODO (decreasing order of urgence):
# * have more samples in D_train than in D_val |D_train| > |D_val|
# * figure out how to make gradients flow through the parameter generation OR repeat that operation to accumulate the gradients
# * implement second order (perhaps)

import torch
import torch.nn as nn
import torch.optim as optim

import itertools as it

import os
import argparse
from copy import deepcopy
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


def protomaml(config, sw, batch_managers, model, val_bms):

    CLASSIFIER_DIMS = 768
    
    # TODO learnable alpha, beta learning rates?
    beta = config.beta
    alpha = config.alpha
    
    optimizer = AdamW(model.parameters(), lr=beta)
    criterion = torch.nn.CrossEntropyLoss()    

    # for protomaml we use two samples (support and query)
    SAMPLES_PER_EPISODE = 2
    
    NUM_WORKERS = 3 
    train_episodes = EpisodeLoader.create_dataloader(
        config.samples_per_support, batch_managers, config.batch_size,
        num_workers = NUM_WORKERS
    )

    global_step = 0

    def do_epoch(episode_loader, epoch_length, train=True):
        nonlocal global_step

        for i, batch in enumerate (it.islice(episode_loader, epoch_length)):
            optimizer.zero_grad()

            # external data structured used to accumulate gradients.
            accumulated_gradients = defaultdict(lambda : None)   
            
            # save original parameters. Will be reloaded later.
            original_weights = deepcopy(model.state_dict())

            for j, (support_iter, query_iter, bm) in enumerate(batch):

                print(f'batch {i}, task {j} : {bm.name}', flush = True)
                
                support_set = next(iter(support_iter))

                # [1] Calculate parameters for softmax.
                # TODO: make gradients flow inside this function. (Or create them again)
                classes = bm.classes()
                model.generateParams(support_set, classes)
            
                def log(loss, step, bm):
                    tbname = '{}/{}/loss'.format('train' if train else 'val', bm.name)
                    sw.add_scalar(tbname+'@{}'.format(step), loss, global_step)
                    if hasattr(bm, 'parent'):
                        log(loss, step, bm.parent)

                # [2] Adapt task-specific parameters
                task_optimizer = optim.SGD(model.parameters(), lr=alpha)
                task_criterion = torch.nn.CrossEntropyLoss()
                for step, batch in enumerate([support_set] * config.k):
                    batch_inputs, batch_targets = batch

                    out = model(batch_inputs)
                    loss = task_criterion(out, batch_targets)
                    
                    task_optimizer.zero_grad()
                    loss.backward()
                    task_optimizer.step()

                    if step == 0:
                        log(loss.item(), '1', bm)
                    elif step == config.k-1:
                        log(loss.item(), 'k', bm)

                    global_step += 1


                # [3] Evaluate adapted params on query set, calc grads.            
                for step, batch in enumerate(it.islice(query_iter, 1)):
                    batch_inputs, batch_targets = batch
                    out = model(batch_inputs)

                    loss = task_criterion(out, batch_targets)

                    model.zero_grad()
                    loss.backward()

                    log(loss.item(), 'q', bm)
                    global_step += 1


                # accumulate the gradients
                for n, p in model.named_parameters():
                    if p.requires_grad and n not in ('FFN.weight', 'FFN.bias'):
                        if accumulated_gradients[n] is None:
                            accumulated_gradients[n] = p.grad.data
                        else:
                            accumulated_gradients[n] += p.grad.data


                # return to original model
                model.revert_state(original_weights)

                # end of inner loop

           
            if train:
                # load the accumulated gradients and optimize
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        p.grad.data = accumulated_gradients[n]

                optimizer.step()
 
    val_episodes = EpisodeLoader.create_dataloader(
        config.samples_per_support, val_bms, 1
    )

    for epoch in range(config.nr_epochs):
        
        # train
        print('training...')
        do_epoch(train_episodes, config.nr_episodes)

        # validate
        print('validating...')
        do_epoch(val_episodes, 1, train=False)


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
    parser.add_argument('--nr_episodes', type=int, help='Number of episodes in an epoch', default = 100)
    parser.add_argument('--nr_epochs', type=int, help='Number of epochs', default = 20)
    parser.add_argument('--batch_size', type=int, default="4", help="How many tasks in an episode over which gradients for M_init are accumulated")
    parser.add_argument('--k', type=int, default="4", help="How many times do we update weights prime")
    parser.add_argument('--random_seed', type=int, default="42", help="Random seed")
    parser.add_argument('--resume', action='store_true', help='resume training instead of restarting')
    parser.add_argument('--beta', type=float, help='Beta learning rate', default = 5e-5)
    parser.add_argument('--alpha', type=float, help='Alpha learning rate', default = 1e-2)
    parser.add_argument('--samples_per_support', type=int, help='Number of samples to draw from the support set.', default = 32)
    parser.add_argument('--use_second_order', action='store_true', help='Use the second order version of MAML')

    #TODO use a learning rate decay?

    # Misc
    #parser.add_argument('--loss_print_rate', type=int, default='250', help='Print loss every')
    parser.add_argument('--sw_log_dir', type=str, default='runs', help='The directory in which to create the default logdir.')
    
    config = parser.parse_args()
    config.first_order_approx = not config.use_second_order

    torch.manual_seed(config.random_seed)
    config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = load_model(config)

    batchmanager1 = MultiNLIBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager2 = IBMBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager3 = MRPCBatchManager(batch_size = config.samples_per_support, device = config.device)        
    batchmanager4 = PDBBatchManager(batch_size = config.samples_per_support, device = config.device)        
    batchmanager5 = SICKBatchManager(batch_size = config.samples_per_support, device = config.device)

    train_bms = [ batchmanager2 ]
    train_bms.extend(batchmanager1.get_subtasks(2))
    train_bms.extend(batchmanager4.get_subtasks(2))
    #TODO decide on final mix of tasks in training.

    val_bms = [ batchmanager5 ]
    #val_bms.extend(batchmanager5.get_subtasks(2))


    logdir = logloc(dir_name=config.sw_log_dir)
    sw = SummaryWriter(log_dir=logdir)

    # Train the model
    state_dict, dev_acc = protomaml(config, sw, train_bms, model, val_bms)
    
    print(f"#*#*#*#*#*#*#*#*#*#*#\nFINAL BEST DEV ACC: {dev_acc}\n#*#*#*#*#*#*#*#*#*#*#", flush = True)

    #save model
    torch.save(state_dict, path_to_dicts(config))
