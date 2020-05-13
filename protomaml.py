# Main TODO (decreasing order of urgence):
# * biggest issue right now: support and query might have different number of labels!!
# * labels shuffling
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
from utils.batchManagers import MultiNLIBatchManager, IBMBatchManager, MRPCBatchManager, PDBBatchManager

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


def protomaml(config, sw, batch_managers, model):

    CLASSIFIER_DIMS = 768
    
    # TODO learnable alpha, beta learning rates?
    beta = config.beta
    alpha = config.alpha
    
    optimizer = AdamW(model.parameters(), lr=beta)
    criterion = torch.nn.CrossEntropyLoss()    

    # for protomaml we use two samples (support and query)
    SAMPLES_PER_EPISODE = 2
    
    NUM_WORKERS = 3 
    episode_loader = EpisodeLoader.create_dataloader(
        config.samples_per_support, batch_managers, config.batch_size,
        num_workers = NUM_WORKERS
    )

    global_step = 0
    for i, batch in enumerate(episode_loader):        
        optimizer.zero_grad()

        # external data structured used to accumulate gradients.
        accumulated_gradients = defaultdict(lambda : None)   
        
        for j, (support_iter, query_iter, bm) in enumerate(batch):

            print(f'batch {i}, task {j} : {bm.name}', flush = True)

            # save original parameters. Will be reloaded later.
            original_weights = deepcopy(model.state_dict())

            support_set = next(iter(support_iter))

            # [1] Calculate parameters for softmax.
            # TODO: make gradients flow inside this function. (Or create them again)
            classes = bm.classes()
            model.generateParams(support_set, classes)
           

            tbname = 'train/{}/loss'.format(bm.name)

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
                    sw.add_scalar(tbname+'@1', loss.item(), global_step)
                if step == config.k-1:
                    sw.add_scalar(tbname+'@k', loss.item(), global_step)

                global_step += 1


            # [3] Evaluate adapted params on query set, calc grads.            
            for step, batch in enumerate(it.islice(query_iter, 1)):
                batch_inputs, batch_targets = batch
                out = model(batch_inputs)

                loss = task_criterion(out, batch_targets)

                model.zero_grad()
                loss.backward()

                sw.add_scalar(tbname+'@q', loss.item(), global_step)
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

        
        # load the accumulated gradients and optimize
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.grad.data = accumulated_gradients[n]

        optimizer.step()


    model.deactivate_linear_layer()
    return model.state_dict(), None
    
        
if __name__ == "__main__":
     # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--batch_size', type=int, default="4", help="How many tasks in an episode over which gradients for M_init are accumulated")
    parser.add_argument('--k', type=int, default="8", help="How many times do we update weights prime")
    parser.add_argument('--random_seed', type=int, default="42", help="Random seed")
    parser.add_argument('--resume', action='store_true', help='resume training instead of restarting')
    parser.add_argument('--beta', type=float, help='Beta learning rate', default = 2e-5)
    parser.add_argument('--alpha', type=float, help='Alpha learning rate', default = 2e-3)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default = 25)
    parser.add_argument('--samples_per_support', type=int, help='Number of samples to draw from the support set.', default = 32)
    parser.add_argument('--use_second_order', action='store_true', help='Use the second order version of MAML')
    #parser.add_argument('--loss_print_rate', type=int, default='250', help='Print loss every')

    config = parser.parse_args()

    config.first_order_approx = not config.use_second_order

    torch.manual_seed(config.random_seed)
    config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = load_model(config)

    batchmanager1 = MultiNLIBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager2 = IBMBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager3 = MRPCBatchManager(batch_size = config.samples_per_support, device = config.device)        
    batchmanager4 = PDBBatchManager(batch_size = config.samples_per_support, device = config.device)        

    #batchmanagers = [batchmanager1, batchmanager2, batchmanager3]
    #batchmanagers.extend(batchmanager4.get_subtasks(2))
    batchmanagers = [batchmanager1]

    #TODO decide on final mix of tasks in training.

    sw = SummaryWriter()


    # Train the model
    print('Beginning the training...', flush = True)
    state_dict, dev_acc = protomaml(config, sw, batchmanagers, model)
    
    print(f"#*#*#*#*#*#*#*#*#*#*#\nFINAL BEST DEV ACC: {dev_acc}\n#*#*#*#*#*#*#*#*#*#*#", flush = True)

    #save model
    torch.save(state_dict, path_to_dicts(config))
