# Main TODO (decreasing order of urgence):
# * biggest issue right now: support and query might have different number of labels!!
# * labels shuffling
# * have more samples in D_train than in D_val |D_train| > |D_val|
# * figure out how to make gradients flow through the parameter generation OR repeat that operation to accumulate the gradients
# * implement second order (perhaps)

import torch
import torch.nn as nn
import torch.optim as optim

import itertools

import os
import argparse
from copy import deepcopy
from collections import defaultdict

from utils.episodeLoader import EpisodeLoader
from modules.ProtoMAML import ProtoMAML
from utils.batchManagers import MultiNLIBatchManager, IBMBatchManager, MRPCBatchManager

from transformers import AdamW, get_linear_schedule_with_warmup

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


def protomaml(config, batch_managers, model):

    CLASSIFIER_DIMS = 768
    
    # TODO figure out proper strategy for saving/loading these 
    # meta-learning models.

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
        samples_per_episode = SAMPLES_PER_EPISODE, 
        num_workers = NUM_WORKERS
    )

    for i, batch in enumerate(episode_loader):        

        # a batch of episodes
        
        optimizer.zero_grad()

        # external data structured used to accumulate gradients.
        # TODO: are we sure we are accumulating the gradients
        # across different tasks?

        accumulated_gradients = defaultdict(lambda : None)   
        
        for j, (task_iter, bm) in enumerate(batch):

            # save original parameters. Will be reloaded later.

            print(f'batch {i}, task {j}', flush = True)

            original_weights = deepcopy(model.state_dict())

            # k     samples per tasks
            # t     length of sequence per sample
            # d     features per sequence-element (assuming same in and out)

            support_set = next(iter(task_iter))

            # [1] Calculate parameters for softmax.

            # TODO: make gradients flow inside this function. (Or create them again)
            classes = bm.classes()
            model.generateParams(support_set, classes)
            
            # [2] Adapt task-specific parameters

            # not this simple...
            # TODO Make shallow copy of state_dict, and then replace
            # references of task-specific parameters with those to
            # clones? then use that state dict for a new 
            # f_theta_prime instance of model?
            # Use Tensor.clone so we can backprop through it and 
            # update f_theta as well (if no first order approx).
    
            """W.requires_grad, b.requires_grad = True, True
            params = [
                f_theta_prime.parameters(), 
                h_phi_prime.parameters(),
                [W, b]
            ]
            params = itertools.chain(*params)"""

            task_optimizer = optim.SGD(model.parameters(), lr=alpha)
            task_criterion = torch.nn.CrossEntropyLoss()

            for step, batch in enumerate([support_set] * config.k):

                batch_inputs, batch_targets = batch

                out = model(batch_inputs)
                loss = task_criterion(out, batch_targets)
                
                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()

            # [3] Evaluate adapted params on query set, calc grads.
            
            if config.first_order_approx:
                # We are using the approximation so we only backpropagate to the
                # 'prime' models and will then copy gradients to originals later.
                
                # TODO set parameters in f_theta_prime, h_phi_prime, 
                # to have requires_grad = True
                pass
            else:
                raise NotImplementedError()
                # Backpropagate all the way back to originals (through optimization
                # on the support set and thus requiring second order gradients.)

                # TODO set parameters in f_theta, h_phi (originals)
                # to have requires_grad = True
                
                
            # evaluate on query set (D_val) 
            for step, batch in enumerate(itertools.islice(task_iter, 1)):
                batch_inputs, batch_targets = batch
                out = model(batch_inputs)

                loss = task_criterion(out, batch_targets)

                model.zero_grad()
                loss.backward()

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

        # we have now accumulated gradients in the originals
        # TODO divide gradients by number of tasks? or just have learning rate be lower?

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
    parser.add_argument('--batch_size', type=int, default="4", help="Batch size")
    parser.add_argument('--k', type=int, default="4", help="How many times do we update weights prime")
    parser.add_argument('--random_seed', type=int, default="42", help="Random seed")
    parser.add_argument('--resume', action='store_true', help='resume training instead of restarting')
    parser.add_argument('--beta', type=float, help='Beta learning rate', default = 2e-5)
    parser.add_argument('--alpha', type=float, help='Alpha learning rate', default = 2e-5)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default = 25)
    parser.add_argument('--samples_per_support', type=int, help='Number of samples per each episode', default = 8)
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

    batchmanagers = [batchmanager1, batchmanager2, batchmanager3]

    # Train the model
    print('Beginning the training...', flush = True)
    state_dict, dev_acc = protomaml(config, batchmanagers, model)
    
    print(f"#*#*#*#*#*#*#*#*#*#*#\nFINAL BEST DEV ACC: {dev_acc}\n#*#*#*#*#*#*#*#*#*#*#", flush = True)

    #save model
    torch.save(state_dict, path_to_dicts(config))
