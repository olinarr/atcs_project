import torch
import torch.nn as nn

import itertools

import os
import argparse

from utils.episodeLoader import EpisodeLoader
from modules.FineTunedBERT import FineTunedBERT
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
    model = FineTunedBERT(device = config.device, n_classes = None, trainable_layers = trainable_layers, use_classifier = False)

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


def protomaml(config, batch_managers, BERT):

    CLASSIFIER_DIMS = 768
    
    # TODO figure out proper strategy for saving/loading these 
    # meta-learning models.
    
    # initialization
    f_theta = BERT
    h_phi = nn.Linear(CLASSIFIER_DIMS, CLASSIFIER_DIMS).to(config.device)

    params = itertools.chain(
        f_theta.parameters(),
        h_phi.parameters()
    )

    # TODO learnable alpha, beta learning rates?
    beta = config.beta
    alpha = config.alpha
    
    optimizer = AdamW(params, lr=beta)
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
        
        for j, task_iter in enumerate(batch):

            print(f'episode {i}, task {j}', flush = True)

            # k     samples per tasks
            # t     length of sequence per sample
            # d     features per sequence-element (assuming same in and out)

            support_set = next(iter(task_iter))

            # [1] Calculate parameters for softmax.

            with torch.no_grad():
            
                batch_input = support_set[0]            # d x t x k <-- TODO this is not the dimension!
                batch_target = support_set[1]           # k

                # encode sequences 
                batch_input = f_theta(batch_input)      # k x d
                
                classes = batch_target.unique()         # N
                W = []
                b = []
                for cls in classes:
                    cls_idx   = (batch_target == cls).nonzero()
                    cls_input = torch.index_select(batch_input, dim=0, index=cls_idx.flatten())
                                                        # C x d
                                    
                    # prototype is mean of support samples (also c_k)
                    prototype = cls_input.mean(dim=0)         # d
                    
                    # see proto-maml paper, this corresponds to euclidean distance
                    W.append(2 * prototype)
                    b.append(- prototype.norm() ** 2)
                
                # the transposes are because the dimensions were wrong!
                W = torch.stack(W).t()         # d x C
                b = torch.stack(b)
                # h_phi, W, b together now make up the classifier.
        
            
            # [2] Adapt task-specific parameters

            h_phi_prime = nn.Linear(CLASSIFIER_DIMS, CLASSIFIER_DIMS).to(config.device)
            h_phi_prime.load_state_dict(h_phi.state_dict())

            f_theta_prime = load_model(config)
            f_theta_prime.load_state_dict(f_theta.state_dict())
            # not this simple...
            # TODO Make shallow copy of state_dict, and then replace
            # references of task-specific parameters with those to
            # clones? then use that state dict for a new 
            # f_theta_prime instance of model?
            # Use Tensor.clone so we can backprop through it and 
            # update f_theta as well (if no first order approx).
    
            W.requires_grad, b.requires_grad = True, True
            params = [
                f_theta_prime.parameters(), 
                h_phi_prime.parameters(),
                [W, b]
            ]
            params = itertools.chain(*params)

            task_optimizer = AdamW(params, lr=alpha)
            task_criterion = torch.nn.CrossEntropyLoss()    
            
            def process_batch(batch):
                # pass to device!
                batch_input, batch_target = batch
                batch_output = f_theta_prime(batch_input)       # k x d
                batch_output = h_phi_prime(batch_output)        # k x l
                # TODO add nonlinearity?
                batch_output = batch_output @ W + b             # k x N
                # TODO no softmax, right? it's already in CELoss
                # batch_output = F.softmax(batch_output, dim=0)   # k x N
                loss = task_criterion(batch_output, batch_target)
                return loss

            for step, batch in enumerate([support_set]):

                loss = process_batch(batch)
                
                task_optimizer.zero_grad() 
                loss.backward()
                task_optimizer.step()

                # TODO logging

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
                f_theta_prime.zero_grad()
                h_phi_prime.zero_grad()
                W.grad.data.zero_()
                b.grad.data.zero_() # TODO can we avoid these last two?

                loss = process_batch(batch)
                loss.backward()


            # if we use first-order approximation, copy gradients to originals manually.
            # TODO test if this actually works.
            if config.first_order_approx:
                with torch.no_grad():
                    pairs = [(f_theta_prime, f_theta), (h_phi_prime, h_phi)]
                    for prime, original in pairs:
                        params = zip(prime.named_parameters(), original.named_parameters())
                        for (pNamePr, pValuePr), (pNameOr, pValueOr) in params:
                            if pNamePr != pNameOr:
                                raise Error("Order in which named parameters are returned is probably not deterministic? \n names: {}, {}"
                                           .format(pNamePr, pNameOr))
                            if pValuePr.requires_grad:
                                assert pValueOr.requires_grad # check we did not add new parameters
                                pValueOr.grad = pValuePr.grad if pValueOr.grad is None else pValueOr.grad + pValuePr.grad
            else:
                raise NotImplementedError()

            # end of inner loop

        # we have now accumulated gradients in the originals
        #TODO divide gradients by number of tasks? or just have learning rate be lower?
        optimizer.step()

    
    
        
if __name__ == "__main__":
     # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--batch_size', type=int, default="4", help="Batch size")
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

    BERT = load_model(config)

    batchmanager1 = MultiNLIBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager2 = IBMBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager3 = MRPCBatchManager(batch_size = config.samples_per_support, device = config.device)        

    batchmanagers = [batchmanager1, batchmanager2, batchmanager3]

    # Train the model
    print('Beginning the training...', flush = True)
    state_dict, dev_acc = protomaml(config, batchmanagers, BERT)
    
    print(f"#*#*#*#*#*#*#*#*#*#*#\nFINAL BEST DEV ACC: {dev_acc}\n#*#*#*#*#*#*#*#*#*#*#", flush = True)

    #save model
    torch.save(state_dict, path_to_dicts(config))