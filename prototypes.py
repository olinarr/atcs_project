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
    return MODELS_PATH + 'Prototypes' + ".pt"

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
    print(config.device)
    #model = FineTunedBERT(device = config.device, n_classes = None, trainable_layers = trainable_layers)#, use_classifier = False)
    model = FineTunedBERT(device = config.device, n_classes = 3, trainable_layers = trainable_layers)
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

def run_prototype(config, batch_managers, model):
    """
    Function to run a prototypical network

    Args:
      Config:         Contains all of the parameters defined in train.py
      batch_managers: Variable containing the batch managers

    Returns:
      Nthn: Not really something to declare yet
    """

    # TODO: Change all of this stuff to requirements for prototypes
    CLASSIFIER_DIMS = 768
    f_theta = BERT
    h_phi = nn.Linear(CLASSIFIER_DIMS, CLASSIFIER_DIMS).to(config.device)
    params = itertools.chain(
        f_theta.parameters(),
        h_phi.parameters()
    )
    beta = config.beta 
    alpha = config.alpha

    # Standard NN variables
    optimizer = AdamW(params, lr = beta)#TODO: parameters
    criterion = torch.nn.CrossEntropyLoss()
    
    # Episode Loader stuff
    SAMPLES_PER_EPISODE = 2
    NUM_WORKERS = 0
    episode_loader = EpisodeLoader.create_dataloader(
        config.samples_per_support, batch_managers, config.batch_size,
        samples_per_episode = SAMPLES_PER_EPISODE, 
        num_workers = NUM_WORKERS
    )

    # Run over episode loader
    for i, batch in enumerate(episode_loader):
        # batch is a list of length 4
        print(len(batch))
        
        for j, task_iter in enumerate(batch):
            print(task_iter)
            support_set = next(iter(task_iter))
            # Support_set is a tuple of len 2
            print(support_set)
            print(len(support_set))

            for step, batch in enumerate([support_set]):
                
                # Get inputs and targets
                inputs, targets = batch
                print(inputs)
                print(targets)

                # TODO: Figure out the number of classes thing
                outputs = model(inputs)
                loss = criterion(outputs, targets)


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
    #batchmanager3 = MRPCBatchManager(batch_size = config.samples_per_support, device = config.device)        

    #batchmanagers = [batchmanager1, batchmanager2, batchmanager3]
    batchmanagers = [batchmanager1, batchmanager2]

    # Train the model
    print('Beginning the training...', flush = True)
    #state_dict, dev_acc = protomaml(config, batchmanagers, BERT)
    run_prototype(config, batchmanagers, BERT)
    print("Finished the run_prototype function!")
    
    #print(f"#*#*#*#*#*#*#*#*#*#*#\nFINAL BEST DEV ACC: {dev_acc}\n#*#*#*#*#*#*#*#*#*#*#", flush = True)

    #save model
    #torch.save(state_dict, path_to_dicts(config))
