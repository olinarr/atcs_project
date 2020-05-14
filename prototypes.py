import torch
import torch.nn as nn

import itertools

import os
import argparse

from utils.episodeLoader import EpisodeLoader
from modules.FineTunedBERT import FineTunedBERT
from modules.PrototypeModel import ProtoMODEL
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
    config: Contains all predefined argparse arguments

    Returns:
    ProtoMODEL: the loaded model for the prototype network"""

    # Retrieve all trainable layers for the model
    trainable_layers = [9, 10, 11]
    assert min(trainable_layers) >= 0 and max(trainable_layers) <= 11 # BERT has 12 layers!

    # Instantiate the prototype network model
    print(config.device)
    model = ProtoMODEL(device = config.device, trainable_layers = trainable_layers)

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

    # Parameters
    beta = config.beta
    params = model.parameters() 

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
        batch_loss = 0
        
        for j, task_iter in enumerate(batch):
            print(task_iter)
            support_set = next(iter(task_iter))
            # Support_set is a tuple of len 2
            print(len(support_set))

            for step, batch in enumerate([support_set]):
                
                # Get inputs and targets
                inputs, targets = batch
                classes = targets.unique()

                # Prototypes are mean of all support set points per class
                outputs = model(inputs)
                prototypes = torch.empty(len(classes), outputs.shape[1]).to(config.device)

                # Get prototype for each class and append
                for cls in classes:
                
                    # Subset the correct class and take mean over ClassBatch dimension
                    cls_idx = (targets == cls).nonzero().flatten()
                    cls_input = torch.index_select(outputs, dim = 0, index=cls_idx)
                    proto = cls_input.mean(dim=0)
                    prototypes[cls.item(), :] = proto

            # evaluate on query set (D_val) 
            for step, batch in enumerate(itertools.islice(task_iter, 1)):
                inputs, targets = batch
                outputs = model(inputs)

                # Calculate euclidean distance in a vectorized way
                diffs = outputs.unsqueeze(1) - prototypes.unsqueeze(0)
                distances = torch.sum(diffs*diffs, -1) * -1 # get negative distances
                loss = criterion(distances, targets)
                batch_loss += loss.item()

                model.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"The loss for this batch is {loss:.4f}")


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

    model = load_model(config)

    batchmanager1 = MultiNLIBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager2 = IBMBatchManager(batch_size = config.samples_per_support, device = config.device)
    #batchmanager3 = MRPCBatchManager(batch_size = config.samples_per_support, device = config.device)        

    #batchmanagers = [batchmanager1, batchmanager2, batchmanager3]
    batchmanagers = [batchmanager1, batchmanager2]

    # Train the model
    print('Beginning the training...', flush = True)
    #state_dict, dev_acc = protomaml(config, batchmanagers, BERT)
    run_prototype(config, batchmanagers, model)
    print("Finished the run_prototype function!")
    
    #print(f"#*#*#*#*#*#*#*#*#*#*#\nFINAL BEST DEV ACC: {dev_acc}\n#*#*#*#*#*#*#*#*#*#*#", flush = True)

    #save model
    #torch.save(state_dict, path_to_dicts(config))
