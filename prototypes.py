import torch
import torch.nn as nn

import itertools as it

import os
import argparse
import csv

from utils.episodeLoader import EpisodeLoader
from modules.FineTunedBERT import FineTunedBERT
from modules.PrototypeModel import ProtoMODEL
from utils.batchManagers import MultiNLIBatchManager, IBMBatchManager, MRPCBatchManager, PDBBatchManager, SICKBatchManager

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category = UserWarning) 

# path of the trained state dict
MODELS_PATH = './state_dicts/'
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

###############
### GLOBALS ###
###############
# To keep track of # of batches processed
global_step = 0
csv_file_name = "" # will be set in main function
###############

def path_to_dicts(config):
    return MODELS_PATH + 'Prototypes' + ".pt"

def load_model(config, trainable_layers = [9,10,11]):
    """Load a model (either a new one or from disk)

    Parameters:
    config: Contains all predefined argparse arguments

    Returns:
    ProtoMODEL: the loaded model for the prototype network"""

    # Make sure we have valid layers in layer list
    assert min(trainable_layers) >= 0 and max(trainable_layers) <= 11 # BERT has 12 layers!

    # Instantiate the prototype network model
    print(config.device)
    model = ProtoMODEL(device = config.device, trainable_layers = trainable_layers)

    return model

def get_test_acc(config, val_bm, model, prototypes):
    """
    Function to get accuracy on validation task
    """
    count, num = 0., 0
    model.eval()
    with torch.no_grad():

        for batch in val_bm.test_iter:

            inputs, targets = batch

            distances = compute_distances(model, prototypes, batch)
            preds = distances.argmax(dim=1)

            count += (preds == targets).sum().item()
            num += len(targets)

    model.train()

    return count / num

def k_shot_test(config, val_loader, val_bms, model, optimizer, test = False):
    """
    Function that retrains with k samples 

    Args:
      Config: Contains all user-defined (hyper)parameters
      Val_loader: Loader with the correct number of samples per label for the test task
      val_bms: Batchmanager for validation task 
      Model, optimizer, criterion are standard NN parameters
      num_times: How many times we perform the k sample update
      test: Toggles whether we're evaluating for test set or for validation set
    """

    # Iterate for how_many_updates we update

    test_acc = []

    assert len(val_bms) == 1
    val_bm = val_bms[0]

    val_episodes = iter(val_loader)
    episode_iter = next(val_episodes)[0][0]

    # bitch = None
    # targets = None
    # for t, batch in enumerate(episode_iter):
    #     inputs, targets = batch
    #     bitch = batch
    # batch = bitch 


    # # DO K UPDATES
    # k = 10
    # criterion = torch.nn.CrossEntropyLoss()
    # for i in range(k):

    #     prototypes = compute_prototypes(model, batch)
    #     distances = compute_distances(model, prototypes, batch)

    #     model.zero_grad()
    #     loss = criterion(distances, targets)

    #     loss.backward()
    #     optimizer.step()

    with torch.no_grad():

        for t, batch in enumerate(episode_iter):

            # repeat experiment t times
            if t >= config.nr_val_experiments:
                break

            prototypes = compute_prototypes(model, batch)
            test_acc.append(get_test_acc(config, val_bm, model, prototypes))

            print(f'experiment {t+1}, test acc: {test_acc[-1]:.2f}', flush = True)

    # Print the list with validation accuracy
    test_acc = torch.tensor(test_acc)
    mean, std = torch.mean(test_acc), torch.std(test_acc)

    if test:
        return test_acc
    print(f'mean: {mean:.2f}, std: {std:.2f}', flush = True)

    # Write stuff to csv 
    write_to_csv(test_acc)

    # 
    return 

def write_to_csv(test_acc):
    """
    Function that writes dev accuracies to a specific CSV file
    Args:
      test_acc: List with 10 accuracies and std
    """

    # Test acc has tensors, so get items
    test_list = []
    for tensor in test_acc:
        test_list.append(tensor.item())
    test_list.append(torch.std(test_acc).item())

    # Actual writing of ints to csv
    with open(csv_file_name, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(test_list) 

def run_prototype(config, train_bms, model, val_bms, sw, test_bm):
    """
    Function to run a prototypical network

    Args:
      Config:    Contains all of the parameters defined in train.py
      train_bms: Variable containing the batch managers for training data
      model    : The model that we are using, in this case the ProtoMODEL
      val_bms  : Batch managers for the validation task
      sw       : Tensorboard Summary Writer

    Returns:
      Nthn: Not really something to declare yet
    """
    
    # Make a train loader with training tasks
    NUM_WORKERS = 0
    train_loader = EpisodeLoader.create_dataloader(
        config.samples_per_support, 
        train_bms, 
        config.batch_size,
        num_workers = NUM_WORKERS
    )

    # Make a validation loader with validation task
    val_loader = EpisodeLoader.create_dataloader(
        config.samples_per_support,
        val_bms,
        config.batch_size,
        shuffle_labels = False, # in validation, we dont wanna
        num_workers=NUM_WORKERS
    )

    # Make a test loader for test task
    test_loader = EpisodeLoader.create_dataloader(
        config.samples_per_support,
        test_bm,
        config.batch_size,
        shuffle_labels = False, # in validation, we dont wanna
        num_workers=NUM_WORKERS
    )

    # Parameters
    params = model.parameters() 

    # Standard NN variables
    optimizer = AdamW(params, lr = config.lr)#TODO: parameters
    criterion = torch.nn.CrossEntropyLoss()

    # Iterate over epochs
    for i in range(config.epochs):

        if i < 1:
            model.eval()
            print("NEXT RESULT IS FIRST RANDOM DEV EPOCH")
            k_shot_test(config, val_loader, val_bms, model, optimizer)
        # Training
        model.train()
        run_epoch(config, train_loader, model, optimizer, criterion, val_loader = val_loader, sw = sw)

        # Do the k-shot test thing and evaluate
        model.eval()
        k_shot_test(config, val_loader, val_bms, model, optimizer)

        # # Validation
        # model.eval()
        # with torch.no_grad():
        #     run_epoch(config, val_loader, model, optimizer, criterion, val_loader = val_loader, sw = sw)

    # Report final performance on test set 
    test_acc = k_shot_test(config, test_loader, test_bm, model, optimizer, test = True)
    mean, std = torch.mean(test_acc), torch.std(test_acc)
    test_list = []
    for tensor in test_acc:
        test_list.append(tensor.item())
    print("Final performance on test set: ")
    print(f"Mean: {mean.item():.2f}, std: {std.item():.2f}")
    print(test_list)

    # Write test row as last row to csv
    write_to_csv(test_acc)

def run_epoch(config, episode_loader, model, optimizer, criterion, val_loader, sw):
    """
    Function to run a full epoch for either training or validation

    Args:
      Config:     Argparse object containing all parameters
      Loader:     Episodeloader for training data or validation data
      Model :     The model to be used (e.g. ProtoMODEL)
      Optimizer:  The optimizer to be used (default: ADAM)
      Criterion:  Criterion to calculate loss (default: CrossEntropyLoss)
      val_loader: Loader to get episodes for validation data for accuracy
      sw        : Tensorboard Summary Writer

    Returns:
      Something
    """

    # To keep track of losses over batch
    losses = []

    try:
        # Run over episode loader
        for i, batch in enumerate(it.islice(episode_loader, config.nr_episodes)):
            #print(i, flush = True)
            run_batch(config, episode_loader, model, batch, optimizer, criterion, sw = sw)

            # if i % config.dev_acc_print_rate == 10000:
            #     dev_acc = get_dev_acc(config, val_loader, model, optimizer, criterion)
            #     print(f"DEV ACC IS: {dev_acc:.3f}")

    # .. so we can interrupt and still save a model
    except KeyboardInterrupt:
        print("Training stopped by KeyboardInterrupt!", flush = True)
        torch.save(model.state_dict(), path_to_dicts(config))

def compute_prototypes(model, batch):
    """ Compute prototypes """

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

    return prototypes

def compute_distances(model, prototypes, batch):
    """ Computes the distances between all vectors to all prototypes """
    inputs, targets = batch

    outputs = model(inputs)

    # Calculate euclidean distance in a vectorized way
    diffs = outputs.unsqueeze(1) - prototypes.unsqueeze(0)
    distances = torch.sum(diffs*diffs, -1) * -1 # get negative distances

    return distances


def run_batch(config, episode_loader, model, batch, optimizer, criterion, sw = None):
    """
    Function to process a single batch
    """

    # batch is a list of length 4
    batch_loss = 0
    
    # episodeloader returns tuples of supp iter, query iter, batchmanager
    for j, (support_iter, query_iter, bm) in enumerate(batch):
        support_set = next(iter(support_iter))
        query_set = next(iter(query_iter))
        # Support_set is a tuple of len 2
        #print(len(support_set))
            
        # Get inputs and targets
        prototypes = compute_prototypes(model, support_set)

        # evaluate on query set (D_val) 
        for step, batch in enumerate([query_set]):
            inputs, targets = batch
            distances = compute_distances(model, prototypes, batch)

            loss = criterion(distances, targets)
            batch_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()

    #print(f"The loss for this batch is {batch_loss:.4f}")
    
    # Write to tensorboard
    if sw != None:
        # This is basically training data, so be careful with interpreting
        global global_step
        global_step += 1
        sw.add_scalar('batch/acc', batch_loss, global_step)
    

if __name__ == "__main__":
     # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--random_seed', type=int, default="42", help="Random seed")
    parser.add_argument('--resume', action='store_true', help='resume training instead of restarting')
    parser.add_argument('--lr', type=float, help='learning rate', default = 2e-5)
    parser.add_argument('--nr_episodes', type=int, help='Number of episodes in an epoch', default = 25)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default = 1)
    parser.add_argument('--samples_per_support', type=int, help='Number of samples per each episode', default = 32)
    parser.add_argument('--nr_val_experiments', type=int, help='How many times we perform validation on SICK', default = 10)

    ############################################
    ### Parameters for hyperparameter search ###
    ############################################

    #parser.add_argument('--random_seed', type=int, default="42", help="Random seed")
    parser.add_argument('--val_task',  type=str, default="IBM" , help="Value for the validation task")
    parser.add_argument('--test_task', type=str, default="SICK", help="Value for the test task")
    
    # Model hyperparams
    parser.add_argument('--num_layers', type=int, default=3, help="Number of BERT layers to fine-tune")

    ############################################

    # Retrieve argparse object and report some details
    config = parser.parse_args()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", config.device)
    print("Number of episodes: ", config.nr_episodes)

    # Set the number of layers going down from 11 to 11-num_layers
    trainable_layers = []
    for i in reversed(range(11+1 - config.num_layers,11+1)):
        trainable_layers.append(i)
    assert len(trainable_layers) == config.num_layers

    print("trainable_layers: ", trainable_layers)

    # Instantiate model
    model = load_model(config, trainable_layers = trainable_layers)

    # All batchmanagers
    batchmanager1 = MultiNLIBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager2 = IBMBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager3 = MRPCBatchManager(batch_size = config.samples_per_support, device = config.device)        
    batchmanager4 = PDBBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager5 = SICKBatchManager(batch_size = config.samples_per_support, device = config.device)    

    #train_bms = [batchmanager5]
    #train_bms = [batchmanager5]
    #train_bms = [batchmanager2]
    #train_bms = [batchmanager4.get_subtasks(2)]
    train_bms = []
    #train_bms.extend(batchmanager4.get_subtasks(2))
    #train_bms.extend([batchmanager2])
    train_bms.extend(batchmanager1.get_subtasks(2))
    #train_bms.extend(batchmanager4.get_subtasks(2))
    #train_bms.extend(batchmanager3.get_subtasks(2))

    """ Batchmanagers for validation and test task """
    val_bms = [batchmanager5] # SICK
    test_bm = [batchmanager2] # IBM 

    # To write results to sw and csv
    sw = SummaryWriter()

    # Be wary that csv_file_name is a global but it's in the same "block" as this
    if not os.path.exists("csv"):
        os.makedirs("csv")

    # Get model- and csv file name
    model_name = os.path.join("csv", "lr"+str(config.lr) + 
                              "_val" + config.val_task +
                              "_test" + config.test_task +
                              "_layers" + str(len(trainable_layers)))
    csv_file_name = model_name + ".csv"       

    # Create column headers in csv file
    with open(csv_file_name, "w", newline= "") as file:

        # numbers + "std" as header
        write_list = []
        for i in range(1, config.nr_val_experiments + 1):
            write_list.append(i)
        write_list.append("std")

        # Do the actual writing
        writer = csv.writer(file)
        writer.writerow(write_list) 

    # Train the model
    print('Beginning the training...', flush = True)
    #state_dict, dev_acc = protomaml(config, batchmanagers, BERT)
    run_prototype(config, train_bms, model, val_bms, sw, test_bm)
    print("Finished the run_prototype function!")

    #save model
    model_path = os.path.join("state_dicts", model_name + ".pth")
    torch.save(model.state_dict(), model_path)
