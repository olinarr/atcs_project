import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.utils.data as data
from torch.optim import Adam, SGD

from modules.MultiTaskBERT import MultiTaskBERT
from utils.MultiTaskTrainLoader import MultiTaskTrainLoader
from utils.batchManagers import SICKBatchManager

from utils.episodeLoader import EpisodeLoader

from collections import defaultdict

from copy import deepcopy


# path of the trained state dict
MODELS_PATH = './state_dicts/'
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def path_to_dicts(config):
    return MODELS_PATH + "multitask.pt"

def get_dev_accuracy(model, task, batchmanager):
    """compute dev accuracy on a certain task

    Parameters:
    model (MultiTaskBERT): the model to train
    task (str): the task on which to eval
    batchmanager (MultiTaskTrainLoader): the batchmanager

    Returns:
    float: said accuracy"""

    model.eval()
    count, num = 0., 0
    dev_iter = batchmanager.batchmanagers[task].dev_iter
    with torch.no_grad():
        for batch in dev_iter:
            data, targets = batch
            out = model(data, task)
            predicted = out.argmax(dim=1)
            count += (predicted == targets).sum().item()
            num += len(targets)

    model.train()
    return count / num


def load_model(config, batchmanager):
    """Load a model (either a new one or from disk)

    Parameters:
    config: argparse flags
    batchmanager (MultiTaskTrainLoader): the batchmanager

    Returns:
    MultiTaskBERT: the loaded model"""
    
    # this function returns a dictionary mapping
    # name of the task (string) --> number of classes in the task (int)
    tasks = batchmanager.getTasksWithNClasses()
    # this "tasks" object is used to initialize the model (with the right output layers)
    model = MultiTaskBERT(device = config.device, tasks = tasks)

    # if we saved the state dictionary, load it.
    if config.resume:
        try :
            model.load_state_dict(torch.load(path_to_dicts(config), map_location = config.device))
        except Exception:
            print(f"WARNING: the `--resume` flag was passed, but `{path_to_dicts(config)}` was NOT found!")
    else:
        if os.path.exists(path_to_dicts(config)):
            print(f"WARNING: `--resume` flag was NOT passed, but `{path_to_dicts(config)}` was found!")   

    return model

def train(config, batchmanager, model):
    """Main training loop

    Parameters:
    config: argparse flags
    batchmanager (MultiTaskTrainLoader): the batchmanager
    model (MultiTaskBERT): the model to train

    Returns:
    (dict, float): the state dictionary of the final model"""

    model.train()

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # global optimizer.
    globalOptimizer = AdamW(model.globalParameters(), lr=config.lr)    
    # scheduler for lr
    scheduler = get_linear_schedule_with_warmup(globalOptimizer, num_warmup_steps = 0, num_training_steps = batchmanager.totalBatches * config.epochs)

    # TODO: task specific lr???

    taskOptimizer = {task: Adam(model.taskParameters(task), lr=config.lr) for task in batchmanager.tasks}

    ## PRINT ACCURACY ON ALL TASKS
    print("#########\nInitial dev accuracies: ")
    for task in batchmanager.tasks:
        dev_acc = get_dev_accuracy(model, task, batchmanager)
        print(f"dev acc of {task}: {dev_acc:.2f}")
    print("#########", flush = True)

    try :
        for epoch in range(config.epochs):
            # cumulative loss for printing
            loss_c = defaultdict(lambda : [])
            # iterating over the batchmanager returns
            # name_of_task, batch_of_the_task
            for i, (task, batch) in enumerate(batchmanager):

                globalOptimizer.zero_grad()
                taskOptimizer[task].zero_grad()

                data, targets = batch
                # the model needs to know the task
                out = model(data, task)

                loss = criterion(out, targets)
                loss.backward()

                # cumulative loss (for printing)
                loss_c[task].append(loss.item())

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                globalOptimizer.step()
                taskOptimizer[task].step()
                scheduler.step() # update lr

                # printing: we print the avg loss for every
                # task in the last loss_print_rate steps.
                if i != 0 and i % config.loss_print_rate == 0:
                    print(f'epoch #{epoch+1}/{config.epochs}, ', end = '')
                    print(f'batch #{i}/{batchmanager.totalBatches}:')
                    for task, value in sorted(loss_c.items()):
                        print(task + ": ")
                        print(f'avg_loss = ', end = '')
                        print(f'{sum(value) / len(value):.2f} ({len(value)} samples)')
                    # re-init cumulative loss
                    loss_c = defaultdict(lambda : [])                        

                    print("***", flush = True)

            # end of an epoch
            print(f'\n\n#*#*#*#*# Epoch {epoch+1} concluded! #*#*#*#*#')
            # print accuracies
            for task in batchmanager.tasks:
                dev_acc = get_dev_accuracy(model, task, batchmanager)
                print(f"{task} dev_acc = {dev_acc:.2f}.")
            print(f'#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*\n\n', flush = True)
            torch.save(model.state_dict(), path_to_dicts(config))

    except KeyboardInterrupt:
        print("Training stopped!", flush = True)
        torch.save(model.state_dict(), path_to_dicts(config))

    return model.state_dict()


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--batch_size', type=int, default="32", help="Batch size")
    parser.add_argument('--random_seed', type=int, default="42", help="Random seed")
    parser.add_argument('--resume', action='store_true', help='resume training instead of restarting')
    parser.add_argument('--lr', type=float, help='Learning rate', default = 2e-5)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default = 10)
    parser.add_argument('--loss_print_rate', type=int, default='250', help='Print loss every')
    parser.add_argument('--force_cpu', action = 'store_true', help = 'force the use of the cpu')
    config = parser.parse_args()

    torch.manual_seed(config.random_seed)
    config.device = 'cuda:0' if torch.cuda.is_available() and not config.force_cpu else 'cpu'

    # iterating over this batchmanager yields 
    # batches from one of the datasets NLI, PBC or MRPC. 
    # ofc it's random and proportional to sizes
    batchmanager = MultiTaskTrainLoader(batch_size = config.batch_size, device = config.device)
    model = load_model(config, batchmanager)

    # Train the model
    print('Beginning the training...', flush = True)
    state_dict = train(config, batchmanager, model)