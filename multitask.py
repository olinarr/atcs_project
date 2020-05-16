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

from utils.episodeLoader import EpisodeLoader

from collections import defaultdict

from copy import deepcopy


# path of the trained state dict
MODELS_PATH = './state_dicts/'
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def path_to_dicts(config):
    return MODELS_PATH + "multitask.pt"

def k_shots(config, model, batchmanager, times = 10):

    task = config.eval_task
    bm = batchmanager.eval_batchmanagers[task]
    n_classes = len(bm.classes())

    original_model_dict = deepcopy(model.state_dict())

    globalOptimizer = SGD(model.globalParameters(), lr = config.lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_iterator = iter(bm.train_iter)

    test_acc = []
    for _ in range(times):

        model.addTask(task, n_classes)
        taskOptimizer = SGD(model.taskParameters(task), lr = config.lr)
        model.train()

        # it's shuffled, so we can simply do next
        batch = next(train_iterator)
        data, targets = batch

        for _ in range(config.k):

            globalOptimizer.zero_grad()
            taskOptimizer.zero_grad()

            # the model needs to know the task
            out = model(data, task)

            loss = criterion(out, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            globalOptimizer.step()
            taskOptimizer.step()

        model.eval()

        test_acc.append(get_dev_accuracy(model, task, batchmanager, test_set = True))
        print(test_acc[-1], 'debug print, remove me', flush = True)

        model.removeTask(task)

        model.load_state_dict(original_model_dict)


    return sum(test_acc) * 1. / len(test_acc)


def get_dev_accuracy(model, task, batchmanager, test_set=False):
    """compute dev accuracy on a certain task

    Parameters:
    model (MultiTaskBERT): the model to train
    task (str): the task on which to eval
    batchmanager ([MultiTaskTrainLoader, BatchManager]): the multi task batchmanager OR a single batch manager.

    Returns:
    float: said accuracy"""

    model.eval()
    count, num = 0., 0
    batchmanager = batchmanager.batchmanagers[task] if task in batchmanager.batchmanagers.keys() \
        else batchmanager.eval_batchmanagers[task]

    iter = batchmanager.test_iter if test_set else batchmanager.dev_iter

    with torch.no_grad():
        for batch in iter:  
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

    # if we evaluate only, model MUST be loaded.
    if config.k_shot_only:
        try :
            model.load_state_dict(torch.load(path_to_dicts(config), map_location = config.device))
        except Exception:
            print(f"WARNING: the `--k_shot_only` flag was passed, but `{path_to_dicts(config)}` was NOT found!")
            raise Exception()

    # if we saved the state dictionary, load it.
    if config.resume or config.k_shot_only:
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
    scheduler = get_linear_schedule_with_warmup(globalOptimizer, num_warmup_steps = 0, num_training_steps = len(batchmanager) * config.epochs)

    # TODO: task specific lr???

    taskOptimizer = {task: Adam(model.taskParameters(task), lr=config.lr) for task in batchmanager.tasks}

    ## PRINT ACCURACY ON ALL TASKS
    print("#########\nInitial dev accuracies: ")
    for task in batchmanager.tasks:
        dev_acc = get_dev_accuracy(model, task, batchmanager)
        print(f"dev acc of {task}: {dev_acc:.2f}", flush = True)
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
                    print(f'batch #{i}/{len(batchmanager)}:')
                    for task, value in sorted(loss_c.items()):
                        print(f'{task} avg_loss = ', end = '')
                        print(f'{sum(value) / len(value):.2f} ({len(value)} samples)')
                    # re-init cumulative loss
                    loss_c = defaultdict(lambda : [])                        

                    print("***", flush = True)

            # end of an epoch
            print(f'\n\n#*#*#*#*# Epoch {epoch+1} concluded! #*#*#*#*#')
            # print accuracies
            for task in batchmanager.tasks:
                dev_acc = get_dev_accuracy(model, task, batchmanager)
                print(f"{task} dev_acc = {dev_acc:.2f}.", flush = True)

            # zero-shot test on SICK dataset, TODO:should only be tested once after training
            for task in batchmanager.eval_batchmanagers:
                test_acc = get_dev_accuracy(model, task, batchmanager)
                print(f"{task} test_acc = {test_acc:.2f}.", flush = True)

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
    parser.add_argument('--k', type=int, default='4', help='How many times do we perform backprop on the evaluation?')
    parser.add_argument('--eval_task', type=str, default='SICK', help='Task to perform k-shot eval on')
    parser.add_argument('--eval_batch_size', type=int, default='8', help='Support size in k-shot training')
    parser.add_argument('--k_shot_only', action='store_true', help = 'Avoid training, load a model and evaluate it on the k-shot challenge')
    parser.add_argument('--force_cpu', action = 'store_true', help = 'force the use of the cpu')
    config = parser.parse_args()

    torch.manual_seed(config.random_seed)
    config.device = 'cuda:0' if torch.cuda.is_available() and not config.force_cpu else 'cpu'

    # iterating over this batchmanager yields 
    # batches from one of the datasets NLI, PBC or MRPC. 
    # ofc it's random and proportional to sizes
    batchmanager = MultiTaskTrainLoader(batch_size = config.batch_size, device = config.device, eval_batch_size = config.eval_batch_size)
    model = load_model(config, batchmanager)

    # Train the model
    if config.k_shot_only:
        print(k_shots(config, model, batchmanager))
    else:
        #train
        print('Beginning the training...', flush = True)
        state_dict = train(config, batchmanager, model)