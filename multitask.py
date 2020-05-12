import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.utils.data as data

import random

from modules.MultiTaskBERT import MultiTaskBERT
from utils.batchManagers import MultiNLIBatchManager, IBMBatchManager, MRPCBatchManager, PDBBatchManager, SICKBatchManager

# path of the trained state dict
MODELS_PATH = './state_dicts/'
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def path_to_dicts(config):
    return MODELS_PATH + "multitask.pt"

def get_accuracy(model, iter, task):
    """compute accuracy on a certain iterator

    Parameters:
    model (MultiTaskBERT): the model to train
    iter (MyIterator): iterator on a set
    task (str): name of the task

    Returns:
    float: said accuracy"""

    model.eval()
    count, num = 0., 0
    with torch.no_grad():
        for i, batch in enumerate(iter):
            data, targets = batch
            out = model(data, task)
            predicted = out.argmax(dim=1)
            count += (predicted == targets).sum().item()
            num += len(targets)

    model.train()
    return count / num

def load_model(config, batchmanagers):
    """Load a model (either a new one or from disk)

    Parameters:
    config: argparse flags
    batchmanagers (dict(str, BatchManager)): the batchmanagers with names

    Returns:
    MultiTaskBERT: the loaded model"""

    # some datasets have 3 classes, some other 2!
    trainable_layers = [9, 10, 11]
    assert min(trainable_layers) >= 0 and max(trainable_layers) <= 11 # BERT has 12 layers!
    tasks = [(name, len(bm.classes())) for name, bm in batchmanagers.items()]
    model = MultiTaskBERT(device = config.device, trainable_layers = trainable_layers, tasks = tasks)

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

def getProportions(batchmanagers):
    """Define the training proportions

    Parameters:
    batchmanagers (dict(str, BatchManager)): batchamangers with name

    Returns:
    (list(str)): list representing the proportions"""

    min_size = min((bm.task_size() for bm in batchmanagers.values()))
    proportions = []
    for name, bm in batchmanagers.items():
        size = round(bm.task_size() / min_size)
        proportions += [name] * size

    return proportions

def train(config, batchmanagers, model):
    """Main training loop

    Parameters:
    config: argparse flags
    batchmanagers (dict(str, BatchManager)): batchamangers with name
    model (MultiTaskBERT): the model to train

    Returns:
    (dict, float): the state dictionary of the best model and its dev accuracy"""

    dev_iter = batchmanagers['NLI'].dev_iter

    iter_dataloaders = {task:iter(bm.train_iter) for task, bm in batchmanagers.items()}
    proportions = getProportions(batchmanagers)
    totalBatches = sum((bm.task_size() for bm in batchmanagers.values())) // config.batch_size

    model.train()

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # filter out from the optimizer the "frozen" parameters,
    # which are the parameters without requires grad.
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = totalBatches * config.epochs)

    # compute initial dev accuracy (to check whether it is 1/n_classes)
    last_dev_acc = get_accuracy(model, dev_iter, 'NLI')
    best_dev_acc = last_dev_acc # to save the best model
    best_model_dict = model.state_dict() # to save the best model
    print(f'inital dev accuracy: {last_dev_acc}', flush = True)

    try :
        for epoch in range(config.epochs):
            loss_c = 0.
            for i in range(totalBatches):
                task = random.choice(proportions)
                dataloader = iter_dataloaders[task]

                try:
                    batch = next(dataloader)
                except StopIteration:
                    dataloaders[task] = iter(batchamangers[task].train_iter)
                    dataloader = iter_dataloaders[task]
                    batch = next(dataloader)

                optimizer.zero_grad()

                data, targets = batch
                out = model(data, task)

                loss = criterion(out, targets)
                loss.backward()

                loss_c += loss.item()

                if i != 0 and i % config.loss_print_rate == 0:
                    print(f'epoch #{epoch+1}/{config.epochs}, batch #{i}/{len(batchmanager.train_iter)}: loss = {loss.item()}', flush = True)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step() # update lr

            # end of an epoch
            print(f'#####\nEpoch {epoch+1} concluded!\n')
            print(f'Average train loss: {loss_c / len(batchmanager.train_iter)}')
            print(f'Average train acc : {get_accuracy(model, dev_iter, "NLI")}')
            new_dev_acc = get_accuracy(model, batchmanager.dev_iter)
            last_dev_acc = new_dev_acc

            print(f'dev accuracy: {new_dev_acc}')
            print('#####', flush = True)

            # if it improves, this is the best model
            if new_dev_acc > best_dev_acc:
                best_dev_acc = new_dev_acc
                best_model_dict = model.state_dict()

    except KeyboardInterrupt:
        print("Training stopped!")
        new_dev_acc = get_accuracy(model, dev_iter, 'NLI')
        print(f'Recomputing dev accuracy: {new_dev_acc}')
        if new_dev_acc > best_dev_acc:
            best_dev_acc = new_dev_acc
            best_model_dict = model.state_dict()

    return best_model_dict, best_dev_acc

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--batch_size', type=int, default="32", help="Batch size")
    parser.add_argument('--random_seed', type=int, default="42", help="Random seed")
    parser.add_argument('--resume', action='store_true', help='resume training instead of restarting')
    parser.add_argument('--lr', type=float, help='Learning rate', default = 2e-5)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default = 25)
    parser.add_argument('--loss_print_rate', type=int, default='250', help='Print loss every')
    parser.add_argument('--force_cpu', action = 'store_true', help = 'force the use of the cpu')
    config = parser.parse_args()

    torch.manual_seed(config.random_seed)
    config.device = 'cuda:0' if torch.cuda.is_available() and not config.force_cpu else 'cpu'

    batchmanagers = {
        'NLI'  : MultiNLIBatchManager(config.batch_size, config.device),
        'IBM'  : IBMBatchManager(config.batch_size, config.device),
        'PDB'  : PDBBatchManager(config.batch_size, config.device),
        'MRPC' : MRPCBatchManager(config.batch_size, config.device)
    }

    model = load_model(config, batchmanagers)

    # Train the model
    print('Beginning the training...', flush = True)
    state_dict, dev_acc = train(config, batchmanagers, model)
    print(f"#*#*#*#*#*#*#*#*#*#*#\nFINAL BEST DEV ACC: {dev_acc}\n#*#*#*#*#*#*#*#*#*#*#", flush = True)

    #save model
    torch.save(state_dict, path_to_dicts(config))