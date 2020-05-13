import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.utils.data as data
from torch.optim import Adam

from modules.MultiTaskBERT import MultiTaskBERT
from utils.MultiTaskTrainLoader import MultiTaskTrainLoader
from utils.BatchManagers import SICKBatchManager

from utils.episodeLoader import EpisodeLoader


# path of the trained state dict
MODELS_PATH = './state_dicts/'
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def path_to_dicts(config):
    return MODELS_PATH + "multitask.pt"


def evaluateModel(model, episodeLoader):
    """compute accuracy on a certain iterator

    Parameters:
    model (MultiTaskBERT): the model to evaluate
    devEpisodeLoader (EpisodeLoader): episode loader used to validate

    Returns:
    float: loss on new task """

    # TODO: finish here

    model.eval()
    model.initTask('SICK', 5)
    with torch.no_grad():
        for i, batch in enumerate(iter):
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

    trainable_layers = [9, 10, 11]
    assert min(trainable_layers) >= 0 and max(trainable_layers) <= 11 # BERT has 12 layers!
    # this method retuns a list of tuples (str, int) of form (name_of_task, n_of_classes)
    tasks = batchmanager.getTasksWithNClasses()
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

def train(config, batchmanager, model, devEpisodeLoader):
    """Main training loop

    Parameters:
    config: argparse flags
    batchmanager (MultiTaskTrainLoader): the batchmanager
    model (MultiTaskBERT): the model to train
    devEpisodeLoader (EpisodeLoader): episode loader used to validate

    Returns:
    (dict, float): the state dictionary of the best model and its dev accuracy"""

    model.train()

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # filter out from the optimizer the "frozen" parameters,
    # which are the parameters without requires grad.
    globalOptimizer = AdamW(model.parameters(), lr=config.lr)    
    scheduler = get_linear_schedule_with_warmup(globalOptimizer, num_warmup_steps = 0, num_training_steps = batchmanager.totalBatches * config.epochs)

    # TODO: task specific lr

    taskOptimizer = {task: Adam(model.taskParameters(task), lr=config.lr) for task in batchmanager.tasks}

    # compute initial dev accuracy (to check whether it is 1/n_classes)
    """last_dev_acc = get_accuracy(model, dev_iter, 'NLI')
    best_dev_acc = last_dev_acc # to save the best model
    best_model_dict = model.state_dict() # to save the best model
    print(f'inital dev accuracy: {last_dev_acc}', flush = True)"""

    # TODO check all the parameters are registered correctly.

    try :
        for epoch in range(config.epochs):
            for i, (task, batch) in enumerate(batchmanager):

                print(i, flush = True)

                globalOptimizer.zero_grad()
                taskOptimizer[task].zero_grad()

                data, targets = batch
                out = model(data, task)

                loss = criterion(out, targets)
                loss.backward()

                print(loss.item(), flush = True) # remove this later

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                globalOptimizer.step()
                taskOptimizer[task].step()
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

    batchmanager = MultiTaskTrainLoader(batch_size = config.batch_size, device = config.device)
    model = load_model(config, batchmanager)

    # stuff to do evaluatoon
    # TO VALIDATE
    SICK = SICKBatchManager(batch_size = 8, device = config.device)
    devEpisodeLoader = EpisodeLoader.create_dataloader(
            8, [SICK], config.batch_size,
            samples_per_episode = 2, 
            num_workers = 2
        )

    # Train the model
    print('Beginning the training...', flush = True)
    state_dict, dev_acc = train(config, batchmanager, model, devEpisodeLoader)
    print(f"#*#*#*#*#*#*#*#*#*#*#\nFINAL BEST DEV ACC: {dev_acc}\n#*#*#*#*#*#*#*#*#*#*#", flush = True)

    #save model
    torch.save(state_dict, path_to_dicts(config))