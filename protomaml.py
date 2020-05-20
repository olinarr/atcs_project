# Main TODO (decreasing order of urgence):
# * have more samples in D_train than in D_val |D_train| > |D_val|
# * figure out how to make gradients flow through the parameter generation OR repeat that operation to accumulate the gradients
# * implement second order (perhaps)

import torch
import torch.nn as nn
import torch.optim as optim

import itertools as it

import os, sys
import argparse
import random 
import numpy

from copy import copy, deepcopy
from collections import defaultdict

from utils.episodeLoader import EpisodeLoader
from modules.ProtoMAML import ProtoMAML
from utils.batchManagers import MultiNLIBatchManager, IBMBatchManager, MRPCBatchManager, PDBBatchManager, SICKBatchManager

from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category = UserWarning)

# path of the trained state dict
MODELS_PATH = './state_dicts/'
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def path_to_dicts(config):
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    return os.path.join(config.model_save_dir,'ProtoMAML{}.pt'.format(current_time))

def k_shots(config, model, val_episodes, val_bms, times = 10):
    """ Run k-shot evaluation. 

    Parameters:
    config (Object): argparse flags
    model (protoMAML): the model to k-shot evaluate
    val_episodes (Iterator): the episodeLoader iterator
    val_bms (list(BatchManager)): evaluatio batch managers
    times (int): average over how many times?

    Returns:
    float: test acc mean over 'times' times
    float: test acc std over 'times' times """

    assert not hasattr(model, 'FFN'), "You are k-shot testing a model with a linear layer. Are you sure you meant that?"
    assert len(val_bms) == 1, "As of now, this test is thought to be only with one BMs (SICK)"

    bm = val_bms[0]
    # why / 2? Because as of now, all train tasks are binary, so we have samples_per_support / 2 examples per label
    # TODO change this to be a parameter: examples per support
    examples_per_label = config.samples_per_support // 2

    print(f"Running {config.k}-shot evaluation on task {bm.name}, averaged over {times} times. Number of examples per class: {examples_per_label}.", flush= True)

    # this object will yield balanced support sets of size k_dim
    episode_iter = next(val_episodes)[0][0]

    # save model, so we can revert
    original_model_dict = deepcopy(model.state_dict())

    criterion = torch.nn.CrossEntropyLoss()

    # saved as a list, so we can get mean and std
    test_acc = []
    # repeat the experiment 'times' times...
    for t, batch in enumerate(episode_iter):

        # only 'times' steps...
        if t == times:
            break

        data, targets = batch

        # init new task
        model.generateParams(batch, bm.classes())
        # alpha = inner learning rate
        optimizer = optim.SGD(model.parameters(), lr = config.alpha)

        # repeat weight updates k times
        for _ in range(config.k):

            optimizer.zero_grad()

            out = model(data)

            loss = criterion(out, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        # get test accuracy of the k-shotted model
        test_acc.append(get_accuracy(model, bm, test_set = True))
        print(f'iteration {t+1}: test accuracy is {test_acc[-1]}', flush = True)

        # revert back to original model: remove test task and load old weights

        model.deactivate_linear_layer()
        model.load_state_dict(original_model_dict)

    print('Completed.')

    test_acc = torch.tensor(test_acc)
    return test_acc.mean().item(), test_acc.var().item()

def get_accuracy(model, batchmanager, test_set=False):
    """compute dev or test accuracy on a certain task
    Parameters:
    model (MultiTaskBERT): the model to train
    batchmanager (MultiTaskTrainLoader, BatchManager): the multi task batchmanager OR a single batchManagers
    Returns:
    float: said accuracy"""

    model.eval()
    count, num = 0., 0

    iter = batchmanager.test_iter if test_set else batchmanager.dev_iter

    with torch.no_grad():
        for batch in iter:  
            data, targets = batch
            out = model(data)
            predicted = out.argmax(dim=1)
            count += (predicted == targets).sum().item()
            num += len(targets)

    model.train()
    return count / num

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


def protomaml(config, sw, batch_managers, model_init, val_bms):

    CLASSIFIER_DIMS = 768
    
    beta = config.beta
    alpha = config.alpha
    
    optimizer = AdamW(model_init.parameters(), lr=beta)
    criterion = torch.nn.CrossEntropyLoss()    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = config.warmup, num_training_steps = config.nr_epochs * config.nr_episodes)

    NUM_WORKERS = 3 
    train_episodes = iter(EpisodeLoader.create_dataloader(
        config.samples_per_support, batch_managers, config.batch_size,
        num_workers = NUM_WORKERS
    ))

    global_step = 0

    def do_epoch(episode_loader, config, train=True):
        nonlocal global_step

        totals = {}
        avg_n = 0
        def log(loss, step, bm, avg, extra=""):
            nonlocal avg_n
            tbname = '{}/{}{}/loss'.format('train' if train else 'val', bm.name, extra)
            tag = tbname+'@{}'.format(step)
            sw.add_scalar(tag, loss, global_step)
            if hasattr(bm, 'parent'):
                log(loss, step, bm.parent, avg, extra="_stX")
            if avg:
                totals[step] = loss if step not in totals else totals[step] + loss
                avg_n += 1

        for i, batch in enumerate(it.islice(episode_loader, config.nr_episodes)):

            # external data structured used to accumulate gradients.
            accumulated_gradients = defaultdict(lambda : None)   
            
            for j, (support_iter, query_iter, bm) in enumerate(batch):

                print(f'batch {i}, task {j} : {bm.name}', flush = True)
                
                support_set = next(iter(support_iter))

                # [1] Calculate parameters for softmax.
                classes = bm.classes()
                model_init.deactivate_linear_layer()
                weights = deepcopy(model_init.state_dict())
                
                model_init.generateParams(support_set, classes)
                final = deepcopy(model_init.FFN)

                model_episode = type(model_init)(device=config.device)
                model_episode.load_state_dict(weights)
                model_episode.FFN = final.to(config.device)
                
                # [2] Adapt task-specific parameters
                task_optimizer = optim.SGD(model_episode.parameters(), lr=alpha)
                task_criterion = torch.nn.CrossEntropyLoss()
                for step, batch in enumerate([support_set] * config.k):
                    batch_inputs, batch_targets = batch

                    out = model_episode(batch_inputs)
                    loss = task_criterion(out, batch_targets)
                    
                    task_optimizer.zero_grad()
                    loss.backward()
                    task_optimizer.step()

                    if step == 0:
                        log(loss.item(), '1', bm, not train)
                    elif step == config.k-1:
                        log(loss.item(), 'k', bm, not train)

                    if train:
                        global_step += 1
   
                if not config.skip_prototypes:
                    print('backprop prototype trick')
                    # this will make gradients flow back to orignal model too.
                    model_episode.FFN.weight = nn.Parameter(model_init.prototypes + (model_episode.FFN.weight - model_init.prototypes).detach_())
                    model_episode.FFN.bias = nn.Parameter(model_init.prototype_norms + (model_episode.FFN.bias - model_init.prototype_norms).detach_())

                # [3] Evaluate adapted params on query set, calc grads.            
                for step, batch in enumerate(it.islice(query_iter, 1)):
                    batch_inputs, batch_targets = batch
                    out = model_episode(batch_inputs)

                    loss = task_criterion(out, batch_targets)

                    task_optimizer.zero_grad()
                    loss.backward()

                    log(loss.item(), 'q', bm, not train)

                    if train:
                        global_step += 1


                def accumulate_gradients(model_):
                    # accumulate the gradients
                    for n, p in model_.named_parameters():
                        if p.requires_grad and n not in ('FFN.weight','FFN.bias'):
                            if accumulated_gradients[n] is None:
                                accumulated_gradients[n] = p.grad.data
                                print("setting: {} = {}".format(n, p.grad.data.norm()))
                            else:
                                if p.grad == None:
                                    print("{} is None".format(n))
                                    continue
                                accumulated_gradients[n] += p.grad.data
                                print("adding: {} = {}".format(n, p.grad.data.norm()))

                #model_init.zero_grad()
                print("accumulating model_episode")
                accumulate_gradients(model_episode)
                print("accumulating model_init")
                accumulate_gradients(model_init)

                # end of inner loop

            model_init.deactivate_linear_layer()
            if train: # and thus not validation
                optimizer.zero_grad()
                # load the accumulated gradients and optimize
                for n, p in model_init.named_parameters():
                    if p.requires_grad:
                        p.grad.data = accumulated_gradients[n] 
                        print("setting back: {} = {}".format(n, accumulated_gradients[n]))
                optimizer.step()
                scheduler.step()
        
        for key, total in totals.items():
            log(total/avg_n, key+'/avg', bm, False)

        return { key : total/avg_n for key,total in totals.items() }
 
    val_episodes = iter(EpisodeLoader.create_dataloader(
        config.samples_per_support, val_bms, config.samples_per_support
    ))

    val_config = deepcopy(config)
    val_config.nr_episodes = 1

    filename = path_to_dicts(config)

    best_loss = sys.maxsize
    for epoch in range(config.nr_epochs):
        
        # train
        print('training...')
        do_epoch(train_episodes, config)

        # validate
        print('validating...')

        results = do_epoch(val_episodes, val_config, train=False)

        if results['q'] < best_loss:
            best_loss = results['q']
            torch.save(model_init.state_dict(), filename)
            print("New best loss found at {}, written model to {}".format(best_loss, filename))
            
            test_mean, test_std = k_shots(config, model_init, val_episodes, val_bms)
            sw.add_scalar('val/acc', test_mean, global_step)
            print(f'mean: {test_mean:.2f}, std: {test_std:.2f}')

    model.deactivate_linear_layer()    

    # K-SHOT VALIDATION!
    test_mean, test_std = k_shots(config, model, val_episodes, val_bms)
    print(f'mean: {test_mean:.2f}, std: {test_std:.2f}')

    return model.state_dict()
 
###########

def logloc(comment='',dir_name='runs'):
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(dir_name, current_time + '_' + socket.gethostname() + comment)
    return log_dir   
        
if __name__ == "__main__":
     # Parse training configuration
    parser = argparse.ArgumentParser()

    # Training params
    parser.add_argument('--nr_episodes', type=int, help='Number of episodes in an epoch', default = 25)
    parser.add_argument('--nr_epochs', type=int, help='Number of epochs', default = 80)
    parser.add_argument('--batch_size', type=int, default="64", help="How many tasks in an episode over which gradients for M_init are accumulated")
    parser.add_argument('--k', type=int, default="3", help="How many times do we update weights prime")
    parser.add_argument('--random_seed', type=int, default="42", help="Random seed")
    parser.add_argument('--resume', action='store_true', help='resume training instead of restarting')
    parser.add_argument('--beta', type=float, help='Beta learning rate', default = 1e-3)
    parser.add_argument('--alpha', type=float, help='Alpha learning rate', default = 5e-5)
    parser.add_argument('--warmup', type=float, help='For how many episodes we do warmup on meta-optimization.', default = 200)
    parser.add_argument('--samples_per_support', type=int, help='Number of samples to draw from the support set.', default = 32)
    parser.add_argument('--skip_prototypes', action='store_true')

    # Misc
    #parser.add_argument('--loss_print_rate', type=int, default='250', help='Print loss every')
    parser.add_argument('--model_save_dir', type=str, default=MODELS_PATH, help='The directory in which to store the models.')
    parser.add_argument('--sw_log_dir', type=str, default='runs', help='The directory in which to create the default logdir.')
    parser.add_argument('--device', type=str, help='')
    
    config = parser.parse_args()
    print(config) #print config, so we have 'paper trail' where we can make double sure what paramaters caused a result.

    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    numpy.random.seed(config.random_seed)

    if not config.device:
        config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("Running on: {}".format(config.device))
    print("Encoding: {} (should probably be UTF-8)".format(sys.stdout.encoding), flush=True)

    model = load_model(config)

    batchmanager1 = MultiNLIBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager2 = IBMBatchManager(batch_size = config.samples_per_support, device = config.device)
    batchmanager3 = MRPCBatchManager(batch_size = config.samples_per_support, device = config.device)        
    batchmanager4 = PDBBatchManager(batch_size = config.samples_per_support, device = config.device)        
    batchmanager5 = SICKBatchManager(batch_size = config.samples_per_support, device = config.device)

    pdb_subtasks = list(batchmanager4.get_subtasks(2))
    mnli_subtasks = list(batchmanager1.get_subtasks(2))

    # Double the weighting of tasks that aren't represented twice (normal, binary-sub-tasks).
    batchmanager3.weight_factor *= 2 # (only original)
    for bm in pdb_subtasks:
        bm.weight_factor *= 2        # (only subtasks)

    # MultiNLI, MRPC, PDB for training.
    train_bms = [ batchmanager1, batchmanager3 ]
    train_bms.extend(mnli_subtasks)
    #train_bms.extend(pdb_subtasks)

    # SICK for validation
    val_bms = [ batchmanager5 ] #[ batchmanager2 ]

    logdir = logloc(dir_name=config.sw_log_dir)
    sw = SummaryWriter(log_dir=logdir)

    # Train the model
    state_dict = protomaml(config, sw, train_bms, model, val_bms)
    
    #save model
    torch.save(state_dict, path_to_dicts(config))
