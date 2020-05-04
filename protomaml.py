import torch
import torch.nn as nn

import itertools

from utils.episodeLoader import EpisodeLoader


def protomaml(config, batch_managers, model):

    CLASSIFIER_DIMS = 768
    
    # TODO figure out proper strategy for saving/loading these 
    # meta-learning models.
    
    # initialization
    f_theta = model.BERT
    h_phi = nn.Linear(CLASSIFIER_DIMS, CLASSIFIER_DIMS)

    params = itertools.chain(
        f_theta.parameters(),
        h_phi.parameters()
    )

    # TODO learnable alpha, beta learning rates?
    
    optimizer = optim.Adam(params, lr=beta)
    criterion = torch.nn.CrossEntropyLoss()    

    # for protomaml we use two samples (support and query)
    SAMPLES_PER_EPISODE = 2
    
    NUM_WORKERS = 3 
    episode_loader = EpisodeLoader.create_dataloader(
        config.k, batch_managers, config.batch_size,
        samples_per_episode = SAMPLES_PER_EPISODE, 
        num_workers = NUM_WORKERS
    )

    for batch in episode_loader:
        # a batch of episodes
        
        optimizer.zero_grad()
        
        for task_iter in batch:
            # k     samples per task
            # t     length of sequence per sample
            # d     features per sequence-element (assuming same in and out)

            support_set = next(iter(task_iter))

            # [1] Calculate parameters for softmax.
            
            batch_input = support_set[0]            # d x t x k
            batch_target = support_set[1]           # k

            classes = batch_target.unique()         # N
            W = []
            b = []
            for cls in classes:
                cls_idx   = (batch_target == cls).non_zero()
                cls_input = torch.index_select(batch_input, dim=2, cls_idx)
                                                    # d x t x C
                # encode sequences 
                # TODO this won't work we should probably adapt
                # FineTunedBERT so it can forward without immediatly 
                # applying the classifier.
                cls, _, _ = f_theta(cls_input)      # d x C
                
                # prototype is mean of support samples (also c_k)
                prototype = cls.mean(dim=1)         # d
                
                # see proto-maml paper, this corresponds to euclidean distance
                W.append(2 * prototype)
                b.append(- prototype.norm() ** 2)
            
            W = torch.stack(W)                      
            b = torch.stack(b)
            # h_phi, W, b together now make up the classifier.
        
            
            # [2] Adapt task-specific parameters

            h_phi_prime = h_phi.copy()
            f_theta_prime = f_theta.copy() 
            # not this simple...
            # TODO Make shallow copy of state_dict, and then replace
            # references of task-specific parameters with those to
            # clones? then use that state dict for a new 
            # f_theta_prime instance of model?
            # Use Tensor.clone so we can backprop through it and 
            # update f_theta as well (if no first order approx).


            # TODO have task-specific parameters in f_theta_prime
            # as well as parameters in h_phi_prime, W and b 
            # set with requires_grad = True.
    
            params = [
                f_theta_prime.parameters(), 
                h_phi_prime.parameters(),
                [W, b]
            ]
            params = itertools.chain(*params)
            task_optimizer = optim.Adam(params, lr=alpha)
            task_criterion = torch.nn.CrossEntropyLoss()    
            
            def process_batch(batch):
                batch_input, batch_target = batch
                batch_output = f_theta_prime(batch_input)       # d x k
                batch_output = h_phi_prime(batch_output)        # l x k
                batch_output = W @ batch_output + b             # N x k
                batch_output = F.softmax(batch_output, dim=0)   # N x k
                loss = task_criterion(batch_output, batch_target)
                return loss

            for step, batch in enumerate([support_set]):

                loss = process_batch(batch)
                
                task_optimizer.zero_grad() 
                loss.backward()
                task_optimizer.step()

                # TODO logging


            # [3] Evaluate adapted params on query set, calc grads.
            
            if first_order_approx:
                # We are using the approximation so we only backpropagate to the
                # 'prime' models and will then copy gradients to originals later.
                
                # TODO set parameters in f_theta_prime, h_phi_prime, 
                # to have requires_grad = True
            else:
                # Backpropagate all the way back to originals (through optimization
                # on the support set and thus requiring second order gradients.)

                # TODO set parameters in f_theta, h_phi (originals)
                # to have requires_grad = True
                
                
            # evaluate on query set (D_val) 
            for step, batch in enumerate(itertools.islice(task_iter, 1)):
                loss = process_batch(batch)
                loss.backward()


            # if we use first-order approximation, copy gradients to originals manually.
            # TODO test if this actually works.
            if first_order_approx:
                with torch.no_grad():
                    pairs = [(f_theta_prime, f_theta), (h_phi_prime, h_phi)]
                    for prime, original in pairs:
                        params = zip(prime.named_parameters(), original.named_parameters())
                        for (pNamePr, pValueOr), (pNamePr, pValueOr) in params:
                            if pNamePr != pNameOr:
                                raise Error("Order in which named parameters are returned is probably not deterministic? \n names: {}, {}"
                                           .format(pNamePr, pNameOr))
                            pValueOr.grad += pValuePr.grad.clone()

            # end of inner loop

        # we have now accumulated gradients in the originals
        #TODO divide gradients by number of tasks? or just have learning rate be lower?
        optimizer.step()

    
    
        
if __name__ == "__main__":
     # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--batch_size', type=int, default="32", help="Batch size")
    parser.add_argument('--random_seed', type=int, default="42", help="Random seed")
    parser.add_argument('--resume', action='store_true', help='resume training instead of restarting')
    #parser.add_argument('--lr', type=float, help='Learning rate', default = 2e-5)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default = 25)
    #parser.add_argument('--loss_print_rate', type=int, default='250', help='Print loss every')
    config = parser.parse_args()

    torch.manual_seed(config.random_seed)
    config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = load_model(config)

    batchmanager1 = MultiNLIBatchManager(batch_size = config.batch_size, device = config.device)
    batchmanager2 = IBMBatchManager(batch_size = config.batch_size, device = config.device)
    batchmanager3 = MRPCBatchManager(batch_size = config.batch_size, device = config.device)        

    batchmanagers = [batchmanager1, batchmanager2, batchmanager3]

    # Train the model
    print('Beginning the training...', flush = True)
    state_dict, dev_acc = protomaml(config, batchmanagers, model)
    
    print(f"#*#*#*#*#*#*#*#*#*#*#\nFINAL BEST DEV ACC: {dev_acc}\n#*#*#*#*#*#*#*#*#*#*#", flush = True)

    #save model
    torch.save(state_dict, path_to_dicts(config))