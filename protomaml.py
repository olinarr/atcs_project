import itertools

from utils.episodeLoader import EpisodeLoader


def protomaml(config, batch_managers, model):

    # TODO figure out best way to load BERT without having the
    # classification layer included inside.
    
    # initialization
    f_theta = model
    h_phi = MLP() 

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

    # outer loop
    for batch in episode_loader:

        total_loss = 0
        
        optimizer.zero_grad()
        
        # inner loop
        for task_iter in batch:
            # k     samples per task
            # t     length of sequence per sample
            # d     features per sequence-element (assuming same in and out)
            # ---
            # G     number of batches, and thus SGD steps.   


            # [1] Calculate parameters for softmax.
            
            
            # TODO prototypical calculation of W and b

            
            Wb = torch.stack(Wb)                    # N x l+1
            W, b = Wb[:,:-1], Wb[:,-1]              # N x l, N x 1
        
            
            # [2] Adapt task-specific parameters

            h_phi_prime = h_phi.copy()
            f_theta_prime = f_theta.copy() 
            # not this simple...
            # Make shallow copy of state_dict, and then replace
            # references of task-specific parameters with those to
            # deep clones? then use that state dict for new 
            # f_theta_prime instance of model?
            # Use Tensor.clone so we can backprop through it and 
            # update f_theta as well (if no first order approx).


            # TODO make sure that task-specific parameters in f_theta_prime
            # as well as parameters in h_phi_prime, W and b 
            # have requires_grad = True.
    
            params = [
                f_theta_prime.parameters(), 
                h_phi_prime.parameters(),
                [W, b]
            ]

            params = itertools.chain(*params)
            task_optimizer = optim.Adam(params, lr=alpha)
            task_criterion = torch.nn.CrossEntropyLoss()    
            
            def process_batch(batch):
                batch_output = f_theta_prime(batch.input)       # d x k
                batch_output = h_phi_prime(batch_output)        # l x k
                batch_output = W @ batch_output + b             # N x k
                batch_output = F.softmax(batch_output, dim=0)   # N x k
                loss = task_criterion(batch_output, batch.target)
                return loss

            # update task-specific parameters on support set (D_tr)
            for step, batch in enumerate(itertools.islice(task_iter, 1)):

                loss = process_batch(batch)
                
                task_optimizer.zero_grad() 
                loss.backward()
                task_optimizer.step()

                # TODO logging


            # [3] Evaluate adapted params on query set, calc grads.
            
            if first_order_approx:
                # TODO set parameters in f_theta_prime, h_phi_prime, 
                # to have requires_grad = True
            else:
                # TODO set parameters in f_theta, h_phi
                # to have requires_grad = True
                        

            # evaluate on query set (D_val) 
            for step, batch in enumerate(itertools.islice(task_iter, 1)):
                
                loss = process_batch(batch)
                loss.backward()


            if first_order_approx:
                with torch.no_grad():
                    pairs = [(f_theta_prime, f_theta), (h_phi_prime, h_phi)]
                    for prime, original in pairs:
                        params = zip(prime.named_parameters(), original.named_parameters())
                        for (pNamePr, pValueOr), (pNamePr, pValueOr) in params:
                            if pNamePr != pNameOr:
                                print(pNamePr)
                                print(pNameOr)
                                raise Error("Order in which named parameters are returned is probably not deterministic?")
                            pValueOr.grad += pValuePr.grad.clone()

            # end of inner loop

        #TODO divide gradients by number of tasks?
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