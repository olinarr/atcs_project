"""

This is just to mess around and get a feel for how LEOPARD might look in code.
Partially pseudocode, partially python.

Variable names chosen to make sense in context of paper on LEOPARD, well hopefully ;-) .

"""


class LeopardEncoder(nn.Module):

    def __init__(self, cls_dim, l):
        super(LeopardEncoder, self).__init__()

        #TODO ask if this hidden dim size is okay.
        self.fc1 = nn.Linear(cls_dim, cls_dim)
        self.fc2 = nn.Linear(cls_dim, l + 1)

    def forward(self, x):
        x = self.fc1(x).tanh()
        x = self.fc2(x).tanh()
        return x


class EpisodeLoader(data.IterableDataset)
"""
Used to obtain episodes for meta-learning, i.e. batches of 'tasks' (support and query sets).
Essentially a dataset of dataloaders.
"""

    def __init__(self, k, train_sets, valid_sets):
        """
        Params:
          k: the amount of samples included in every support set.
          datasets: the 
        """
        super(EpisodeLoader).__init__()

        self.k = k
        self.train_sets = train_sets
        self.valid_sets = valid_sets


    def __iter__(self, idx):
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            nr_workers = 1
            worker_id = 0
        else:
            nr_workers = worker_info.num_workers
            worker_id = worker_info.id

        for dataset in self.datasets:
            setsize = len(dataset)
            per_worker = int(math.ceil(setsize/float(nr_workers)))

            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, setsize-1)



def leopard():

    # initialization
    f_theta = BertEncoder()
    g_psi = LeopardEncoder()
    h_phi = MLP() 

    params = itertools.chain(
        f_theta.parameters(),
        g_psi.parameters(),
        h_phi.parameters()
    )

    optimizer = optim.Adam(params, lr=beta)
    criterion = torch.nn.CrossEntropyLoss()    

    episode_loader = EpisodeLoader()

    # outer loop
    while not done:

        optimizer.zero_grad()

        # sample batch of tasks
        tasks = next(iter(episode_loader))
        total_loss = 0

        # inner loop
        for support_iter, query_iter in tasks:
            # k     samples per task
            # t     length of sequence per sample
            # d     features per sequence-element (assuming same in and out)
            # ---
            # G     number of batches, and thus SGD steps.   


            # [1] Calculate initial parameters for softmax.
            # Get batch specially for generation of softmax params.
            batch = next(support_iter)                 
            batch_input = batch.input               # d x t x k
            batch_target = batch.target             # k

            classes = batch_target.unique()         # N
            Wb = []
            for cls in classes:
                cls_idx   = (batch_target == cls).non_zero()
                cls_input = torch.index_select(batch_input, dim=2, cls_idx)
                                                    # d x t x C
                # encode sequences 
                cls, _, _ = f_theta(cls_input)      # d x C

                # apply generator
                encoded = g_psi(cls)                # l+1 x C

                encoded = encoded.mean(dim=1)       # l+1
                Wb.append(encoded)
                
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
            for step, batch in enumerate(support_iter):
                if step >= G - 1:
                    break

                loss = process_batch(batch)
                
                task_optimizer.zero_grad() 
                loss.backward()
                task_optimizer.step()

                # TODO logging


            # [3] Evaluate adapted params on query set, calc grads.
            
            if first_order_approx:
                # TODO set parameters in f_theta_prime, h_phi_prime, 
                # g_psi to have requires_grad = True
            else:
                # TODO set parameters in f_theta, h_phi, g_psi
                # to have requires_grad = True
                           

            # evaluate on query set (D_val) 
            for step, batch in enumerate(query_iter):
                if step >= ??: #TODO name number, or is it always 1?
                    break

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




            

            

            
