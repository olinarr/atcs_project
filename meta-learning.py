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


class EpisodeLoader()
# TODO make dataloader that combines the iterators for each task
# and bundles batches for those tasks into batches of episodes for meta-learning


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

    # outer loop
    while not done:

        # sample batch of tasks
        tasks = MagicTaskSampler.sample()
        total_loss = 0

        # inner loop
        for support_iter, query_iter in tasks:
            # k     samples per task
            # t     length of sequence per sample
            # d     features per sequence-element (assuming same in and out)
            # ---
            # G     number of batches, and thus SGD steps.   


            # Calculate initial parameters for softmax.
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
        
            
            # Update parameters

            h_phi_prime = h_phi.copy()
            f_theta_prime = f_theta.copy() 
            # not this simple...
            # Make shallow copy of state_dict, and then replace
            # references of task-specific parameters with those to
            # deep clones? then use that state dict for new 
            # f_theta_prime instance of model?
            # Use Tensor.clone so we can backprop through it and 
            # update f_theta as well.


            # TODO make sure that task-specific parameters in f_theta_prime
            # as well as parameters in h_phi_prime, W and b and no others 
            # have requires_grad = True.

            params = itertools.chain(
                    f_theta_prime.parameters(), 
                    h_phi_prime.parameters(),
                    [W], [b]
                )
            task_optimizer = optim.Adam(params, lr=alpha)
            task_criterion = torch.nn.CrossEntropyLoss()    
            
            def process_batch(batch_input):
                batch_output = f_theta_prime(batch_input)       # d x k
                batch_output = h_phi_prime(batch_output)              # l x k
                batch_output = W @ batch_output + b             # N x k
                batch_output = F.softmax(batch_output, dim=0)   # N x k
                loss = task_criterion(batch_output, batch_target)
                return loss

            # update task-specific parameters 
            for step, batch in enumerate(support_iter):
                if step >= G - 1:
                    break

                batch_input = batch.input                       # d x t x k
                batch_target = batch.target
               
                loss = proces_batch(batch_input)
                
                task_optimizer.zero_grad() 
                loss.backward()
                task_optimizer.step()

                # TODO logging


            # TODO now set all parameters in f_theta, g_psi, h_phi 
            # to have requires_grad = True
            

            # evaluate on query set (D_val) 
            for step, batch in enumerate(query_iter):
                if step >= ??: #TODO name number, or is it always 1?
                    break

                batch_input = batch.input                       # d x t x k
                batch_target = batch.target

                loss = proces_batch(batch_input)
                loss.backward()

                




            

            

            
