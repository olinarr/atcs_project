"""

This is just to mess around and get a feel for how LEOPARD might look in code.
Almost more pseudocode than real python.

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

    optimizer = optim.Adam(task_model.parameters(), lr=alpha)
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
            batch = next(suppert_iter)                 
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
                encoded = g_psi(encoded)            # l+1 x C

                encoded = encoded.mean(dim=1)       # l+1
                Wb.append(encoded)
                
            Wb = torch.stack(Wb)                    # N x l+1
            W, b = Wb[:,:-1], Wb[:,-1]              # N x l, N x 1
        
            
            # Update parameters

            # TODO copy model and update parameters on copy
            # use that copy to calculate the loss for this task
            # in the meta step.
            f_theta_prime = f_theta.copy() # not this simple (?/!)
            # can probably save/load to and from a buffer instead

            task_optimizer = optim.Adam(task_model.parameters(), lr=alpha)
            task_criterion = torch.nn.CrossEntropyLoss()    

            for step, batch in enumerate(support_iter):
                if step >= G - 1:
                    break

                batch_input = batch.input                       # d x t x k
                batch_target = batch.target

                batch_output = f_theta_prime(batch_input)       # d x k
                batch_output = h_phi(batch_output)              # l x k
                batch_output = W @ batch_output + b             # N x k
                batch_output = F.softmax(batch_output, dim=0)   # N x k

                loss = task_criterion(batch_output, batch_target)

                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()

                # logging


            # evaluate 
            for step, batch in enumerate(query_iter):
                if step >= ??: #TODO what should this number be called? or is it always 1?
                    break

                batch_input = batch.input                       # d x t x k
                batch_target = batch.target

                batch_output = f_theta_prime(batch_input)       # d x k
                batch_output = h_phi(batch_output)              # l x k
                batch_output = W @ batch_output + b             # N x k
                batch_output = F.softmax(batch_output, dim=0)   # N x k

                loss = criterion(batch_output, batch_target)

                #TODO decide if we literally sum losses like this
                # or if we backward all of them
                total_loss += loss  


        # TODO determine how we transfer gradients from theta_prime to theta
        # this might be a start? (source: https://discuss.pytorch.org/t/solved-copy-gradient-values/21731)
#        for paramName, paramValue, in net.named_parameters():
#            for netCopyName, netCopyValue, in netCopy.named_parameters():
#                if paramName == netCopyName:
#                    netCopyValue.grad = paramValue.grad.clone()

        
            
                




            

            

            
