"""

This is just to mess around and get a feel for how LEOPARD might look in code.
Almost more pseudocode than real python.

Variable names chosen to make sense in context of paper on LEOPARD, well hopefully ;-) .

"""


class LeopardEncoder(nn.Module):

    def __init__(self, cls_dim, l):
        super(LeopardEncoder, self).__init__()

        self.fc1 = nn.Linear(cls_dim, cls_dim)
        self.fc2 = nn.Linear(cls_dim, l + 1)


    def forward(self, x):
        
        x = self.fc1(x).tanh()
        x = self.fc2(x).tanh()
        return x

def leopard():

    # This is the training data per task for an episode.
    iter_by_task = {
            "MultiNLI" : None,
            "IBMStance" : None,
            "MSParaphrase" : None,
            "Discourse" : None,
        }

    
    f_theta = BertEncoder()
    g_psi = LeopardEncoder()
    h_phi = MLP() 

    for task, data_iter in data_by_task.items():

        batch = next(data_iter)                 # d x t x B
        batch_input = batch.input           
        batch_target = batch.target

        classes = batch_target.unique()         # N
        W_ = []
        b_ = []
        for cls in classes:

            # Bool index for current class
            cls_idx = batch_target == cls
            
            # Input data for current class
            # TODO this doesn't work directly cls_idx needs to be same shape...
            cls_input = batch_input[cls_idx]    # d x t x C

            # create
            cls, _, _ = f_theta(cls_input)      # d x C
            encoded = g_psi(encoded)            # l+1 x C
            encoded = encoded.mean(dim=1)       # l+1
            w, b = encoded[:-1], encoded[-1]    # l, 1
            W_.append(w)
            b_.append(w)
            
        # TODO should probably not split into w,b until after stacking?
        W = torch.stack(W_)                     # N x l
        b = torch.stack(b_)                     # N
        
        batch_pred = F.softmax(W @ h_phi(f_theta(
        
