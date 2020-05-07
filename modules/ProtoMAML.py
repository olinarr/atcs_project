from transformers import BertModel, BertTokenizerFast
import torch
import torch.nn as nn

class ProtoMAML(nn.Module):

    def __init__(self, device = 'cpu', trainable_layers = [9, 10,11]):
        """Init of the model

        Parameters:
        device (str): CUDA or cpu
        trainable_layers (list(int)): BERT layers to be trained
        """
        super(ProtoMAML, self).__init__()

        self.device = device
        self.trainable_layers = trainable_layers
        
        # load pre-trained BERT: tokenizer and the model.
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.BERT = BertModel.from_pretrained('bert-base-uncased').to(device)

        # until we initialize it, it will be None.
        self.FFN = None

        # deactivate gradients on the parameters we do not need.

        # generate the name of the layers
        trainable_param_names = ["encoder.layer."+str(ll) for ll in trainable_layers]
        # for all the parameters...
        for name, params in self.BERT.named_parameters():
            flag = False
            # look in all layers: if this parameter belongs to one of this layers, we
            # set the flag
            for trainable_param in trainable_param_names:
                if trainable_param in name:
                    flag = True
                    break

            # if the flag was not set, then it is not in the layers to train:
            # deactivate gradients.
            if not flag:
                params.requires_grad = False

    def _applyBERT(self, inputs):
        """Forward function of BERT only

        Parameters:
        inputs (list((str, str))): List of tuples of strings: pairs of premises and hypothesis

        Returns:
        torch.Tensor: a BATCH x HIDDEN_DIM tensor

        """

        # tokenizes and encodes the input string to a pytorch tensor.
        # it also adds [CLS], [SEQ], takes care of masked attention and all that racket.
        # we pad them and output is a pytorch.
        # output is a dict with ready to be passed to BERT directly
        encoded = self.tokenizer.batch_encode_plus(inputs, pad_to_max_length=True, return_tensors="pt")
        # to cuda
        encoded['input_ids'] = encoded['input_ids'].to(self.device)
        encoded['token_type_ids'] = encoded['token_type_ids'].to(self.device)
        encoded['attention_mask'] = encoded['attention_mask'].to(self.device)

        # output of bert
        # the zero-th is the hidden states, which we need
        out = self.BERT(**encoded)[0]

        # out is BATCH x SENT_LENGTH x EMBEDDING LENGTH
        # we get the first token (the [CLS] token)
        return out[:, 0, :]

    def generateParams(self, support_set):
        """ Generate linear classifier form support set.

        support (list((str, str)), torch.Tensor): support set to generate the parameters W and B from."""
        with torch.no_grad():
            
            batch_input = support_set[0]                    # k-length list of (str, str)
            batch_target = support_set[1]                   # k

            # encode sequences 
            batch_input = self._applyBERT(batch_input)      # k x d
            
            classes = batch_target.unique()                 # N
            W = []
            b = []
            for cls in classes:
                cls_idx   = (batch_target == cls).nonzero()
                cls_input = torch.index_select(batch_input, dim=0, index=cls_idx.flatten())
                                                    # C x d
                                
                # prototype is mean of support samples (also c_k)
                prototype = cls_input.mean(dim=0)         # d
                
                # see proto-maml paper, this corresponds to euclidean distance
                W.append(2 * prototype)
                b.append(- prototype.norm() ** 2)
            
            # the transposes are because the dimensions were wrong!
            W = torch.stack(W)                          # C x d
            b = torch.stack(b)

            linear = nn.Linear(768, W.shape[0])
            linear.weight = nn.Parameter(W)
            linear.bias = nn.Parameter(b)

            # two layers for more flexbility.
            # TODO Decide on whether it makese sense to use it
            # self.FFN = nn.Sequential(nn.Linear(768, 768).to(device), nn.ReLU(), linear)
            self.FFN = linear

    def revert_state(self, state_dict):
        """ Revert to the original model. Of course, we deactivate the generated layer.

        Parameters:
        state_dict: the original weights"""

        self.FFN = None
        self.load_state_dict(state_dict)

    def forward(self, inputs):
        """Forward function of the model

        Parameters:
        inputs (list((str, str))): List of tuples of strings: pairs of premises and hypothesis

        Returns:
        torch.Tensor: a BATCH x N_CLASSES tensor

        """

        if self.FFN is None:
            raise Exception('You have called the forward function without having initialized the parameters! Call generateParams() on the support first.')
        else:
            output = self._applyBERT(inputs)
            output = self.FFN(output)
            return output