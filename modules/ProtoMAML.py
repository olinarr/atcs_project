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

    def duplicate(self, first_order = True):
        """ Duplicate this model. 

        Parameters:
        first_order (bool): whether to use fo approximation of MAML. If False, then the gradients should flow through it.

        Returns:
        (ProtoMAML): a copy of this model"""

        if first_order:
            # TODO figure out why deepcopy won't work
            with torch.no_grad():
                new_model = ProtoMAML(device = self.device, trainable_layers = self.trainable_layers)
                new_model.FFN = None if self.FFN is None else nn.Linear(768, self.FFN.out_features)
                new_model.load_state_dict(self.state_dict())
                return new_model 
        else:
            raise NotImplementedError("Second order not implemtented.")

    def accumulateGradients(self, model_prime, first_order = True):
        """Accumulate (add) the gradients from another model.

        Parameters:
        model_prime (ProtoMAML): the model from which the gradients must be 'Inherited'
        first_order (bool): whether to use fo approximation of MAML"""

        # if we use first-order approximation, copy gradients to originals manually.
        # TODO test if this actually works.
        if first_order:
            with torch.no_grad():
                params = zip(model_prime.named_parameters(), self.named_parameters())
                for (pNamePr, pValuePr), (pNameOr, pValueOr) in params:
                    assert pNamePr == pNameOr, \
                        "Order in which named parameters are returned is probably not deterministic? \n names: {}, {}".format(pNamePr, pNameOr)

                    if pValuePr.requires_grad:
                        assert pValueOr.requires_grad, \
                            "A parameter which did not need the gradients, now needs it!\nparameter {}:".format(pNameOr)

                        # sum to the original if it was already something, otherwise init with it
                        pValueOr.grad = pValuePr.grad if pValueOr.grad is None else pValueOr.grad + pValuePr.grad
        else:
            raise NotImplementedError("Second order not implemtented.")

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