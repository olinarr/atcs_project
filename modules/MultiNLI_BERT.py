from transformers import BertModel, BertTokenizerFast
import torch.nn as nn

class MultiNLI_BERT(nn.Module):

    def __init__(self, n_classes = 3, device = 'cpu', trainable_layers = [10,11]):
        """Init of the model

        Parameters:
        n_classes (int): Number of output classes
        device (str): CUDA or cpu
        trainable_layers (list(int)): BERT layers to be trained

        """
        super(MultiNLI_BERT, self).__init__()
        
        # load pre-trained BERT: tokenizer and the model.
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.BERT = BertModel.from_pretrained('bert-base-uncased').to(device)

        # FFN: classifier to fine-tune.
        # 768 is the dimension of BERT hidden layers!
        self.classifier = nn.Linear(768, n_classes).to(device)
        self.device = device


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

    def forward(self, inputs):
        """Forward function of the model

        Parameters:
        inputs (list((str, str))): List of tuples of strings: pairs of premises and hypothesis

        Returns:
        torch.Tensor: a BATCH x N_CLASSES tensor

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
        # we get the first token (the [CLS] token) and classify on it.
        return self.classifier(out[:, 0, :])

    def trainable_parameters(self):
        """Returns the non-frozen parameters

        Returns:
        Generator object of parameters s.t. require_grads = True

        """
        return filter(lambda p: p.requires_grad, self.parameters())