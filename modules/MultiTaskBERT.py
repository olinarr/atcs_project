from transformers import BertModel, BertTokenizerFast
import torch.nn as nn

class MultiTaskBERT(nn.Module):

    def __init__(self, device = 'cpu', trainable_layers = [9, 10, 11], tasks = None):
        """Init of the model

        Parameters:
        device (str): CUDA or cpu
        trainable_layers (list(int)): BERT layers to be trained
        tasks (dict(str, int)): dictionary mapping name of the tasks to number of labels

        """
        super(MultiTaskBERT, self).__init__()

        self.device = device
        
        # load pre-trained BERT: tokenizer and the model.
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.BERT = BertModel.from_pretrained('bert-base-uncased').to(device)

        # shared linear layer
        self.sharedLinear = nn.Sequential(nn.Linear(768, 768), nn.ReLU()).to(device)

        # all the tasks
        self.tasks = (name for name, labels in tasks)

        self.taskSpecificLayer = nn.ModuleDict(
                {task: nn.Linear(768, n_labels) for task, n_labels in tasks.items()}
            ).to(self.device)

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

    def addTask(self, task, n_classes):
        self.taskSpecificLayer[task] = nn.Linear(768, n_classes)

    def forward(self, inputs, task):
        """Forward function of the model

        Parameters:
        inputs (list((str, str))): List of tuples of strings: pairs of premises and hypothesis
        task (str): Name of the task we are using

        Returns:
        torch.Tensor: a BATCH x N_CLASSES tensor

        """

        # tokenizes and encodes the input string to a pytorch tensor.
        # it also adds [CLS], [SEQ], takes care of masked attention and all that racket.
        # we pad them and output is a pytorch.
        # output is a dict with ready to be passed to BERT directly
        encoded = self.tokenizer.batch_encode_plus(inputs, pad_to_max_length=True, return_tensors="pt", max_length = self.BERT.config.max_position_embeddings)
        # to cuda
        encoded['input_ids'] = encoded['input_ids'].to(self.device)
        encoded['token_type_ids'] = encoded['token_type_ids'].to(self.device)
        encoded['attention_mask'] = encoded['attention_mask'].to(self.device)

        # output of bert
        # the zero-th is the hidden states, which we need
        out = self.BERT(**encoded)[0]

        # out is BATCH x SENT_LENGTH x EMBEDDING LENGTH
        # we get the first token (the [CLS] token) and classify on it.
        out = self.sharedLinear(out[:, 0, :])

        # return the task specific output

        return self.taskSpecificLayer[task](out)

    def globalParameters(self):
        """ Returns the global parameters """
        return (p for n, p in self.named_parameters() if 'taskSpecificLayer' not in n)

    def named_globalParameters(self):
        """ Returns the global parameters """
        return ((n, p) for n, p in self.named_parameters() if 'taskSpecificLayer' not in n)

    def taskParameters(self, task):
        """ Returns the task specific parameters """
        return (p for n, p in self.named_parameters() if f'taskSpecificLayer.{task}' in n)

    def named_taskParameters(self, task):
        """ Returns the task specific parameters """
        return ((n, p) for n, p in self.named_parameters() if f'taskSpecificLayer.{task}' in n)
