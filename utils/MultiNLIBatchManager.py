import torchtext, torch
import pandas as pd
import os

class MyIterator():
    """Used in the batch manager. An endless generator with length"""
    def __init__(self, dataset, batch_size, l2i, device, IBM = False):
        """Init of the model

        Parameters:
        dataset: torchtext dataset
        batch_size (int): batch size
        l2i (dict): mapping from labels to indexes
        device (str): cpu or cuda:0

        """
        self.IBM = IBM
        self.batch_size = batch_size
        self.dataset = dataset
        # compute the number of batches.
        self.length = len(dataset) // batch_size + (0 if len(dataset) % batch_size == 0 else 1)
        self.l2i = l2i

        self.idx = 0

        self.device = device

    def __next__(self):
        # if we're still within the dataset....
        if self.idx < len(self.dataset):
            # select next batch (list of torchtext Example instances)
            # the "min" is used in order not to exceed the length
            if self.IBM: # Check whether we're using the IBM
                batch = self.dataset.iloc[self.idx:min(self.idx+self.batch_size, len(self.dataset)),:]
            else:
                batch = self.dataset[self.idx:min(self.idx+self.batch_size, len(self.dataset))]

            # update index
            self.idx += self.batch_size

            # create batch: it is a tuple of a list of (premise, hypotesis) and a tensor of label_indexes
            if self.IBM:
                return [(batch.loc[i,'topicText'], batch.loc[i,'claims.claimCorrectedText']) for i in batch.index],\
                    torch.tensor([self.l2i[batch.loc[i,'claims.stance']] for i in batch.index], device = self.device, requires_grad = False)
            else:
                return [(example.premise, example.hypothesis) for example in batch],\
                    torch.tensor([self.l2i[example.label] for example in batch], device = self.device, requires_grad = False)

        # else, we are finished, but restart (endless iterator)
        else:
            self.idx = 0
            raise StopIteration

    def __len__(self):
        return self.length

    def __iter__(self):
        return self


class MultiNLIBatchManager():
    def __init__(self, batch_size = 256, device = 'cpu'):
        # sequential false -> no tokenization. Why? Because right now this
        # happens instead of BERT
        TEXT = torchtext.data.Field(use_vocab = False, sequential = False)
        LABEL = torchtext.data.Field(sequential = False)

        self.train_set, self.dev_set, self.test_set = torchtext.datasets.MultiNLI.splits(TEXT, LABEL)
        self.batch_size = batch_size

        # mapping from classes to integers
        self.l2i = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        # create the three iterators
        self.train_iter = MyIterator(self.train_set, batch_size, self.l2i, device)
        self.dev_iter = MyIterator(self.dev_set, batch_size, self.l2i, device)
        self.test_iter = MyIterator(self.test_set, batch_size, self.l2i, device)

        self.device = device

class IBMBatchManager():
    """
    Batch Manager for the IBM dataset
    """
    def __init__(self, batch_size = 256, device = 'cpu'):
        """
        Initializes the dataset

        Args:
            batch_size: Number of elements per batch
            device    : Device to run it on: cpu or gpu
        """
        
        # Get a mapping from stances to labels
        self.l2i = {'PRO': 0, 'CON':1}

        # IBM dataset doesn't offer a separate validation set!
        df = pd.read_csv(os.path.join("ibm", "claim_stance_dataset_v1.csv"))
        self.train_set = df.query("split == 'train'")[['topicText', 'claims.claimCorrectedText', 'claims.stance']]
        self.dev_set   = df.query("split == 'test'")[['topicText', 'claims.claimCorrectedText', 'claims.stance']]
        self.test_set  = df.query("split == 'test'")[['topicText', 'claims.claimCorrectedText', 'claims.stance']]

        # Turn the datasets into iterators
        self.train_iter = MyIterator(self.train_set, batch_size, l2i = self.l2i, device = device, IBM = True)
        self.dev_iter = MyIterator(self.dev_set, batch_size, l2i = self.l2i, device = device, IBM = True)
        self.test_iter = MyIterator(self.test_set, batch_size, l2i = self.l2i, device = device, IBM = True)

if __name__ == "__main__":
    batchmanager = IBMBatchManager()
    #batchmanager = MultiNLIBatchManager()

