import torchtext, torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class MyIterator():
    """Used in the batch manager. An endless generator with length"""
    def __init__(self, dataset, batch_size, l2i, device, name = 'IBM'):
        """Init of the model

        Parameters:
        dataset: torchtext dataset
        batch_size (int): batch size
        l2i (dict): mapping from labels to indexes
        device (str): cpu or cuda:0
        name (str): which dataset?

        """
        self.name = name
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

            if self.name == 'IBM':
                batch = self.dataset.iloc[self.idx:min(self.idx+self.batch_size, len(self.dataset)),:]
            elif self.name in ('NLI', 'MRPC'):
                batch = self.dataset[self.idx:min(self.idx+self.batch_size, len(self.dataset))]
            else:
                raise NotImplementedError

            # update index
            self.idx += self.batch_size

            # create batch: it is a tuple of a list of (premise, hypotesis) and a tensor of label_indexes
            if self.name == 'IBM':
                return [(batch.loc[i,'topicText'], batch.loc[i,'claims.claimCorrectedText']) for i in batch.index],\
                    torch.tensor([self.l2i[batch.loc[i,'claims.stance']] for i in batch.index], device = self.device, requires_grad = False)
            elif self.name == 'NLI':
                return [(example.premise, example.hypothesis) for example in batch],\
                    torch.tensor([self.l2i[example.label] for example in batch], device = self.device, requires_grad = False)
            elif self.name == 'MRPC':
                # MRPC is of the form: label (int already, no mapping needed), sent_1, sent_2
                return [(example[1], example[2]) for example in batch],\
                    torch.tensor([example[0] for example in batch], device = self.device, requires_grad = False)
            else:
                raise NotImplementedError

        # else, we are finished, but restart (endless iterator)
        else:
            self.idx = 0
            raise StopIteration

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

class BatchManager():
    """
    Class to define a batch manager.
    For new implementations:
     *  create a subclass;
     *  in the constructer create a self.train_set, self.dev_set and self.test_set;
     all of which should be subclasses of the torch.utils.data.Dataset class;
     *  call the initialize function at the end of your constructor;
     *  override the collate function which should take a list of dataset elements 
     and combine them into a batch;
    """
    
    SHUFFLE = False
    
    def collate(self):
        pass
    
    def initialize(self, batch_size):
        # create the three iterators
        self.train_iter = DataLoader(self.train_set, batch_size=batch_size, shuffle=self.SHUFFLE, collate_fn=self.collate)
        self.dev_iter   = DataLoader(self.dev_set,   batch_size=batch_size, shuffle=self.SHUFFLE, collate_fn=self.collate)
        self.test_iter  = DataLoader(self.test_set,  batch_size=batch_size, shuffle=self.SHUFFLE, collate_fn=self.collate)

        
class MultiNLIBatchManager(BatchManager):
    
    SHUFFLE = True
    
    def collate(self, samples):
        return [(example.premise, example.hypothesis) for example in samples],\
                torch.tensor([self.l2i[example.label] for example in samples], device = self.device, requires_grad = False)
    
    def __init__(self, batch_size = 32, device = 'cpu'):
        # sequential false -> no tokenization. Why? Because right now this
        # happens instead of BERT
        TEXT = torchtext.data.Field(use_vocab = False, sequential = False)
        LABEL = torchtext.data.Field(sequential = False)

        self.train_set, self.dev_set, self.test_set = torchtext.datasets.MultiNLI.splits(TEXT, LABEL)
        self.batch_size = batch_size

        # mapping from classes to integers
        self.l2i = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        
        self.initialize(batch_size)
        self.device = device

        
class DataframeDataset(Dataset):
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def __len__(self):
        return self.dataframe.shape[0]
        
    def __getitem__(self, idx):
        return self.dataframe.iloc[idx, :]
        
        
class IBMBatchManager(BatchManager):
    """
    Batch Manager for the IBM dataset
    """
    
    def collate(self, samples):
        batch = [(sample['topicText'], sample['claims.claimCorrectedText']) for sample in samples]
        labels = [sample['claims.stance'] for sample in samples]
        labels = [self.l2i[label] for label in labels]
        return batch, torch.tensor(labels, device = self.device, requires_grad = False)
    
    def __init__(self, batch_size = 32, device = 'cpu'):
        """
        Initializes the dataset

        Args:
            batch_size: Number of elements per batch
            device    : Device to run it on: cpu or gpu
        """
        
        # Get a mapping from stances to labels
        self.l2i = {'PRO': 0, 'CON':1}

        # IBM dataset doesn't offer a separate validation set!
        #TODO: maybe make new train/valid/test split so we have validation data?
        
        df = pd.read_csv(os.path.join(".data/ibm", "claim_stance_dataset_v1.csv"))
        self.train_set = DataframeDataset(df.query("split == 'train'")[['topicText', 'claims.claimCorrectedText', 'claims.stance']])
        self.dev_set   = DataframeDataset(df.query("split == 'test'")[['topicText', 'claims.claimCorrectedText', 'claims.stance']])
        self.test_set  = DataframeDataset(df.query("split == 'test'")[['topicText', 'claims.claimCorrectedText', 'claims.stance']])

        self.initialize(batch_size)
        self.device = device


class ListDataset(Dataset):
    
    def __init__(self, lst):
        self.lst = lst
        
    def __len__(self):
        return len(self.lst)
        
    def __getitem__(self, idx):
        return self.list[idx]
        
        
class MRPCBatchManager(BatchManager):
    """
    Batch Manager for the Microsoft Research Paraphrase Corpus dataset
    """
    
    def collate(self, samples):
        return [(example[1], example[2]) for example in samples],\
                    torch.tensor([example[0] for example in samples], device = self.device, requires_grad = False)
    
    def __init__(self, batch_size = 32, device = 'cpu'):
        """
        Initializes the dataset

        Args:
            batch_size: Number of elements per batch
            device    : Device to run it on: cpu or gpu
        """

        train_reader =  open('.data/MRPC/msr_paraphrase_train.txt', 'r')
        train_set = [example.split("\t") for example in train_reader.readlines()][1:]
        # datasets are of the form: [label, id, id, sent_1, sent_2]
        # we only keep [label, sent_1, sent_2]
        train_set = [(int(sample[0]), sample[3], sample[4]) for sample in self.train_set]

        test_reader =  open('.data/MRPC/msr_paraphrase_test.txt', 'r')
        test_set = [example.split("\t") for example in test_reader.readlines()][1:]
        test_set = [(int(sample[0]), sample[3], sample[4]) for sample in test_set]
        
        #TODO: maybe make new train/valid/test split so we have validation data?
        
        self.train_set = ListDataset(train_set)
        self.valid_set = ListDataset(test_set)
        self.test_set  = ListDataset(test_set)

        self.initialize(batch_size)
        self.device = device


if __name__ == "__main__":
    batchmanager = IBMBatchManager()
    #batchmanager = MultiNLIBatchManager()