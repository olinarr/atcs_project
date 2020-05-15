import os, random
  
import torchtext, torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd

import numpy as np

from sklearn.utils import shuffle

class BatchManager():
    """
    Class to define a batch manager.
    For new implementations:
     *  create a subclass;
     *  in the constructer create a self.train_set, self.dev_set and self.test_set;
     all of which should be subclasses of the torch.utils.data.Dataset class;
     *  create a l2i dictionary that maps between labels in the data and indices
     *  call the initialize function at the end of your constructor;
     *  override the collate function which should take a list of dataset elements 
     and combine them into a batch, make sure to use the l2i dict to get the labels
  
     
     Override class variables in your subclass as needed.
    """
    
    SHUFFLE = True 
    MAX_NR_OF_SUBTASKS = 10
   
    def _extract_label(self, sample):
        raise NotImplementedError

    def _extract_sentences(self, sample):
        raise NotImplementedError

    def collate(self, samples):
        return [self._extract_sentences(example) for example in samples],\
                torch.tensor([self.l2i[self._extract_label(example)] for example in samples], device = self.device, requires_grad = False)

    def classes(self):
        return list(set(self.l2i.values()))

    def task_size(self):
        return len(self.train_set)

    def __init__(self, batch_size):

        self.weight_factor = 1

        # create the three iterators
        self.train_iter = DataLoader(self.train_set, batch_size=batch_size, shuffle=self.SHUFFLE, collate_fn=self.collate)
        self.dev_iter   = DataLoader(self.dev_set,   batch_size=batch_size, shuffle=self.SHUFFLE, collate_fn=self.collate)
        self.test_iter  = DataLoader(self.test_set,  batch_size=batch_size, shuffle=self.SHUFFLE, collate_fn=self.collate)

        sets = [ self.train_set, self.dev_set, self.test_set ]
        self.label_indices = { s : { lbl : [] for lbl in self.l2i.keys() } for s in sets }
        for s in sets:
            for i,e in enumerate(s):
                lbl = self._extract_label(e)
                self.label_indices[s][lbl].append(i)

        sizes = [str(len(s)) for s in sets]
        print("Split {} (train/dev/test): {}".format(self.name, '/'.join(sizes)))



    def randomize_class_indices(self):
        classes = self.classes()
        random.shuffle(classes)
        
        mapping = dict(enumerate(classes))
        self.l2i = { lbl : mapping[i] for lbl, i in self.l2i.items() }

    def _get_partitions(self, k):
        import more_itertools
        return more_itertools.set_partitions(self.l2i.keys(), k)

    def get_subtasks(self, k = None):
        import copy 
        
        partitions = list(self._get_partitions(k))

        if len(partitions) > self.MAX_NR_OF_SUBTASKS:
            print("WARNING: the amount of partitions for {} is {} > {}.".format(self, len(partitions), self.MAX_NR_OF_SUBTASKS))

        for j, part in enumerate(partitions):
            subtask = copy.copy(self)
            subtask.l2i = { label : i for i,sub in enumerate(part) for label in sub }
            subtask.name = self.name + '_st{}'.format(j)
            subtask.weight_factor = 1 / len(partitions)
            subtask.parent = self
            yield subtask
    

class MultiNLIBatchManager(BatchManager):
    
    def _extract_sentences(self, example):
        return (example.premise, example.hypothesis)

    def _extract_label(self, example):
        return example.label

    def __init__(self, batch_size = 32, device = 'cpu'):
        # sequential false -> no tokenization. Why? Because right now this
        # happens instead of BERT
        TEXT = torchtext.data.Field(use_vocab = False, sequential = False)
        LABEL = torchtext.data.Field(sequential = False)

        self.train_set, self.dev_set, self.test_set = torchtext.datasets.MultiNLI.splits(TEXT, LABEL)

        self.train_set = ListDataset(list(self.train_set))
        self.dev_set   = ListDataset(list(self.dev_set))
        self.test_set  = ListDataset(list(self.test_set))

        self.batch_size = batch_size

        # mapping from classes to integers
        self.l2i = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        
        self.device = device
        self.name = 'multinli'

        super(MultiNLIBatchManager, self).__init__(batch_size)
        
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
    def _extract_sentences(self, sample):
        return (sample['topicText'], sample['claims.claimCorrectedText']) 

    def _extract_label(self, sample):
        return sample['claims.stance']    

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
        df = pd.read_csv(os.path.join(".data", "ibm", "claim_stance_dataset_v1.csv"))

        # Shuffle and reset index for loc
        df = shuffle(df)
        df.reset_index(inplace=True, drop = True)

        # Get dataset split 80/20
        df.loc[:int(0.8*len(df)),'split'] = 'train'
        df.loc[int(0.8*len(df)):int(0.9*len(df)),'split'] = 'test'
        df.loc[int(0.9*len(df)):,'split'] = 'dev'

        # Generate splits
        self.train_set = DataframeDataset(df.query("split == 'train'")[['topicText', 'claims.claimCorrectedText', 'claims.stance']])
        self.dev_set   = DataframeDataset(df.query("split == 'dev'")[['topicText', 'claims.claimCorrectedText', 'claims.stance']])
        self.test_set  = DataframeDataset(df.query("split == 'test'")[['topicText', 'claims.claimCorrectedText', 'claims.stance']])

        # Sanity check for lengths
        assert len(self.train_set) == 1915
        assert len(self.test_set ) == 239
        assert len(self.dev_set  ) == 240

        self.device = device
        self.name = 'ibm'

        super(IBMBatchManager, self).__init__(batch_size)
        
class ListDataset(Dataset):
    
    def __init__(self, lst):
        self.lst = lst
        
    def __len__(self):
        return len(self.lst)
        
    def __getitem__(self, idx):
        return self.lst[idx]
        
        
class MRPCBatchManager(BatchManager):
    """
    Batch Manager for the Microsoft Research Paraphrase Corpus dataset
    """
    def _extract_sentences(self, example):
        return (example[1], example[2])

    def _extract_label(self, example):
        return example[0]

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
        train_set = [(int(sample[0]), sample[3], sample[4]) for sample in train_set]

        test_reader =  open('.data/MRPC/msr_paraphrase_test.txt', 'r')
        test_set = [example.split("\t") for example in test_reader.readlines()][1:]
        test_set = [(int(sample[0]), sample[3], sample[4]) for sample in test_set]
        
        #TODO: maybe make new train/valid/test split so we have validation data?
        
        self.train_set = ListDataset(train_set)
        self.dev_set   = ListDataset(test_set)
        self.test_set  = ListDataset(test_set)

        self.l2i = {0: 0, 1: 1}

        self.device = device
        self.name = 'mrpc'

        super(MRPCBatchManager, self).__init__(batch_size)

class SICKBatchManager(BatchManager):
    """
    Batch Manager for the Microsoft Research Paraphrase Corpus dataset
    """

    def _extract_sentences(self, example):
        return (example[1], example[2])

    def _extract_label(self, example):
        return example[0]

    def __init__(self, batch_size = 32, device = 'cpu'):
        """
        Initializes the dataset

        Args:
            batch_size: Number of elements per batch
            device    : Device to run it on: cpu or gpu
        """

        reader =  open('.data/SICK/SICK.txt', 'r')
        data = [example.split("\t") for example in reader.readlines()][1:]
        # datasets are of the form: see files
        # we only keep [label, sent_1, sent_2]
        # labels are rounded cuz we want to do classification
        train_set, dev_set, test_set = [], [], []
        for sample in data:
            if sample[-1] == 'TRAIN\n':
                # we get the REAL valued in [1,5] label
                # we round it and map it to {0,1,2,3,4} (easier to use torch losses)
                train_set.append((round(float(sample[4]))-1, sample[1], sample[2]))
            elif sample[-1] == 'TRIAL\n':
                dev_set.append((round(float(sample[4]))-1, sample[1], sample[2]))
            elif sample[-1] == 'TEST\n':
                test_set.append((round(float(sample[4]))-1, sample[1], sample[2]))
            else:
                raise Exception()

        self.l2i = {i: i for i in [0,1,2,3,4]}
        
        self.train_set = ListDataset(train_set)
        self.dev_set   = ListDataset(dev_set)
        self.test_set  = ListDataset(test_set)

        self.device = device
        self.name = 'sick'

        super(SICKBatchManager, self).__init__(batch_size)


class PDBBatchManager(BatchManager):
    """
    Batch Manager for the Penn Discourse Bank dataset
    """
    def _extract_sentences(self, sample):
        return (sample['sent1'], sample['sent2'])

    def _extract_label(self, sample):
        return sample['label']


    def __init__(self, batch_size = 32, device = 'cpu'):
                # Get a mapping from relations to labels
        self.l2i = {'Temporal.Asynchronous.Succession': 0, 'Comparison.Contrast': 1, 'Expansion.Level-of-detail.Arg2-as-detail': 2, 'Contingency.Cause.Result': 3, 
                    'Temporal.Asynchronous.Precedence': 4, 'Expansion.Conjunction': 5, 'Contingency.Cause.Reason': 6, 'Comparison.Concession.Arg1-as-denier': 7, 
                    'Comparison.Concession.Arg2-as-denier': 8, 'Expansion.Instantiation.Arg2-as-instance': 9, 'Comparison.Similarity': 10, 'Contingency.Condition.Arg2-as-cond': 11, 
                    'Temporal.Synchronous': 12, 'Expansion.Equivalence': 13, 'Expansion.Manner.Arg2-as-manner': 14, 'Expansion.Level-of-detail.Arg1-as-detail': 15, 
                    'Contingency.Purpose.Arg2-as-goal': 16, 'Expansion.Disjunction': 17, 'Expansion.Substitution.Arg2-as-subst': 18, 
                    'Contingency.Negative-condition.Arg2-as-negCond': 19, 'Comparison.Concession+SpeechAct.Arg2-as-denier+SpeechAct': 20, 'Contingency.Cause+Belief.Reason+Belief': 21, 
                    'Contingency.Condition+SpeechAct': 22, 'Contingency.Negative-condition.Arg1-as-negCond': 23, 'Expansion.Manner.Arg1-as-manner': 24, 
                    'Contingency.Cause+Belief.Result+Belief': 25, 'Expansion.Substitution.Arg1-as-subst': 26, 'Contingency.Cause+SpeechAct.Result+SpeechAct': 27, 
                    'Expansion.Exception.Arg2-as-excpt': 28, 'Contingency.Condition.Arg1-as-cond': 29, 'Expansion.Exception.Arg1-as-excpt': 30, 'Contingency.Purpose.Arg1-as-goal': 31, 
                    'Contingency.Negative-cause.NegResult': 32, 'Expansion.Instantiation.Arg1-as-instance': 33, 'Contingency.Cause+SpeechAct.Reason+SpeechAct': 34, 'EntRel': 35, 
                    'NoRel': 36, 'Hypophora': 37}
        
        train = pd.read_csv('./.data/pdb/PDB_train_labeled.csv',sep="|")
        dev = pd.read_csv('./.data/pdb/PDB_dev_labeled.csv',sep="|")
        test = pd.read_csv('./.data/pdb/PDB_test_labeled.csv',sep="|")
        self.train_set = DataframeDataset(train[['sent1','sent2','label']])
        self.dev_set   = DataframeDataset(dev[['sent1','sent2','label']])
        self.test_set  = DataframeDataset(test[['sent1','sent2','label']])
    
        self.device = device
        self.name = 'pdb'

        super(PDBBatchManager, self).__init__(batch_size)


    def _get_partitions(self, k):
        # Returning all possible ways to partition the set of classes is not possible for so many classes.
        # So here we just return a fixed number of random partitions, using a fixed seed for replicability.
        # While also making sure not, for example put different kinds of 'Contingency' in different partitions.
    
        classes = ['Temporal', 'Comparison', 'Expansion', 'Contingency', 'EntRel', 'NoRel', 'Hypophora']
        for _ in range(self.MAX_NR_OF_SUBTASKS):
            random.shuffle(classes)

            parts = np.array_split(classes, k) 
            parts = [[cls for cls in self.l2i.keys() for Cls in part if cls.startswith(Cls)] for part in parts] 
            yield parts
        

if __name__ == "__main__":
    batchmanager1 = PDBBatchManager()
    #batchmanager = IBMBatchManager()
    batchmanager2 = MultiNLIBatchManager()


    #for k in range(2,5):
    #    parts = batchmanager1._get_partitions(5)
    #    for part in parts:
    #        print(part)

    print(batchmanager1.label_indices)

    for test in batchmanager2.get_subtasks(2):
        print(test.l2i) 
