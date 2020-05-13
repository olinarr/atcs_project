import itertools as it
import random

import torch

class BalancedSampler(torch.utils.data.Sampler):
    """
    Used to sample in a balanced way from one of a BatchManager's DataLoaders.
    """

    def __init__(self, dataset, batch_manager):
        l2i = batch_manager.l2i
        self.label_indices = { i : [] for lbl,i in l2i.items()  }
        for lbl, idx in batch_manager.label_indices[dataset].items():
            self.label_indices[l2i[lbl]].extend(idx)

        #TODO possibly allow for a subset to sample from to be specified
        # this would be usefule primarily for dividing samples between workers

    def __iter__(self):
        idxs = list(self.label_indices.values())

        def iterable(idx):
            return ( idx[i] for i in torch.randperm(len(idx))  ) 

        iterables = [ iterable(idx) for idx in idxs ]

        # roundrobin
        nexts = it.cycle(iter(itr).__next__ for itr in iterables)
        while True:
            try:
                for nxt in nexts:
                    yield nxt() 
            except StopIteration:
                raise 
  
    def __len__(self):
        return min(len(idx) for idx in self.label_indices.values()) 

