import numpy as np

import torch
import torch.utils.data as data

import math, itertools

from heapq import heappop, heappush, heapify


class EpisodeLoader(data.IterableDataset):
    """
    Used to obtain episodes for meta-learning, i.e. batches of 'tasks' (support and query sets).
    Essentially a dataset of dataloaders.
    """
    
    @classmethod
    def weight_equal(cls, length):
        return 1
    
    @classmethod
    def weight_sqrt(cls, length):
        return math.sqrt(length)
    
    @classmethod
    def weight_proportional(cls, length):
        return length
    
    @classmethod
    def create_dataloader(cls, k, batch_managers, batch_size, samples_per_episode=2, weight_fn=None, num_workers=0):
        # TODO actually fix this problem with multithreading...
        if num_workers != 0:
            print("num_workers overriden to 0 as temporary fix for problem with MultiNLI")
            num_workers=0

        episodeDataset = EpisodeLoader(k, batch_managers, samples_per_episode=samples_per_episode, weight_fn=weight_fn)
        collate_fn = lambda x : x # have identity function as collate_fn so we just get list.
        return data.DataLoader(episodeDataset, collate_fn = collate_fn, batch_size = batch_size, num_workers=num_workers)
    
    def __init__(self, k, batch_managers, samples_per_episode=2, shuffle_labels=True, weight_fn=None):
        """
        Params:
          k: the amount of samples included in every support set.
          batch_managers: the batch managers to draw samples from.
          samples_per_episode: the amount of times that each 'task' will 
              be sampled, default=2 (for standard MAML, one to train 
              with (support set) and one to evaluate with (query set).
          weight_fn: function that will be applied to dataset lengths before
              calculating in what proportion dataloaders from each dataset 
              should be returned. (Default: EpisodeLoader.weight_sqrt)
        """
        super(EpisodeLoader).__init__()

        if not weight_fn:
            weight_fn = EpisodeLoader.weight_sqrt
        
        self.k = k
        self.samples_per_episode = samples_per_episode
        self.shuffle_labels = shuffle_labels
        self.batch_managers = batch_managers
        self.weight_fn = weight_fn

        self.weighted_lengths = [self.weight_fn(btchmngr.task_size()) for btchmngr in batch_managers]
        self.total_weighted = sum(self.weighted_lengths)
        self.target_proportions = [weighted / self.total_weighted for weighted in self.weighted_lengths]
        
        
    def __iter__(self):
        
        # If we have multiple workers, we make sure to not have them all return the same data.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            nr_workers = 1
            worker_id = 0
        else:
            nr_workers = worker_info.num_workers
            worker_id = worker_info.id

        iter_starts = []
        iter_ends = []

        for btchmngr in self.batch_managers:
            setsize = len(btchmngr.train_set)
            
            per_worker = int(math.ceil(setsize/float(nr_workers)))
            iter_starts.append(worker_id * per_worker)
            iter_ends.append(min(iter_starts[-1] + per_worker, setsize-1))

            
        def get_dataloader(i):
            samples = np.arange(iter_starts[i], iter_ends[i])
            sampler = data.SubsetRandomSampler(samples)
            return data.DataLoader(self.batch_managers[i].train_set, batch_size=self.k,\
                                   sampler=sampler, collate_fn=self.batch_managers[i].collate, drop_last=True)
        

        # Use heap for prioqueue to select tasks in correct proportion.
        worker_subsets = [(0, 0, i, get_dataloader(i), bm) for i,bm in enumerate(self.batch_managers)]
        heapify(worker_subsets)
        
        while True:
            prio, count, i, next_set, bm = worker_subsets[0]
           
            if self.shuffle_labels:
                bm.randomize_class_indices()

            # update count
            count += 1
            worker_subsets[0] = (prio, count, i, next_set, bm)
            
            # calculate new priorities
            def calc_prio(i, count, total_count):
                return (count / total_count) - self.target_proportions[i]
            
            total_count = sum(count for _, count, _, _, _ in worker_subsets)
            worker_subsets = [(calc_prio(i, count, total_count), count, i, next_set, bm) \
                              for prio, count, i, next_set, bm in worker_subsets]
            
            # update heap with new priorities before yielding.
            heapify(worker_subsets)
                              
            yield (next_set, bm) # next_set is the T_i from which we draw D_tr and D_val
        
        
        
if __name__ == "__main__":
    # test this class
    
    k = 8
    samples_per_episode = 2
    batch_size = 4
    
    device = "cpu"
    from batchManagers import MultiNLIBatchManager, IBMBatchManager, MRPCBatchManager, SICKBatchManager, PDBBatchManager
    batchManager1 = IBMBatchManager(batch_size = k, device = device)
    batchManager2 = MRPCBatchManager(batch_size = k, device = device)      
    batchManager3 = MultiNLIBatchManager(batch_size = k, device = device)      
    batchManager4 = PDBBatchManager(batch_size = k, device = device)
    #batchManager5 = SICKBatchManager(batch_size = k, device = device)

    """
    samples = np.arange(0, 50)
    sampler = data.SubsetRandomSampler(samples)

    print("Confirming that MultiNLI can work with dataloader")
    test = data.DataLoader(batchManager3.train_set, batch_size=k,\
                           sampler=sampler, collate_fn=batchManager3.collate, drop_last=True, num_workers=2)
    for x in test:
        print(x)
    """    

    bms = [batchManager1, batchManager2, batchManager3]
    #bms.extend(list(batchManager3.get_subtasks(2)))
    bms.extend(list(batchManager4.get_subtasks(2)))

    episodeLoader = EpisodeLoader.create_dataloader(
        k, bms, batch_size,
        samples_per_episode = samples_per_episode
    )

    for i, batch in enumerate(episodeLoader): 
        if i == 5000:
            break
        print("=====")            
        for j, (episode, bm) in enumerate(batch):
           
            print("Episode of {:>25} can have class-indices: {:12} {}".format(type(bm).__name__, str(bm.classes()), str(bm.l2i)))
            
            for k, task in enumerate(episode):
                
                if k + 1 == samples_per_episode:
                    break
                    
            assert k + 1 == samples_per_episode
        assert j + 1 == batch_size


            
