import math
from heapq import heappop, heappush, heapify


class EpisodeLoader(data.IterableDataset)
"""
Used to obtain episodes for meta-learning, i.e. batches of 'tasks' (support and query sets).
Essentially a dataset of dataloaders.
"""
    @classmethod
    def weight_equal(length):
        return 1
    
    @classmethod
    def weight_sqrt(length):
        return math.sqrt(length)
    
    @classmethod
    def weight_proportional(length):
        return length
    

    def __init__(self, k, batch_managers, samples_per_episode=2, weight_fn=None):
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
        self.batch_managers = batch_managers
        self.weight_fn = weight_fn

        self.weighted_lengths = [self.weight_fn(len(btchmngr.train.set)) for btchmngr in batch_managers]
        self.total_weighted = sum(weighted_lengths)
        self.target_proportions = [weighted / self.total_weighted for weighted in self.weighted_lengths]
        
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            nr_workers = 1
            worker_id = 0
        else:
            nr_workers = worker_info.num_workers
            worker_id = worker_info.id

        iter_starts = []
        iter_ends = []
        iter_lengths = []

        for btchmngr in self.batch_managers:
            setsize = len(btchmngr.train_set)
            
            per_worker = int(math.ceil(setsize/float(nr_workers)))
            iter_starts.append(worker_id * per_worker)
            iter_ends.append(min(iter_start + per_worker, setsize-1))
            iter_lengths.append(iter_ends[-1] - iter_starts[-1])

            
        def create_dataloader(i):
            subset = self.batch_managers[i].train_set[iter_starts[i]:iter_ends[i]]
            return DataLoader(subset, batch_size=self.k, shuffle=True, collate_fn=self.batch_manager[i].collate, drop_last=True)
        
        # Use heap as prioqueue to select tasks in correct proportion.
        worker_subsets = [(0, 0, iter_lengths[i], create_dataloader(i), i) for i,_ in enumerate(self.batch_managers)]
        heapify(worker_subsets)
        
        while not done:
            prio, count, length, next_set, i = heappop(worker_subsets)
            
            # check if dataloader is done, if so replace
            if length - self.samples_per_episode >= count % length:
                next_set = create_dataloader(i)
            
            # push it back on with new priority before yielding.
            count += 1
            cur_prop = count / self.total_weighted
            prio = cur_prop - self.target_proportions[i]
            heappush( worker_subsets, (prio, count, length, next_set, i) )
                              
            yield next_set # this is the T_i from which we draw D_tr and D_val
            