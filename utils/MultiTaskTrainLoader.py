from utils.batchManagers import MultiNLIBatchManager, MRPCBatchManager, PDBBatchManager, SICKBatchManager, IBMBatchManager
from math import sqrt
import random

random.seed(42)

class MultiTaskTrainLoader():
    """ Custom batch manager for multi-task learning. Iterating over this object yields a batch from one of the datasets (randomly) """

    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device

        # batchmanagers we care about
        self.batchmanagers = {
            'NLI'  : MultiNLIBatchManager(batch_size, device),
            'PDB'  : PDBBatchManager(batch_size, device),
            'MRPC' : MRPCBatchManager(batch_size, device)
        }
        
        self.tasks = list(self.batchmanagers.keys())

        # save the iterator of the dataloaders
        # need to do so because dataloaders are not iterators directly

        self.iter_dataloaders = {task:iter(bm.train_iter) for task, bm in self.batchmanagers.items()}

        # this is used to sample the dataloaders. See function description

        self.proportions = self._getProportions()
        # task iterator (we shuffle every time)
        random.shuffle(self.proportions)
        self.task_iter = iter(self.proportions)

        # total number of batches per one epoch

        self.totalBatches = max((bm.task_size() for bm in self.batchmanagers.values())) // self.batch_size

        # used to iterate.

        self.counter = 0

    def getTasksWithNClasses(self):
        return {name: len(bm.classes()) for name, bm in self.batchmanagers.items()}

    def _getProportions(self):
        """ returns a list of strings, each string is the name of the task. The number of strings in this list are proportional to the sizes of the datasets (square rooted)

        Returns:
        (list(str)): list representing the proportions """

        min_size = min((bm.task_size() for bm in self.batchmanagers.values()))
        proportions = []
        for name, bm in self.batchmanagers.items():
            size = round(sqrt(bm.task_size() / min_size))
            proportions += [name] * size

        return proportions

    def __len__(self):
        return self.totalBatches

    def __iter__(self):
        return self

    def __next__(self):
        """ Iterator main function

        Returns:
        (str, object): name of the task and a batch """

        # if we are out of index, stop the iteration...
        # (but restart the counter for next time!)
        if self.counter >= self.totalBatches:
            self.counter = 0
            raise StopIteration
        else:
            # augment the index
            self.counter += 1

            # pick a task (this is a string)
            try:
                task = next(self.task_iter)
            except StopIteration:
                random.shuffle(self.proportions)
                self.task_iter = iter(self.proportions)

                task = next(self.task_iter)

            # get the corresponding dataloader-iterator
            dataloader = self.iter_dataloaders[task]

            try:
                # get the next batch
                batch = next(dataloader)
            except StopIteration:
                # if this did not work, restart the iterator.
                self.iter_dataloaders[task] = iter(self.batchmanagers[task].train_iter)
                dataloader = self.iter_dataloaders[task]
                batch = next(dataloader)

            return task, batch