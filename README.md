# Meta Learning
Project on Meta-Learning for the "Advanced Topics on Computational Semantics" course (UvA, 2020)

## How To

* This project can most easily be executed by running the following line in your Command Line Interface: `python3 train.py --batch_size = 256`, where batch_size is a parameter.  Parameters include:
  * `batch_size` to set the number of elements in a batch
  * `random_seed` to set a seed for reproducibility
  * `epochs` to set the number of epochs to run for
  * `dataset`, e.g. `NLI` or `IBM` to set the dataset you want to use (for single task only)
  * More parameters can be found in the file `train.py`

## Dataset Locations

* `.data/multinli` for the NLI dataset
  * Downloading is taken care of by PyTorch
* `.data/ibm` for the IBM dataset
  * Download dataset 3.1 from [here](https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml#Claim%20Stance). 
* `.data/MRPC` for the MRPC dataset
  * Download dataset from [here](https://github.com/wasiahmad/paraphrase_identification/tree/master/dataset/msr-paraphrase-corpus).

## TO DO:

* [x] ..
* [ ] New test / dev split for MRPC
* [x] Implement the k-times-per-label thing on the multitask eval
* [ ] Verify we are doing the k-times-per-label thing on PROTOMAML
* [ ] Sort out splits
* [ ] Fix number of workers for multi-threading
* [ ] protomaml.py
  * [ ] Make gradients flow inside this function?
* [ ] TODO's in episodeLoader.py?
* [ ] Retrieve IBM BatchManager that actually did shuffle
* [ ] Shall we have taskspecific learning rates in multi-task?