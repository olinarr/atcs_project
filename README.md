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

* `.data` for the NLI dataset
  * Downloading is taken care of by PyTorch
* `IBM` for the IBM dataset
  * Download dataset 3.1 from [here](https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml#Claim%20Stance). 