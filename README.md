# Meta Learning
Project on Meta-Learning for the "Advanced Topics on Computational Semantics" course (UvA, 2020)


## How To

* Single-task training can be started by running the following line in your Command Line Interface: `python3 train.py`. Possible command-line options include:
  * `batch_size` to set the number of elements in a batch
  * `random_seed` to set a seed for reproducibility
  * `epochs` to set the number of epochs to run for
  * `dataset`, e.g. `NLI` or `IBM` to set the dataset you want to use (for single task only)
  * More parameters can be found by executing `train.py --help` or in the file `train.py` itself.

Similarly Multi-task training, training of Prototypical networks and Proto-MAML can be started with the python scripts `multitask.py`, `prototypes.py` and `protomaml.py`.

## Dataset Locations

* `.data/multinli` for the NLI dataset
  * Downloading is taken care of by PyTorch
* `.data/ibm` for the IBM dataset
  * Download dataset 3.1 from [here](https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml#Claim%20Stance). 
* `.data/MRPC` for the MRPC dataset
  * Download dataset from [here](https://github.com/wasiahmad/paraphrase_identification/tree/master/dataset/msr-paraphrase-corpus).
* `.data/pdb` for the Penn Discourse Bank
  * Contact us for an appropriatly pre-processed version; or
  * Download the raw dataset from [here](https://www.sfu.ca/rst/06tools/discourse_relations_corpus.html).
* `.data/SICK` for SICK
  * Download dataset from [here](https://www.microsoft.com/en-us/download/details.aspx?id=52398).
