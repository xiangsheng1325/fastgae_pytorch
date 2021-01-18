# FastGAE in Pytorch
This repository implements FastGAE by Guillaume Salha For details of the model, refer to his original [tensorflow implementation](https://github.com/deezer/fastgae) and [his paper](https://arxiv.org/abs/2002.01910).

# Requirements

* Pytorch 
* python 3.x
* networkx
* scikit-learn
* scipy

# How to run
* Specify your arguments in `args.py` : you can change dataset and other arguments there
* run `python train.py`

# Notes

* The dataset is the same as what Guillaume provided in his original implementation.
* Per-epoch training time is a bit slower then the original implementation.
* Train accuracy, validation(test) average precision, auroc are eliminated due to time limit.
* Dropout is not implemented now.
* Feel free to report some inefficiencies in the code! (It's just initial version so may have much room for pytorch-adaptation)
