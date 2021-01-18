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
* run `python train_fastgae.py`

# How to run faster and save memory
* Change your arguments in `args.py` : you can lower the value of `emb_size` and `sample_size` to reduce memory.

# Notes

* The dataset is the same as what Guillaume provided in his original implementation.
* Per-epoch training time is a bit slower then the original implementation.
* Dynamic updates of pos_weight before calculating loss are implemented to improve performance.
* Train accuracy, validation(test) average precision, auroc are not implemented due to time limit.
* Dropout is not implemented now.
* Pair-Normalization is implemented to accelerate the training.
* Feel free to report some inefficiencies in the code! (It's just initial version so may have much room for pytorch-adaptation)
