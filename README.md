## A Multilingual Transformer-Based System for Opinion Role Labeling (ORL)

This repository contains codes and configs for cross-lingual model training on ORL task. 

There are 10 main folders, each of which represents model from different configuration setting.

## Train and Test
- 1. To train the model, you should get into one folder what you want
- 2. You can see all training, validation, and testing in `expdata/dataset_[SEED_NUMBER]`.
- 3. Configure the hyperparameter in `expdata/opinion.cfg`
- 4. set the GPU device in `runme.sh` and execute `sh runme.sh`
- 5. During the training process, model will be test when a better model is found.
- 6. At the end, all the testing results and the best model will be saved into `expdata/dataset_[SEED_NUMBER]` and `expdata/dataset_output_[LANG]_[SEED_NUMBER]`, respectively.

## Raw and Processed Dataset
- All the raw and processed dataset can be seen in [Thesis_Dataset](https://swtrepo.informatik.uni-mannheim.de/chou/Thesis_Dataset/-/tree/master) repository.


## Reference
>Chinese Opinion Role Labeling with Corpus Translation: A Pivot Study [EMNLP2021]