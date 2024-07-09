# Data Augmentation Based on Large Language Models for Radiological Report Classification

This repository contains all the code implemented for the paper "Data Augmentation Based on Large Language Models for Radiological Report Classification". Each one of the scripts is detailed below:

* `data_aug.py`: Augments the underrepresented chapters in the CARES dataset using `gpt-3.5-turbo` rephrasing.
* `finetuning_*`: Different ways to finetune our model to the chapter classification task. There are 3 different versions for ablation analysis purposes.
* `further_pretraining.py`: Code to further pre-train the model before finetuning it.
* `inferece_multilabel.py`: Code to extract predictions on the test set.
