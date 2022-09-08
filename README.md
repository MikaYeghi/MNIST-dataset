This repository represents a model that is capable of predicting digits in the [MNIST](https://www.kaggle.com/competitions/digit-recognizer/data) data set with a very high accuracy on the test set (>98%). It uses [EfficientNet](https://arxiv.org/abs/1905.11946) as the backbone for image classification.

!!!Before running any script, change the absolute path of the ``ROOT_PATH`` variable in ``config.py`` to the root path of the local copy of this repository on your machine!!!

In order to run the full pipeline, first download the data set by running:

```
python prepare_data.py
```

The command above will download the data set and unzip it in the "raw_data/" folder in the root directory, then randomly split the training data into training and validation sets, and save training, validation and test sets as .csv files to the "data/" folder in the root directory. Note that it performs an agnostic split, so it might result in an imbalanced split.

To train on the newly obtained data set, run the following command:

```
python train.py
```

This will train on the data set using the configurations defined in the ``config.py`` file and plot training and validation losses, as well as the learning rate over iterations at the end of the training process.

Next, there are 2 options to run inference. First, you can run 

```
python inference.py
```

with the ``TEST_DATASET`` variable in ``config.py`` set to ``False``, which will assume that the test data set stored in ``data/test.csv`` also contains the ground-truth labels, thus also printing the accuracy of the model. In case there are no ground-truth labels available for the test set, run 

```
python inference.py
```

with the ``TEST_DATASET`` variable in ``config.py`` set to ``True``. In this case the model will not try to load any ground-truth labels, and instead it will save the prediction to ``predictions/submission.csv`` with 2 columns: ImageId and Label. This corresponds to the submission format required by the [Kaggle competition](https://www.kaggle.com/competitions/digit-recognizer/overview).
