import os

"""General parameters"""
ROOT_PATH = "/home/yemika/Mikael/Code/Python/Kaggle/mnist"              # path to the root of the repository
DATA_PATH = os.path.join(ROOT_PATH, "data/")                            # path to the "data/" directory
NUM_CLASSES = 10                                                        # number of categories (classes) in the problem
SAVE_PATH = "saved_models/"                                             # path where the models are stored
MODEL_NAME = "model.pt"                                                 # model is saved with this name and loaded from this name is LOAD_MODEL=True

"""Ensemble training parameters"""
N_MODELS = 11                                                           # number of models in the ensemble. Preferable to use an odd number greater than 10

"""Train parameters"""
N_EPOCHS = 30                                                           # total number of epochs for training
BATCH_SIZE = 1024                                                       # batch size -- both for training and inference
LR = 0.001                                                              # initial learning rate
SCHEDULER_STEP = 5                                                      # frequency of epochs after which the LR is reduced
SCHEDULER_GAMMA = 0.5                                                   # LR reduction each SCHEDULER_STEP epochs
LOAD_MODEL = False                                                      # if True, it tries to load the model stored at SAVE_PATH/MODEL_NAME
VAL_FREQ = 5                                                            # frequency of validation
PLOT_RESULTS = False                                                    # if True, training details are plotted after training is finished

"""Inference parameters"""
TEST_DATASET = True                                                     # if True, then no labels are attempted to be loaded during inference
PREDS_PATH = os.path.join(ROOT_PATH, "predictions/submission.csv")      # path where to save the predictions .csv file
PLOT_FILTERS = False                                                    # if True, then intermediate layer filters are plotted during inference