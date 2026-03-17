#%% GPU
import os
import sys
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
#%% Libraries
import tensorflow as tf
import pandas as pd
from data.generator import DataGenerator
from models.cae_model import ConvVarAutoencoder
from utils.utils_retrieval import save_reconstructed_images, create_environment, create_json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from PIL import Image,ImageOps
import pickle


def print_device_configuration():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = tf.config.list_physical_devices('GPU')
    if visible_devices:
        print(f"CUDA_VISIBLE_DEVICES preset to: {visible_devices}")
    else:
        print("CUDA_VISIBLE_DEVICES not set; using TensorFlow default device selection.")
    print(f"Detected GPUs: {len(gpus)}")
 
#%%
# Hyper-parametrs (MATCHING PAPER: 128x128 patches!)
input_dim = (128, 128, 3)  # Paper uses 128x128, NOT 512x512!
encoder_conv_filters = [16, 32, 64, 128, 256]
encoder_conv_kernel_size = [3, 3, 3, 3, 3]
encoder_conv_strides = [2, 2, 2, 2, 2]
bottle_conv_filters = [128, 64, 128]  # Removed 1-filter bottleneck, cleaner architecture
bottle_conv_kernel_size = [3, 3, 3]
bottle_conv_strides = [1, 1, 1]
decoder_conv_t_filters = [128, 64, 32, 16, 3]
decoder_conv_t_kernel_size = [3, 3, 3, 3, 3]
decoder_conv_t_strides = [2, 2, 2, 2, 2]
bottle_dim = (16, 16, 128)  # Updated to match new bottleneck architecture
z_dim = 256  # Increased from 200 for better feature capacity
r_loss_factor = 10000
lr = 0.0005
batch_size = 16
epochs = 15  # Reduced from 5 to prevent overfitting to pixel-perfect reconstruction
is_training = True
conv_layeri=[]
conv_t_layeri=[]
#%% I/O paths
run_folders = {
    "tsv_path": "./dataset/TrainSplit.csv"
    ,"tsv_path_val":"./dataset/Val.csv"
    , "data_path": "./dataset/images/"
    , "model_path": './output/models/'
    , "results_path": './output/results/'
    , "log_filename": './output/prostate_cancer.csv'
}
# Creating the required folders
create_environment(run_folders)


# Building JSON with the model hyperparameters
hyperparameters = {
    "input_dim": input_dim
    , "encoder_conv_filters": encoder_conv_filters
    , "encoder_conv_kernel_size": encoder_conv_kernel_size
    , "encoder_conv_strides": encoder_conv_strides
    , "bottle_conv_filters" : bottle_conv_filters
    , "bottle_conv_kernel_size": bottle_conv_kernel_size
    , "bottle_conv_strides":  bottle_conv_strides
    , "decoder_conv_t_filters": decoder_conv_t_filters
    , "decoder_conv_t_kernel_size": decoder_conv_t_kernel_size
    , "decoder_conv_t_strides": decoder_conv_t_strides
    , "z_dim": z_dim
    , "r_loss_factor": r_loss_factor
    , "learning_rate": lr
    , "batch_size": batch_size
    , "epochs": epochs
    , "opt": "Adam"
    , "loss_function": "mse"
    , "data_path": run_folders["data_path"]
    , "bottle_dim" : bottle_dim 
}
create_json(hyperparameters, run_folders)

df_train = pd.read_csv(run_folders["tsv_path"])
df_val = pd.read_csv(run_folders["tsv_path_val"])
print_device_configuration()
print(f"Loaded {len(df_train)} CAE training samples from {run_folders['tsv_path']}")
print(f"Loaded {len(df_val)} CAE validation samples from {run_folders['tsv_path_val']}")


if is_training:

    # import pdb; pdb.set_trace()
    data_flow_train = DataGenerator(df_train
                                    , input_dim[0]
                                    , input_dim[1]
                                    , input_dim[2]
                                    , indexes_output=[True, True, False, False]
                                    , batch_size=batch_size
                                    , path_to_img=run_folders["data_path"]
                                    , data_augmentation=True
                                    , vae_mode=True
                                    , reconstruction=True
                                    , softmax=False
                                    , hide_and_seek=False
                                    , equalization=False
                                    )

    data_flow_dev = DataGenerator(df_val
                                  , input_dim[0]
                                  , input_dim[1]
                                  , input_dim[2]
                                  , indexes_output=[True, True, False, False]
                                  , batch_size=batch_size
                                  , path_to_img=run_folders["data_path"]
                                  , data_augmentation=True
                                  , vae_mode=True
                                  , reconstruction=True
                                  , softmax=True
                                  , hide_and_seek=False
                                  , equalization=False
                                  )

    # VAE instance
    my_VAE = ConvVarAutoencoder(input_dim
                 , encoder_conv_filters
                 , encoder_conv_kernel_size
                 , encoder_conv_strides
                 , bottle_dim
                 , bottle_conv_filters
                 , bottle_conv_kernel_size
                 , bottle_conv_strides
                 , decoder_conv_t_filters
                 , decoder_conv_t_kernel_size
                 , decoder_conv_t_strides
                 , z_dim)

    # Buildig VAE
    # import pdb; pdb.set_trace()
    my_VAE.build(use_batch_norm=True, use_dropout=True)
    print(my_VAE.model.summary())

    # Compiling VAE
    my_VAE.compile(learning_rate=lr, r_loss_factor=r_loss_factor)

    # Training VAE
    steps_per_epoch = len(data_flow_train)
    H = my_VAE.train_with_generator(data_flow_train, epochs, steps_per_epoch, data_flow_dev, run_folders)
     


