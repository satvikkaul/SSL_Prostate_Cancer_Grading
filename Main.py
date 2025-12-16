#%% GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#%% Libraries
import tensorflow as tf
import pandas as pd
from my_data_generator import DataGenerator
from variational_autoencoder import ConvVarAutoencoder
from utils_image_retrieval import save_reconstructed_images, create_environment, create_json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from PIL import Image,ImageOps
import pickle
#%%
# Hyper-parametrs
input_dim = (512, 512, 3)
encoder_conv_filters = [16, 32, 64, 128, 256]
encoder_conv_kernel_size = [3, 3, 3, 3, 3]
encoder_conv_strides = [2, 2, 2, 2, 2]
bottle_conv_filters = [64, 32, 1, 256]
bottle_conv_kernel_size = [3, 3, 3, 3]
bottle_conv_strides = [1, 1, 1, 1]
decoder_conv_t_filters = [128, 64, 32, 16, 3]
decoder_conv_t_kernel_size = [3, 3, 3, 3, 3]
decoder_conv_t_strides = [2, 2, 2, 2, 2]
bottle_dim = (32, 32, 256)
z_dim = 200
r_loss_factor = 10000
lr = 0.0005
batch_size = 16
epochs = 5
is_training = True
conv_layeri=[]
conv_t_layeri=[]
#%% I/O paths
run_folders = {
    "tsv_path": "./dataset/Train.csv"
    ,"tsv_path_test":"./dataset/Test.csv"
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

label2=[]
df_pneumo_2d = pd.read_csv(run_folders["tsv_path"])
df_pneumo_2d.columns = ['image_name', 'NC', 'G3', 'G4', 'G5']

df_pneumo_2d=(df_pneumo_2d.iloc[:,:])
image_names=df_pneumo_2d["image_name"]
dataset = []
image_directory="./dataset/images/"
for t,image_name in enumerate (image_names):
    
    if (image_name.split('.')[1]=="jpg"):
        image=plt.imread(image_directory+image_name)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image=Image.fromarray(image) 
        # image=image.resize((128, 128))        
        dataset.append(np.array(image))
        label2.append(t)

dataset2=np.array(dataset)

#################
from sklearn.model_selection import train_test_split   
x_train, x_val, ytrain,ytest=train_test_split(df_pneumo_2d,label2, test_size = 0.2, random_state = 0)
# Saving Training set
# x_train.to_csv (r'E:\ESR\General Codes\CAE\CAE.VAE\x_train512.csv', index = True, header=True)


if is_training:

    # import pdb; pdb.set_trace()
    data_flow_train = DataGenerator(x_train
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

    data_flow_dev = DataGenerator(x_val
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
     


