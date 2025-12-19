from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Layer, Add, Concatenate
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from utils_image_retrieval import learning_curve_plot
import numpy as np
import os
import pickle
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras import layers

class ConvVarAutoencoder:

    def __init__(self, input_dim
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
                 , z_dim
                 ):
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.bottle_dim = bottle_dim
        self.bottle_conv_filters = bottle_conv_filters
        self.bottle_conv_kernel_size = bottle_conv_kernel_size
        self.bottle_conv_strides = bottle_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim
        
    @staticmethod
    def load_model_AD(run_folders):
        with open(os.path.join(run_folders["model_path"], run_folders["exp_name"]+'/VAE_model.pkl'), 'rb') as f:
            params = pickle.load(f)
        
        my_VAE = ConvVarAutoencoder(*params)
        my_VAE.build(use_batch_norm=True, use_dropout=True)
        my_VAE.model.load_weights(os.path.join(os.path.join(run_folders["model_path"], run_folders["exp_name"]+'/weights/VAE_weights.h5')))

        return my_VAE

    def build(self, use_batch_norm=False, use_dropout=True,kernel_regularizer=tf.keras.regularizers.l1(0.01)):
        # Defining the encoder part
        
        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input
        # import pdb; pdb.set_trace()
        for i in range(len(self.encoder_conv_filters)):
                        
            conv_layer = Conv2D(filters=self.encoder_conv_filters[i],
                                kernel_size=self.encoder_conv_kernel_size[i],
                                strides=self.encoder_conv_strides[i],
                                padding='same',
                                name='encoder_conv'+str(i))

            x = conv_layer(x)
    
            if use_batch_norm:
                x = BatchNormalization(axis=-1)(x)

            x = LeakyReLU(negative_slope=0.2)(x)

            if use_dropout:
                x = Dropout(rate=0.1)(x)  # Reduced from 0.25 to 0.1 for SSL
            # Save skip connection at encoder layer 2 (output shape: 64x64x64)
            if (i == 2):
                conv_layeri = x  # Save directly without extra conv
                print("Skip layer saved at encoder i=" + str(i))
                
                # Defining the bottle part
        
        bottle_input = Input(shape=self.bottle_dim, name='bottle_input')
        x_bottle = bottle_input
        x_bottle = x
        # import pdb; pdb.set_trace()
        for i in range(len(self.bottle_conv_filters)):
            
            conv_layer_bottle = Conv2D(filters=self.bottle_conv_filters[i],
                                kernel_size=self.bottle_conv_kernel_size[i],
                                strides=self.bottle_conv_strides[i],
                                padding='same',
                                name='bottle_conv'+str(i))
            x_bottle = conv_layer_bottle(x_bottle)
            if use_batch_norm:
                x_bottle = BatchNormalization(axis=-1)(x_bottle)

            x_bottle = LeakyReLU(negative_slope=0.2)(x_bottle)

            if use_dropout:
                x_bottle = Dropout(rate=0.1)(x_bottle)  # Reduced from 0.25 to 0.1
                
        shape_before_flattening = K.int_shape(x_bottle)[1:]
        # import pdb; pdb.set_trace()
        # Removed residual bypass: x_bottle + x (was allowing network to skip compression)
        x_out_bottle = Flatten()(x_bottle)
        # encoder_output in Autoencoder Normal version
        
        # # encoder_output in Variational Autoencoder version
        encoder_output= Dense(self.z_dim, name='encoder_output')(x_out_bottle)
        self.encoder = Model(encoder_input, encoder_output)

        # Defining the decoder part
        # 
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x1 = Dense(np.prod(shape_before_flattening))(encoder_output)
        x1 = Reshape(shape_before_flattening)(x1)
        # import pdb; pdb.set_trace()
        for i in range(len(self.decoder_conv_t_filters)):
            conv_t_layer = Conv2DTranspose(filters=self.decoder_conv_t_filters[i],
                                           kernel_size=self.decoder_conv_t_kernel_size[i],
                                           strides=self.decoder_conv_t_strides[i],
                                           padding='same',
                                           name='decoder_conv_t'+str(i))

  
            if i < len(self.decoder_conv_t_filters) - 1:
                x1 = conv_t_layer(x1)
                # Add skip connection at decoder layer 1 (matches encoder layer 2)
                if (i == 1):
                    x1 = x1 + conv_layeri
                    print("Skip connection added at decoder i=" + str(i))
                x1 = BatchNormalization(axis=-1)(x1)
                x1 = LeakyReLU(negative_slope=0.2)(x1)
            else:
                x1 = conv_t_layer(x1)
                x1 = Activation('sigmoid')(x1)
                
        # decoder_output = Dense( 3 ,name='decoder_output')(x1)
        decoder_output = x1
        # self.decoder = Model(encoder_input, decoder_output)
        var_autoencoder = Model(encoder_input, decoder_output)
        # Joining encoder to the decoder

        # var_autoencoder_input = encoder_input
        # var_autoencoder_ouput = self.decoder(encoder_output)
        # var_autoencoder = Model(var_autoencoder_input, var_autoencoder_ouput)

        self.model = var_autoencoder

    def compile(self, learning_rate=0.005, r_loss_factor=0.4):
        self.learning_rate = learning_rate
        self.r_loss_factor = r_loss_factor

        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
            return self.r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis=1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return r_loss + kl_loss
        
        self.loss_func = vae_loss
        optimizer = Adam(learning_rate=learning_rate)
        # self.model.compile(optimizer=optimizer, loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])
        self.model.compile(optimizer=optimizer, loss="mse")

    def step_decay_schedule(self, initial_lr, decay_factor=0.5, step_size=1):
        '''
        Wrapper function to create a LearningRateScheduler with step decay schedule.
        '''

        def schedule(epoch):
            new_lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))

            return new_lr

        return LearningRateScheduler(schedule)

    def train(self, x_train, epochs, batch_size, x_val):
        self.model.fit(
            x_train
            , x_train
            , batch_size=batch_size
            , validation_data=(x_val, x_val)
            , verbose=True
            , shuffle=True
            , epochs=epochs
            # , callbacks=callbacks_list
        )
    
    def train_with_generator(self, data_flow, epochs, steps_per_epoch, data_flow_dev, run_folders):
        # import pdb; pdb.set_trace()
        csv_logger = CSVLogger(run_folders["log_filename"])
        checkpoint_filepath = os.path.join(run_folders["model_path"], "weights/weights-{epoch:03d}-{loss:.5f}.h5")
        # checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only=True, save_best_only=True, verbose=1)
        
        checkpoint1 = ModelCheckpoint(os.path.join(run_folders["model_path"], run_folders["exp_name"]+'/weights/VAE.weights.h5'), save_weights_only=True, verbose=1)
        lr_sched = self.step_decay_schedule(initial_lr=self.learning_rate, decay_factor=1, step_size=1)
        # early_stop = EarlyStopping(monitor="val_loss", patience=25)
        # callbacks_list = [checkpoint1, csv_logger, early_stop, lr_sched]
        callbacks_list = [checkpoint1, csv_logger, lr_sched]
        # import pdb; pdb.set_trace()
        #callbacks_list = [checkpoint1, csv_logger]
        history = self.model.fit(data_flow,
                                 epochs=epochs,
                                 verbose=True,
                                 validation_data=data_flow_dev,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=callbacks_list
                                 )
        print('FINISHED TRAINING')
        self.save_model(run_folders, history)

    def save_model(self, run_folders, history):

        with open(os.path.join(run_folders["model_path"], run_folders["exp_name"]+'/VAE_model.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim
                , self.encoder_conv_filters
                , self.encoder_conv_kernel_size
                , self.encoder_conv_strides
                , self.bottle_dim
                , self.bottle_conv_filters
                , self.bottle_conv_kernel_size
                , self.bottle_conv_strides
                , self.decoder_conv_t_filters
                , self.decoder_conv_t_kernel_size
                , self.decoder_conv_t_strides
                , self.z_dim
                
            ], f)

        #self.plot_model(run_folders)
        learning_curve_plot(history, run_folders)

    
