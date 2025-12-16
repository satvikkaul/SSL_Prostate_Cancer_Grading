####################################################
#Imports
####################################################

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.transform import resize
from PIL import Image
from data_augmentation import translateit_fast_2d as translateit_fast
from data_augmentation import scaleit_2d as scaleit
from data_augmentation import rotateit_2d as rotateit
from data_augmentation import intensifyit_2d as intensifyit
from sklearn.utils.class_weight import compute_class_weight
import cv2
from utils_huleo import get_random_patch_list, random_hide, image_histogram_equalization, hist_match

####################################################
#classes
####################################################

class DataGenerator(tf.keras.utils.Sequence):
    
    ''' Initialization of the generator '''
    def __init__(self, data_frame, y, x, target_channels, y_cols = None, indexes_output=None, batch_size=128, path_to_img="./dataset/images", shuffle=True, vae_mode=False, data_augmentation=True, reconstruction=False, softmax=False, hide_and_seek=False, equalization=False, mode='mclass', outputs = [True, True, True, True], hist_matching = False,
                 dict_classes=1):

        # Initialization

        if dict_classes==1:
            self.dict_classes = {
                "C": np.array([1, 0, 0, 0]),
                "N": np.array([0, 1, 0, 0]),
                "I": np.array([0, 0, 1, 0]),
                "NI": np.array([0, 0, 0, 1])
            }
        if dict_classes==2:
            self.dict_classes = {
                "C": np.array([1, 0, 0 ]),
                "NV": np.array([0, 1, 0]),
                "NB": np.array([0, 0, 1])
            }

        self.y_cols = y_cols
            
        # Tsv data table
        self.df = data_frame
        # Image Y size
        self.y = y
        # Image X size
        self.x = x
        # Channel size
        self.target_channels = target_channels
        # batch size
        # import pdb; pdb.set_trace()
        self.batch_size = batch_size
        # Boolean that allows shuffling the data at the end of each epoch
        self.shuffle = shuffle
        # Boolean that allows data augmentation to be applied
        self.data_augmentation = data_augmentation
        # Array de posiciones creada a partir de los elementos de la tabla
        self.indexes = np.arange(len(data_frame.index))
        # Array of positions created from the elements of the table
        self.path_to_img = path_to_img
        # VAE mode
        self.vae_mode = vae_mode
        # Tests
        self.hideAndSeek = hide_and_seek
        self.equalization = equalization
        self.outputs = np.array(outputs)
        self.mode = mode
        self.hist_matching = hist_matching
        self.im_ref = None
        

    def __len__(self):
        ''' Returns the number of batches per epoch '''
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        ''' Returns a batch of data (the batches are indexed) '''
        # Take the id's of the batch number "index"
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Batch initialization
        X, Y = [], []

        # For each index,the sample and the label is taken. Then the batch is appended
        for idx in indexes:
            # Image and idx index tag is get
            x, y = self.get_sample(idx)
            # This image to the batch is added
            X.append(x)
            Y.append(y)
	# The created batch is returned
        return np.array(X), np.array(Y) #X:(batch_size, y, x), y:(batch_size, n_labels_types)

    def on_epoch_end(self):
        ''' Triggered at the end of each epoch '''
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # Shuffles the data

    # Roberto Paredes contribution @RParedesPalacios

    def get_sample(self, idx):
        # import pdb; pdb.set_trace()
        '''Returns the sample and the label with the id passed as a parameter'''
        # Get the row from the dataframe corresponding to the index "idx"
        df_row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.path_to_img, df_row["image_name"]))
        da =  np.asarray(image).shape
        #image.thumbnail((self.x,self.x), Image.ANTIALIAS)
        image = image.resize((self.x, self.y))
        image = np.asarray(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.reshape(image, (len(image)))
        image = image.astype("float32") / 255.0

        if self.mode == 'mClass':
            label = self.dict_classes[df_row["group"]][self.outputs]
        if self.mode == 'mLabel':
            label = np.array(list(df_row.values[-2:]), dtype=np.int)
        
        #Extract the specified columns as labels
        if self.y_cols is not None:
            label = np.array(df_row[self.y_cols].values, dtype=np.float32)

        # image_resampled = np.reshape(image,image.shape + (self.target_channels,))
        image_resampled = np.reshape(image,image.shape)
        img2 = np.array(image_resampled)

        img2.setflags(write=1)
        
        if self.equalization:
            img2 = image_histogram_equalization(img2, number_bins=256)

        if self.hist_matching:
            img2 = hist_match(img2, self.im_ref)

        if self.hideAndSeek:
            img_avg = np.average(img2)
            patch_list = get_random_patch_list(self.x, 16)
            img2 = random_hide(img2, patch_list, hide_prob=0.5, mean=img_avg)

        # Data aumentation **always** if True
        # import pdb; pdb.set_trace()
        if self.data_augmentation:
            do_rotation = True
            do_shift = True
            do_zoom = True
            do_intense= False

            theta1 = float(np.around(np.random.uniform(-5.0,5.0, size=1), 3))
            offset = list(np.random.randint(-10,10, size=2))
            zoom  = float(np.around(np.random.uniform(0.9, 1.05, size=1), 2))
            factor = float(np.around(np.random.uniform(0.8, 1.2, size=1), 2))

            if do_rotation:
                rotateit(img2, theta1)
            if do_shift:
               translateit_fast(img2, offset)


            if do_zoom:
                for channel in range(self.target_channels):
                    img2[:,...,channel] = scaleit(img2[:,...,channel], zoom)
            if do_intense:
                img2[:,...,0]=intensifyit(img2[:,...,0], factor)

        #### DA ends

        img2 = self.norm(img2)
        if self.vae_mode:
            label = img2
        # Return the resized image and the label
        return img2, label

    def norm(self, image):
        image = image / 255.0
        return image.astype( np.float32 )

    def compute_class_weights(self, classes=['C', 'N', 'I', 'NI']):

        w = compute_class_weight('balanced', classes, np.asarray(self.df['group']))

        w_dict = {}
        for i in range(0, len(classes)):
            w_dict[i] = w[i]
            
        return w_dict
    