####################################################
#Imports
####################################################

import numpy as np
import os
import tensorflow as tf

import cv2
from data_augmentation import translateit_fast_2d as translateit_fast
from data_augmentation import scaleit_2d as scaleit
from data_augmentation import rotateit_2d as rotateit
from data_augmentation import intensifyit_2d as intensifyit
from sklearn.utils.class_weight import compute_class_weight
from utils_huleo import get_random_patch_list, random_hide, image_histogram_equalization, hist_match

####################################################
#classes
####################################################

class DataGenerator(tf.keras.utils.Sequence):  # type: ignore
    
    ''' Initialization of the generator '''
    def __init__(self, data_frame, y, x, target_channels, y_cols = None, indexes_output=None, batch_size=128, path_to_img="./dataset/images", shuffle=True, vae_mode=False, data_augmentation=True, reconstruction=False, softmax=False, hide_and_seek=False, equalization=False, mode='mclass', outputs = [True, True, True, True], hist_matching = False,
                 dict_classes=1):
        super().__init__()
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
        if not os.path.exists(os.path.join(self.path_to_img, df_row["image_name"])):
             # Fallback or error handling could go here. 
             # For now, strict fail is fine or let cv2 handle it (it returns None).
             pass

        # OpenCV Load (Faster)
        image = cv2.imread(os.path.join(self.path_to_img, df_row["image_name"]))
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore
        
        # Resize
        image = cv2.resize(image, (self.x, self.y))
        
        # REMOVED: Double Normalization Bug
        # image = image.astype("float32") / 255.0

        if self.mode == 'mClass':
            label = self.dict_classes[df_row["group"]][self.outputs]
        if self.mode == 'mLabel':
            label = np.array(list(df_row.values[-2:]), dtype=np.int32)
        
        #Extract the specified columns as labels
        if self.y_cols is not None:
            label = np.array(df_row[self.y_cols].values, dtype=np.float32)

        # image_resampled = np.reshape(image,image.shape + (self.target_channels,))
        image_resampled = np.reshape(image,image.shape)
        img2 = np.array(image_resampled)

        img2.setflags(write=True)

        # Normalize to [0,1] BEFORE augmentation to prevent quantization artifacts
        img2 = self.norm(img2)
        
        if self.equalization:
            img2 = image_histogram_equalization(img2, number_bins=256)

        if self.hist_matching:
            img2 = hist_match(img2, self.im_ref)

        if self.hideAndSeek:
            img_avg = int(np.average(img2))
            patch_list = get_random_patch_list(self.x, 16)
            img2 = random_hide(img2, patch_list, hide_prob=0.5, mean=img_avg)

        # Data aumentation **always** if True
        if self.data_augmentation:
            do_rotation = True
            do_shift = True
            do_zoom = True
            # do_intense= False # Disabled

            theta1 = float(np.around(np.random.uniform(-5.0,5.0, size=1), 3))
            offset = list(np.random.randint(-10,10, size=2))
            zoom  = float(np.around(np.random.uniform(0.9, 1.05, size=1), 2))
            # factor = float(np.around(np.random.uniform(0.8, 1.2, size=1), 2))

            if do_rotation:
                img2 = rotateit(img2, theta1)
            
            if do_shift:
               img2 = translateit_fast(img2, offset)

            if do_zoom:
                # Apply zoom to the entire 3-channel image at once (Faster & Safer)
                img2 = scaleit(img2, zoom)
            
            # if do_intense:
            #     img2[:,...,0]=intensifyit(img2[:,...,0], factor)

        #### DA ends
        # Normalization now happens BEFORE augmentation (line 141), removed duplicate
        if self.vae_mode:
            label = img2

        # Return the resized image and the label
        return img2, label

    def norm(self, image):
        image = image / 255.0
        return image.astype( np.float32 )

    def compute_class_weights(self, classes=['C', 'N', 'I', 'NI']):

        w = compute_class_weight('balanced', classes=classes, y=np.asarray(self.df['group']))

        w_dict = {}
        for i in range(0, len(classes)):
            w_dict[i] = w[i]
            
        return w_dict
    
def create_tf_dataset(generator):
    """
    Wraps a DataGenerator instance into a tf.data.Dataset.
    Provides parallel data loading with automatic CPU optimization.
    
    Args:
        generator: DataGenerator instance
        
    Returns:
        tf.data.Dataset: Optimized dataset with prefetching and parallel loading
    """
    # Dynamically get label shape from first sample
    sample_idx = 0
    _, sample_label = generator.get_sample(sample_idx)
    label_shape = sample_label.shape

    def generator_func(index):
        # Internal get_sample logic handles reading and augmenting
        sample, label = generator.get_sample(int(index))
        return sample, label
    
    def tf_wrapper(index):
        # Use tf.py_function to call python logic
        result = tf.py_function(
            func=generator_func, 
            inp=[index], 
            Tout=[tf.float32, tf.float32]
        )
        # Safely unpack result
        sample, label = result[0], result[1] # type: ignore
        
        # Explicitly set shape since py_function loses it
        
        sample.set_shape((generator.y, generator.x, generator.target_channels))
        label.set_shape(label_shape)
        return (sample, label)
    
    # 1. Create dataset of indices [0, 1, 2, ... N-1]
    dataset = tf.data.Dataset.range(len(generator.indexes))
    
    # 2. Shuffle globally (if enabled in generator)
    if generator.shuffle:
        dataset = dataset.shuffle(buffer_size=len(generator.indexes))
    
    # 3. Map indices to images (Parallelized across CPU cores!)
    dataset = dataset.map(
        tf_wrapper, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # 4. Batch and Prefetch 
    dataset = dataset.batch(generator.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset