# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:20:54 2022

@author: Theo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:02:04 2022

@author: Faiyza
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'

# For pre-processing 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize

# For the model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU, Cropping2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate, Dropout, MaxPool2D, Concatenate, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from IPython.core.display import display

# train test split
from sklearn.model_selection import train_test_split


'''SCRIPT INPUTS'''
# Note, the data used in this script comes from the "Data Science Bowl 2018" on
# Kaggle: https://www.kaggle.com/competitions/data-science-bowl-2018
# To run this script, go to the link above and download the data set. Make sure
# to update the train_path and test_path variables below to match your directories.

# Define the paths (folders) for training and test
# TODO: Update these paths 
train_path = 'C:/Users/Theo/Documents/Course Work/MBP1413 Biomedical Applications of AI/data-science-bowl-2018/stage1_train/'
test_path = 'C:/Users/Theo/Documents/Course Work/MBP1413 Biomedical Applications of AI/data-science-bowl-2018/stage1_test/'

img_size = (256, 256)
batch_size = 20;
epochs = 60;
steps_per_epoch = 200
data_augmentation = True


'''DATA LOADING AND PREPROCESSING'''

def load_dataset(fpath, img_size):
    # Load a data set of images and corresponding masks from data-science-bowl-2018 dataset
    # Inputs:
        # fpath (string)
        # img_size (tuple)
        
    #get ids of images in data set
    data_ids = next(os.walk(fpath))[1]
    
    data = {}
    for ids in data_ids:
        #images
        img_path = fpath + ids + '\\images\\'
        img = cv2.imread(img_path + ids + '.png')
        img = cv2.resize(img, img_size);
        data[ids] = [img]
        
        #Masks
        masks = np.zeros(img_size + (1,), dtype=bool)
        mask_path = fpath + ids + '\\masks\\'
        for mask_index, filename in enumerate(os.listdir(mask_path)):
            mask = cv2.imread(os.path.join(mask_path, filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            mask = mask > 0
            mask = np.expand_dims(mask,2)
            masks = np.maximum(masks, mask)
        data[ids].append(masks.astype(int)) 
    
    # Split into image and mask variables
    images = []
    masks = []
    for ids in data_ids:
        temp = data[ids]
        images.append(temp[0])
        masks.append(temp[1])
        
    return images, masks

# Load data and split into train and test sets
image_set, mask_set = load_dataset(train_path, img_size)
train_images, test_images, train_masks, test_masks = train_test_split(image_set, mask_set, test_size=0.1)

'''DEFINE MODEL ARCHITECTURE'''
def unet_model2(Image_Size):
    input1 = Input(Image_Size)
    
    # activation method (select selu or relu)
    act_meth = 'selu'
    
    # CONTRACTING
    #1st Layer
    c1 = Conv2D(16, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(input1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPool2D()(c1) 
    
    #2nd Layer
    c2 = Conv2D(32, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPool2D()(c2)

    #3rd Layer
    c3 = Conv2D(64, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPool2D()(c3)
    
    #4th Layer
    c4 = Conv2D(128, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPool2D()(c4) 
    
    #5th Layer
    c5 = Conv2D(256, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(256, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(c5)
    
    # EXPANDING
    #6th Layer
    u6 = Conv2DTranspose(128, (2, 2), strides=(2,2), padding='same')(c5)
    u6 = Concatenate()([u6, c4]) 
    c6 = Conv2D(128, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(c6)
    
    #7th Layer
    u7 = Conv2DTranspose(64, (2, 2), strides=(2,2), padding='same')(c6)
    u7 = Concatenate()([u7, c3]) 
    c7 = Conv2D(64, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(c7)
    
    #8th Layer
    u8 = Conv2DTranspose(32, (2, 2), strides=(2,2), padding='same')(c7)
    u8 = Concatenate()([u8, c2]) 
    c8 = Conv2D(32, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(c8)
    
    #9th Layer
    u9 = Conv2DTranspose(16, (2, 2), strides=(2,2), padding='same')(c8)
    u9 = Concatenate()([u9, c1]) 
    c9 = Conv2D(16, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation=act_meth, kernel_initializer='he_normal', padding='same')(c9)
    
    # Output
    outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)
    
    model = Model(input1, outputs)
    return model

def plot_model(model):
    # Plot model graph. Unused in script, but callable in console.
    # Requires: from IPython.core.display import display
    tf.keras.utils.plot_model(model, 'test_model.png', show_shapes=True)
    display(model)
    return


'''DICE COEFFICIENT LOSS FUNCTION'''
def dice_coeff(y_true, y_pred):
    smooth = 1
    
    intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * y_pred)
    score = (2. * intersection + smooth) / (tf.reduce_sum(tf.cast(y_true, tf.float32)) + tf.reduce_sum(y_pred) + smooth)
    
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

# Define combo loss function as sum of cross-entropy loss and dice loss
def bce_dice_loss(y_true, y_pred):
    loss = tf.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

'''DATA AUGMENTATION'''
# Format input images as float32 between 0 and 1
train_images = (np.array(train_images)/255).astype('float32')

# Cast boolean training masks as integers
train_masks = np.array(train_masks, dtype=int)
train_masks = np.uint8(train_masks)

# Repeat reformatting for test/validation set
test_images = (np.array(test_images)/255).astype('float32')
test_masks = np.array(test_masks, dtype=int)
test_masks = np.uint8(test_masks)

# Generate random seed for data augmentation. Same seed must be used in training
# image augmentation and training mask augmentation to ensure masks still 
# correspond with augmented images.
data_augmenter_seed = np.random.randint(0, 255) # generate random integer seed

# Define parameters for data augmentation.
data_augmenter_args = dict(rotation_range=90, vertical_flip=True,
                           horizontal_flip=True, width_shift_range=0.1,
                           height_shift_range=0.1, zoom_range=0.1)

# Create ImageDataGenerator objects for images and masks.
image_data_augmenter = tf.keras.preprocessing.image.ImageDataGenerator(**data_augmenter_args)
mask_data_augmenter = tf.keras.preprocessing.image.ImageDataGenerator(**data_augmenter_args)

# Fit objects to data using the same random seed.
image_data_augmenter.fit(train_images, seed=data_augmenter_seed)
mask_data_augmenter.fit(train_masks, seed=data_augmenter_seed)

# Use ImageDataGenerator object to create numpy iterable for generating augmented images and masks.
image_generator = image_data_augmenter.flow(train_images, batch_size=batch_size, seed=data_augmenter_seed)
mask_generator = mask_data_augmenter.flow(train_masks, batch_size=batch_size, seed=data_augmenter_seed)

# Zip numpy iterables together to ensure compatibility with model.fit() method.
train_generator = zip(image_generator, mask_generator)


'''MODEL FITTING'''
learning_rate = 0.001

# Modify MeanIoU class to accept probabilistic inputs. By default MeanIoU takes binary inputs. 
class MyMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred > 0.5, sample_weight)

# Compile model
input_shape = img_size + (3,)
model = unet_model2(input_shape)
model.compile(optimizer = 'adam', loss=bce_dice_loss, metrics=['accuracy', MyMeanIoU(2, name='my_mean_io_u_1')])

model.summary()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Fit model using either augmented or non-augmented data controlled by data_augmentation boolean.
if data_augmentation:
    history = model.fit(train_generator, batch_size=batch_size, epochs=epochs, 
                        validation_data=(test_images, test_masks),
                        steps_per_epoch=steps_per_epoch)
else:
    history = model.fit(train_images, train_masks, batch_size=batch_size,
                        epochs=epochs, validation_data=(test_images, test_masks))

# Save the model weights
model.save_weights("./model_weights.h5")

'''PRESENT RESULTS'''

# Plot pixel accuracy vs epoch
fig1 = plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
fig1.savefig('Accuracy_Faiyza.png')

# Plot loss function value vs epoch
fig2 = plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
fig1.savefig('Loss_Faiyza.png')

# Plot mean IoU vs epoch
fig3 = plt.figure()
plt.plot(history.history[list(history.history.keys())[2]], label='Training Mean IoU')
plt.plot(history.history[list(history.history.keys())[5]], label='Validation Mean IoU')
plt.legend()
plt.grid()
plt.title('Model Mean IoU')
plt.ylabel('Mean IoU')
plt.xlabel('Epochs')
fig1.savefig('Mean_IoU_Faiyza.png')

# Sanity check show image, true mask, and predicted mask
fig5, ax5 = plt.subplots(1,3)
sample_index = 50 # Number unimportant. Chooses image randomly because of shuffling of train and val set.
ax5[0].imshow(train_images[sample_index])
ax5[0].axis('off')
ax5[0].set_title('Sample Image')

ax5[1].imshow(train_masks[sample_index])
ax5[1].axis('off')
ax5[1].set_title('Ground Truth')

sample_mask = model.predict(np.expand_dims(train_images[sample_index], 0))
sample_mask_bool = sample_mask > 0.5
ax5[2].imshow(np.squeeze(sample_mask_bool))
ax5[2].axis('off')
ax5[2].set_title('Prediction')
plt.tight_layout()


''' EVALUATE MODEL'''


test_scores = model.evaluate(test_images, test_masks, verbose=2)


'''GENERATE FIGURES FOR REPORT'''
# Plot set of 8 augmented images and the original for report.
image_index = 65
sample_augmenter = tf.keras.preprocessing.image.ImageDataGenerator(**data_augmenter_args)
#image_data_augmenter.fit(train_images, seed=data_augmenter_seed)
sample_image_generator = image_data_augmenter.flow(np.expand_dims(train_images[image_index],0), 
                                            batch_size=8, seed=data_augmenter_seed)


fig4, ax4 = plt.subplots(3,3)
for i in range(3):
    for j in range(3):
        if i == 0 and j == 0:
            ax4[i,j].imshow(train_images[image_index])
            ax4[i,j].axis('off')
            ax4[i,j].set_title('Original Image')
        else:
            image = next(sample_image_generator)[0]
            ax4[i,j].imshow(image)
            ax4[i,j].axis('off')
plt.tight_layout()


# Plot prediction
image_index = 32 # Not random
fig6, ax6 = plt.subplots(3,1)
ax6[0].imshow(image_set[image_index])
ax6[0].axis('off')
ax6[0].set_title('Input Image')

ax6[1].imshow(mask_set[image_index])
ax6[1].axis('off')
ax6[1].set_title('Ground Truth')

sample_mask_2 = model.predict(np.expand_dims(image_set[image_index], 0))
sample_mask_bool_2 = sample_mask > 0.5
ax6[2].imshow(mask_set[image_index])
ax6[2].axis('off')
ax6[2].set_title('Prediction')
plt.tight_layout()