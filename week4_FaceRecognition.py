# In[Model Loading]
import keras
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *


# In[]
###############################################################################
#Triplet loss, which is the FR algorithm wanna minimize.
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    
    return loss
###############################################################################

#Load the model with the well-trained data
num_px = 96       #Pixel size
FRmodel = faceRecoModel(input_shape = (3,num_px,num_px))        #Initialize the tensor shape
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)


#Database loading
database = {}
database["danielle"] = img_to_encoding("images/Database/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/Database/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/Database/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/Database/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/Database/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/Database/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/Database/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/Database/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/Database/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/Database/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/Database/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/Database/arnaud.jpg", FRmodel)

database["Yihao_1"]  = img_to_encoding("images/Database/Yihao_1.jpg", FRmodel)
database["Yihao_2"]  = img_to_encoding("images/Database/Yihao_2.jpg", FRmodel)
database["Yihao_3"]  = img_to_encoding("images/Database/Yihao_3.jpg", FRmodel)
database["Yihao_4"]  = img_to_encoding("images/Database/Yihao_4.jpg", FRmodel)

database["Binhe_1"]  = img_to_encoding("images/Database/Binhe_1.jpg", FRmodel)
database["Binhe_2"]  = img_to_encoding("images/Database/Binhe_2.jpg", FRmodel)
database["Binhe_3"]  = img_to_encoding("images/Database/Binhe_3.jpg", FRmodel)
database["Binhe_4"]  = img_to_encoding("images/Database/Binhe_4.jpg", FRmodel)


# In[Face Recognition Implementation]
def FR_implement(image_path, database, model, threshold, num_px):
    """
    Implements face recognition for the office by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding()
    encoding = test_img_to_encoding(image_path, model, num_px)
    
    ## Step 2: Find the closest encoding ##
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        # Compute L2 distance between the target "encoding" and the current db_enc from the database.
        dist = np.linalg.norm(db_enc - encoding)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. 
        if dist< min_dist:
            min_dist = dist
            identity = name

    if min_dist >= threshold:
        print("Provided image is not in the database.")
    else:
        if identity == "Yihao_1" or identity == "Yihao_2" or identity == "Yihao_3":
            identity = "Yihao"
            print ("This is " + str(identity) + ", the distance is " + str(min_dist))
            
        elif identity == "Binhe_1" or identity == "Binhe_2" or \
        identity == "Binhe_3" or identity == "Binhe_4":
            identity = "Binhe"
            print ("This is " + str(identity) + ", the distance is " + str(min_dist))
            
        else:
            print ("This is " + str(identity) + ", the distance is " + str(min_dist))
    
    return min_dist, identity

###############################################################################
image_path = "images/FR_test/phone_camera_1.jpg"
threshold = 0.46
FR_implement(image_path, database, FRmodel, threshold, num_px)
