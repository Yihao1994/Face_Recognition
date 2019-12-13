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
database["Yihao"]  = img_to_encoding("images/Database/Yihao.jpg", FRmodel)




###############################################################################
#Face vertification
def verify(image_path, identity, database, model, threshold):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras
    
    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    
    ### START CODE HERE ###
    
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(database[identity] - encoding)
    
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < threshold:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
        print('How far is the distance:', dist)
    else:
        print("It's not " + str(identity) + ", walk away")
        door_open = False
        print('How far is the distance:', dist)
        
    ### END CODE HERE ###
        
    return dist, door_open


threshold = 0.7
image_path = "images/FR_test/phone_camera_1.jpg"
verify(image_path, "Yihao", database, FRmodel, threshold)

