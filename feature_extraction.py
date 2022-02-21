# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 08:43:42 2022

@author: ednaw
"""
import pandas as pd
from skimage.feature import hog
import numpy as np


#=================================
ALGORITHM = "XGB" #RF
#=================
#XGB
#=================
## HyperParameter
# MAX_DEPTH improved followed by MIN_CHILD_WEIGHT and GAMMA
ETA = 0.08575668816909966              # 0.01 0.05, 0.1
MAX_DEPTH = 11           # 3,5,7,9,12,15,17,25
SUBSAMPLE = 0.5756891162217281        # 0.6, 0.7, 0.8, 0.9, 1.0
MIN_CHILD_WEIGHT = 2    # 1,3,5,7
COLSAMPLE_BYTREE = 0.6047004091120212    # 0.6,0.7,0.8, 0.9,1.0
GAMMA = 0               # 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0
ALPHA = 0
LAMDA = 1
N_ESTIMATOR = 100

#=================
#RF
#=================
#Hyperparameter
N_ESTIMATORS = 500         #120, 300, 500, 800, 1200
MAX_DEPTH = 10              # 5, 8, 15, 25, 30, None
MIN_SAMPLES_SPLIT = 100       # 2, 5, 10, 15, 100
MIN_SAMPLES_LEAF = 2       # 1, 2, 5, 10
#=================================

#=================================
FEATURES = "combined" #LBP, combined
#=================
#HOG
#=================
ORIENTATION = 9
SIZE_HOG = 24 #24
PIXEL = 4#4 #8
CELL = 2
#=================
#LBP
#=================
NEIGHBOURS_LBP = 240#240 #128
RADIUS_LBP = 2 #2
#=================

DATA = "MSRA" #"MSRAtest"
HORIZONTAL = False
#=================================
TREAT = "UNBALANCE" 
SIZE = "long" #"square"
TRESH_IOU = 0.5
#=================================

#=================================
PATH1 = "MSRA_train_prop.pickle"#"MSRA_train_horizontal_prop.pickle"#"MSRA_train_prop.pickle"
#PATH1 = "MSRA_test_prop.pickle"#"MSRA_test_horizontal_prop.pickle"#
#=================================


#===============================================================================
#HOG
def features_hog(df, pixel = PIXEL, cell = CELL, horizontal = False):
    # get resized image from filtered dataframe 
    if horizontal :
        cc = list(df["horizontal_resized"])
    else:
        cc = list(df["resized"])
    
    # Read from dataframe
    features = []
    for img in cc:
        fd, _ = hog(img, orientations = 9, pixels_per_cell = (PIXEL,PIXEL), 
                    cells_per_block = (CELL,CELL), visualize = True)
        features.append(fd)
        
    # convert into array 2D    
    features = np.array(features)
    # convert label
    label = np.array([list(df["iou_score"])])
    label = np.transpose(label)
    # stack horizontal
    all_features = np.hstack((features,label))
    return all_features
    

#===============================================================================
from skimage import feature

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist


def features_lbp(df, neighbours = NEIGHBOURS_LBP, radius = RADIUS_LBP, 
                 horizontal = False):
    
    # get resized image from filtered dataframe 
    if horizontal :
        cc = list(df["horizontal_resized"])
    else:
        cc = list(df["resized"])
    # Read from dataframe
    #cc = list(df["normalized_cc"])
    
    features = [] # initiate features
    # define object of LBP
    desc = LocalBinaryPatterns(neighbours, radius)
    for img in cc:
        fd = desc.describe(img)
        features.append(fd)
        
    features = np.array(features)
    # convert label
    label = np.array([list(df["iou_score"])])
    label = np.transpose(label)
    # stack horizontal
    all_features = np.hstack((features,label))
    return all_features


#===============================================================================
def combined_hog_lbp(hog,lbp):
    total = np.hstack((hog[:,:-1],lbp))
    return total


#===============================================================================
def process_data(data):
    # Features
    x_train = (data[:,:-1])
    # Label (IOU)
    y_train = (data[:,-1]>=TRESH_IOU)*1
    return x_train, y_train


#===============================================================================
def features_extraction(size = SIZE, horizontal = HORIZONTAL):
    
    # TODO 1: Import file training data in pickle
    dframe = pd.read_pickle(PATH1)
    random_df = dframe.sample(frac = 1, random_state = 2)
    
    
    # TODO 2: Split image based on "square" and "long" and get resized image
    if horizontal:
        df_filter = random_df[random_df["size_horizontal"] == size]
    else:
        df_filter = random_df[random_df["size_img"] == size]
         
    
    # TODO 3: Get resized image then extract features "HOG", "LBP", "Combined"
    if FEATURES == "HOG":
        all_feature = features_hog(df_filter, horizontal = HORIZONTAL)
        np.save(f"{SIZE}_{FEATURES}_{DATA}_{SIZE_HOG}_{PIXEL}.npy",all_feature) 
    elif FEATURES == "LBP":
        all_feature = features_lbp(df_filter, horizontal = HORIZONTAL)
        np.save(f"{SIZE}_{FEATURES}_{DATA}_{NEIGHBOURS_LBP}_{RADIUS_LBP}.npy",all_feature) 
    else:
        hog = features_hog(df_filter, horizontal = HORIZONTAL)
        lbp = features_lbp(df_filter, horizontal = HORIZONTAL)
        all_feature = combined_hog_lbp(hog, lbp) 
        np.save(f"{SIZE}_{FEATURES}_{DATA}_{SIZE_HOG}_{PIXEL}_{NEIGHBOURS_LBP}_{RADIUS_LBP}.npy",all_feature)    
    # TODO 4 : Convert iou into class text and nontext
    #x_train, y_train = process_data(all_feature)
    
    # TODO 5 : Get model using specific algorithm and doing grid search for 
    #           optimization hyperparameter
    
    # TODO 6: Save model
    
    return all_feature


#===============================================================================
#f = features_extraction()
 