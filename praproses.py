# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 19:00:05 2022

@author: ednaw
"""

import os 
import cv2
from matplotlib import pyplot as plt
import pandas as pd

NEW_WIDTH  = 480
T_VARIATION = 0.3
T_AREA = 400
T_HEIGHT = 10 
T_WIDTH = 10
T_FILLING = 0.1
T_STD_VALUE = 0.6


# ================================================================
# TODO 1 : Read file, resize, preprocessing
# ================================================================
def list_file(path, type_file=".jpg"):
    """
    Call all name of file in one folder with specific type of file, 
    put all name of file into one

    Parameters
    ----------
    path : string
        directory of image files.
    type_file : string, optional
        extention of files. The default is ".jpg".

    Returns
    -------
    fname : list
        the collection of file names in 1 folder
    """
        
    fname=[]
    for file in os.listdir(path):
        if file.endswith(type_file):
            fname.append(file)
    return fname



# resizing one image and convert bgr to rgb
def resize_img(img, type_color="BGR"):
    """
    Resize width of image into 480

    Parameters
    ----------
    img : array numpy
        one image
    type_color : string, optional
        Type of image space. The default is "BGR".

    Returns
    -------
    image : array numpy
        one image
    """
    
    new_height = NEW_WIDTH * img.shape[0] // img.shape[1] 
    dim = (NEW_WIDTH, new_height)
    img_resize = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
    if type_color=="BGR":
        image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    else:
        image = img_resize
    return image


# Preprocessing Hasil akhir list 6 image
def preprocessing(img):
    """
    Do a preprocessing: resize, grayscale, hsv, convert negative

    Parameters
    ----------
    img : array numpy
        one image.

    Returns
    -------
    dict
        Dictionary consist of key channel and image of each channel.
    """
    
    # Resize dan ubah ke RGB format
    img_resize = resize_img(img)
        
    # RGB ke Gray 
    gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        
    # Convert ke HSV
    hsv = cv2.cvtColor(img_resize, cv2.COLOR_RGB2HSV)
        
    # H, S channels
    H = hsv[:,:,0]
    S = hsv[:,:,1]
        
    # Negative Gray, H, S
    neg_gray = 255 - gray
    neg_H = 255 - H
    neg_S = 255 - S
        
    # list image each channel
    im=[gray, neg_gray, H, neg_H, S, neg_S]   
    # Name key of each channel
    name =["gray", "neg_gray", "H","neg_H", "S", "neg_S"]
    
    return dict(zip(name, im))


# ================================================================
# TODO 2 : Plotting image with function show_image (1 image, 
#           list image, dic image)
# ================================================================
# each image 
def plot_image(image):
    """
    Plot one image in RGB or grayscale 

    Parameters
    ----------
    image : array numpy
        One image

    Returns
    -------
    None
    """
    
    if len(image.shape)==3:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()       


# show image from list
def show_image(img):
    """
    Plotting image either one image, list or dictionary of image
    
    Parameters
    ----------
    img : array numpy or list or dictionary of image
        one image or dictionary 

    Returns
    -------
    image plotting : array numpy
    """
    
    if type(img) is dict:
        for image in img.values():
            plot_image(image)
    elif type(img) is list:
        for image in img:
            plot_image(image)
    else:
        plot_image(img)     
        
        
# Plot Extreme Region
def plot_extreme(mxt, image):
    """
    Plotting bounding box of extreme region

    Parameters
    ----------
    mxt : object MaxTree
        max tree for one channel image
    image : 
        image rgb after resizing

    Returns
    -------
    image : 
        image rgb after drawing bounding box.
    """
    
    xmin=mxt.getAttributes("xmin").astype(int)
    xmax=mxt.getAttributes("xmax").astype(int)
    ymin=mxt.getAttributes("ymin").astype(int)
    ymax=mxt.getAttributes("ymax").astype(int)
    
    for i in range (len(xmin)):
        image = cv2.rectangle(image, (xmin[i],ymin[i]), (xmax[i],ymax[i]), 
                              (255,255,0), 1)
    return image


# Plotting based on rest of cc
def plotting_cc(rest_cc, image):
    """
    Plotting connected component into image RGB
    
    Parameters
    ----------
    rest_cc : list
        connected component attributes xmin, ymin, xmax, ymax
    image : array numpy
        resized image.

    Returns
    -------
    image : array numpy 
        image rgb after drawing bounding box.
    """
    
    for i in range(len(rest_cc)):
        image = cv2.rectangle(image, (rest_cc[i][0],rest_cc[i][2]), 
                              (rest_cc[i][1],rest_cc[i][3]), (255,255,0), 1)

    return image
   
     
# ================================================================
# TODO 3 : MaxTree and Filtering nodes
# ================================================================
from maxtree.component_tree import MaxTree
import numpy as np

def mxt(img):
    """
    Create maximum tree for each image from 1 file

    Parameters
    ----------
    img : array numpy
        one of image (Gray, Neg_Gray, H, Neg_H, S, Neg_S)

    Returns
    -------
    mt : object MaxTree
        component tree of one channel image 
    """
    
    mt = MaxTree(img.astype(np.uint16)) # compute the max tree
    mt.compute_shape_attributes() # compute shape attribute
    return mt


def filter_area(mxt, thres_area = T_VARIATION):
    """
    Create a dictionary consists of all filtered CC 

    Parameters
    ----------
    mxt : object MaxTree
        max tree of one channel of image.
    thres_area : float, optional
        treshold of variation value. The default is 0.5.

    Returns
    -------
    cc_selection : dictionary
        connected component atributes after area filtering
    """
    
    # Get the attribute "area" then we filter based on area threshold
    area = mxt.getAttributes("area")
    area_parent = np.zeros(len(area))
    list_parent = []
    for i in range(len(area)):
        index_parent = mxt.parent(i)
        area_parent[i] = area[index_parent]
        list_parent.append(index_parent)
    area_parent=np.exp(area_parent) # convert to linear
    area = np.exp(area)
    var_area=(area_parent-area)/area
    idx_retained = np.less_equal(var_area, thres_area).nonzero()[0]
    
    # Create a dictionary of CC after filtering
    # Key number of Connected component
    # Value xmin, xmax, ymin, ymax, area 
    xmin=mxt.getAttributes("xmin").astype(int)
    xmax=mxt.getAttributes("xmax").astype(int)
    ymin=mxt.getAttributes("ymin").astype(int)
    ymax=mxt.getAttributes("ymax").astype(int)
    area=np.exp(mxt.getAttributes("area")).astype(int) 
    cc_selection={}
    for index in idx_retained:
        cc_selection[index]=[xmin[index], xmax[index], ymin[index],
                             ymax[index], area[index], list_parent[index]]
    return cc_selection


def filter_geometric(dic,t_area = T_AREA, t_height = T_HEIGHT, 
                     t_width = T_WIDTH, t_fr = T_FILLING ):
    """
    Filtering maxtree based on geometric criteria
    
    Parameters
    ----------
    dic : TYPE
        DESCRIPTION.
    t_area : int, optional
        Threshold of area. The default is T_AREA.
    t_height : int, optional
        Threshold of height. The default is T_HEIGHT.
    t_width : int, optional
        Threshold of width. The default is T_WIDTH.
    t_fr : float, optional
        Threshold of filling rate. The default is T_FILLING.

    Returns
    -------
    dic_geometric : dict
        Dictionary of connected component and its attributes.
    """ 

    # get value of the dictionary and convert to list numpy
    list_dic_filter = list(dic.values())
    hasil=np.array(list_dic_filter)

    # list of keys of dictionary
    key_index= list(dic.keys())
    
    # Average
    width = (hasil[:,1]-hasil[:,0])+1
    height = (hasil[:,3]-hasil[:,2])+1
    area = hasil[:,4]
    rasioRR = area/(width*height)
     
    # Set Ruled untuk proses filter
    a=(height > 8*t_height) 
    b=(width > 8*t_width) 
    c=(height < 0.1*t_height) & (width < 0.1*t_width)
    d=(area > 8*t_area)
    e=(area < 0.04*t_area)
    f=(rasioRR > 6*t_fr)
    g=(rasioRR < 3*t_fr)
    
    # Set rule to keep
    #nodes = ~(a|b|c|d|e|f|g) 
    nodes = ~(a|b|c|d|e|f|g) 
    
    # dictionary result of geometric filter
    dic_geometric = {key_index[i]: list_dic_filter[i] for i in range(len(nodes)) 
                     if nodes[i] == True}
    return dic_geometric


# ================================================================
# TODO 4 : Prunning MaxTree with Standar Deviation 
#           and Node nearly to branch of tree
# ================================================================
def filtered_std(mxt, dic_cc, std_value = T_STD_VALUE):
    """
    Filter cc based on standar deviation of connected component
    filter image based on idc_retrained then 
    get 'average_0','std_0', 'max_val_0','min_val_0' then
    put as a dictionary
    
    Parameters
    ----------
    mxt : object MaxTree
        maxtree of one image.
    dic_cc : dict
        connected component properties.
    std_value : float, optional
        threshold of standar deviation value. The default is T_STD_VALUE.

    Returns
    -------
    cc_final : dict
        dictionary of connected component with their properties.
    img_filter : array numpy
        gray image after filtering.
    """
    
    # list of index retained
    index_retained = list(dic_cc.keys())
    # image after geometric filter
    filtered = mxt.filter(index_retained)
    
    # count average, std, max_val and min_val from filtered index of cc 
    mxt.compute_layer_attributes(filtered)
    new_attribute = mxt.getAttributes(['average_0','std_0', 'max_val_0','min_val_0'])
    
    # extend the dictionary with new attribute
    for key in dic_cc:
        dic_cc[key].extend(new_attribute[key])
    
    # create dic after selection based val[7] = "std_0"
    cc_final = {key:val for key, val in dic_cc.items() if val[7] >= std_value}
    
    # get list of retained index
    idx_retained = list(cc_final.keys())
    # get image after filter based on std 
    img_filter = mxt.filter(idx_retained)
    
    return cc_final, img_filter


def idx_subtree(dic):
    """
    Classified the cc based on subtree
    the output is list of subtree. Each subtree consist of one path

    Parameters
    ----------
    dic : dict 
        Dictionary of Maxtree from each image channel

    Returns
    -------
    tree_set : list
        cluster of index connected component number.

    """

    dic_new = {key: val[5] for key, val in dic.items()}
    index_list = list(dic_new.keys())
    parent_list = list(dic_new.values())
    
    tree_set=[]
    if len(index_list) == 0:
        return tree_set 
    else:
        subtree = [index_list[0]]

    for i in range(len(index_list)-1):
        if parent_list [i] == index_list[i+1]:
            subtree.append(index_list[i+1])
        else:
            tree_set.append(subtree)
            subtree=[index_list[i+1]]
    
    tree_set.append(subtree)         
    return tree_set


def cutting_tree (groups, dic_filter):
    """
    finding the maximum and create list of index, get the index value

    Parameters
    ----------
    groups : list 
        cluster of index cc
    dic_filter : dict
        connected component attributes

    Returns
    -------
    dict
        prunning result (the attributes of retained cc).
        index: [xmin, xmax, ymin, ymax,area, parents, mean, 
                sd, max value, min value]
    """
    
    retained_index = [max(one_group) for one_group in groups]
    return {key:dic_filter[key] for key in retained_index}

def prunning_cc(dic_SD):
    """
    Prunning for 1 file 6 channel image. Output img_filter and cutting image

    Parameters
    ----------
    dic_SD : dictionary
        Connected component properties such as location and final grayscale
        for 6 channel image
    
    Returns
    -------
    img_filter : dictionary
        6 gray scale
    cutting_img : dictionary
        Finale properties of connected component after filtering

    """
    img_filter = {} # grayscale image after filtering
    cutting_img = {} # cc properties final after filtering
    for key, value in dic_SD.items():
        SD_cc, img_filter[key] = value
        # group index dari 1 path yang sama lalu ambil index yang terakhir
        groups = idx_subtree(SD_cc)
        # perbaharuai dictionary dari tree
        cutting_img[key] = cutting_tree (groups, SD_cc) 
    return img_filter, cutting_img

#=================================================================
# Khusus filtering sekaligus
#=================================================================
def filter_cc(one_tree):
    dic_filter = filter_area(one_tree)

    # filter based on geometric features
    geo_filter = filter_geometric(dic_filter)
  
    final_filtered, img_final = filtered_std(one_tree, geo_filter)
    
    # group index dari 1 path yang sama lalu ambil index yang terakhir
    groups = idx_subtree(final_filtered)
    # perbaharuai dictionary dari tree
    cutting_dic = cutting_tree (groups, final_filtered) 
    return cutting_dic, img_final


# ================================================================
# TODO 5: Get binerized connected component image and 
# paste in black background. Put in one dictionary 
# ================================================================
def get_rotated_bb(feature, img_final, opt = "biner"):
    """
    Paste image in black background then bineraized based on 
    min grayscale value (feature 9) 

    Parameters
    ----------
    feature : dict
        index: [xmin, xmax, ymin, ymax,area, parents, mean, 
                sd, max value, min value]
    img_final : np.array
        image same size as size of resized rgb image

    Returns
    -------
    result : np,array
        binerized connected component.

    """
    # An example of first bounding box
    first_bb_points = [[feature[0], feature[2]], [feature[0], feature[3]],
                       [feature[1], feature[3]], [feature[1], feature[2]]]

    stencil = np.zeros(img_final.shape).astype(img_final.dtype)
    contours = [np.array(first_bb_points)]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)

    im_copy = img_final.copy()
    result1 = cv2.bitwise_and(im_copy, stencil)
    
    if opt == "biner":
        _, result = cv2.threshold(result1, int(feature[9])-1, 255, cv2.THRESH_BINARY)
    elif opt == "rgb":
        result = result1
    # keep result1

    return result


def dic_rotated_bb(dic_feature, img_final, option = "biner"):
    """
    Get rotated bbox coordinate and binary connected component

    Parameters
    ----------
    dic_feature : dict
        dictionary properties of cc from filtering result.
    img_final : np.array
        grayscale image after filtering.

    Returns
    -------
    dic_rotated : dict
        index: binerized image of CC the size same as resized rgb
        
    """
    dic_rotated = {}
    for key in dic_feature:
        feature = dic_feature[key]
        result = get_rotated_bb(feature, img_final.copy(), option)
        dic_rotated[key] = result
    return dic_rotated


def rotated_box_coordinate (feature, img_final):
    """
    Finding the four coordiates of rotated bbox then
    return into  (box, rectangle)

    Parameters
    ----------
    feature : 
        one connected component features
    img_final : np.array
        img_final after filtering each channel.

    Returns
    -------
    box : numpy array
        [x1,y1,x2,y2,x3,y3,x4,y4]
    rect : tuple
        ((x,y),(w,h),angle(in degree)).
    """

    result = get_rotated_bb(feature, img_final)
    active_px = np.argwhere(result!=0)
    # flipped to get coordinate
    a_px = active_px[:,[1,0]]
    # get rectangle (xc,yc),(w,h), degree
    rect =cv2.minAreaRect(a_px)
    # create 8 points of rectangle
    box=cv2.boxPoints(rect)
    # convert to numpay array
    box = np.int0(box) # box 
    return box, rect

def dic_rbox (dic, img_final):
    """
    Finding the four coordiates of rotated bbox then
    save it into indexed dictionary (one channel image)

    Parameters
    ----------
    dic : dict
        connected component features from one channel image
    img_final : numpy array
        one final of grayscale image

    Returns
    -------
    dict
        dictionary of box and rect.

    """
    dic_coordinate = {}
    dic_rotated = {}
    for key, f in dic.items():
        dic_coordinate[key], dic_rotated[key] = rotated_box_coordinate(f, img_final)
    return dic_coordinate, dic_rotated


#=======================================================================
# TODO 6: Normalized image and cropping horizontal image
#=======================================================================

def check_angle (angle):
    """
    Checking angle to determine the rotation of the image

    Parameters
    ----------
    angle : float
        angle of rotated bounding box.

    Returns
    -------
    float
        the new angle for normalization.
    """
    
    if angle > 45 :
        return 90 - angle
    elif angle < -45:
        return -(90 + angle)
    else: 
        return -angle  
    
    
from imutils import rotate_bound
def img_transform(dic_cc, dic_point, opt = "binary"):
    """
    Get image after transformation and put in dictionary

    Parameters
    ----------
    dic_cc : dic
        dictionary consist of binary image with black background and
        same size with the resized rgb
    dic_point : dic
        dictionary consist of (x,,y),(w,h),angle of connected component.

    Returns
    -------
    dic_transform : dic
        index: binary image after normalization. (before cropping)
    """

    dic_transform = {}
    for key in dic_cc:
        (_,_,angle) = dic_point[key]
        im = dic_cc[key].copy()
        angle = check_angle(angle)
        image = rotate_bound(im, angle)
        if opt == "binary" :
            # need to binerized the image after normalization
            _,image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        dic_transform[key] = image
    return dic_transform


# def transform_gray(dic_cc, dic_point):
#     """
#     Get image after transformation and put in dictionary

#     Parameters
#     ----------
#     dic_cc : dic
#         dictionary consist of binary image with black background and
#         same size with the resized rgb
#     dic_point : dic
#         dictionary consist of (x,,y),(w,h),angle of connected component.

#     Returns
#     -------
#     dic_transform : dic
#         index: binary image after normalization. (before cropping)
#     """

#     dic_transform = {}
#     for key in dic_cc:
#         (_,_,angle) = dic_point[key]
#         im = dic_cc[key].copy()
#         angle = check_angle(angle)
#         image = rotate_bound(im, angle)
#         dic_transform[key] = image
#     return dic_transform


def get_cc(dic_img):
    """
    finding contour to create rectangle then do a cropping,
    save the cropping image in dictionary cc

    Parameters
    ----------
    dic_img : dict
        index : image (numpy).

    Returns
    -------
    dic_bin : dict
        index : cropping image.

    """
    # 1 image neg, list cc yang telah difilter
    # cropped dan biner
    # normalized
 
    dic_bin = {}
    for key, image in dic_img.items():
        # convert image into int8
        image = cv2.convertScaleAbs(image)
        # Get max contour
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cropped and binerized
        max_area = 0
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if w*h > max_area:
                max_area = w*h
                roi = image[y:y+h,x:x+w]
        # take the maximum roi to each connected component
        dic_bin[key] = roi
    return dic_bin

def cc_normalized(dic_rotated, dic_cc, opt = "binary"):
    """
    Find the normalized cc for each file image

    Parameters
    ----------
    dic_rotated : dictionary
        DESCRIPTION.
    dic_cc : dictionary
        DESCRIPTION.
    opt : "binary" or "rgb", optional
        Optional for processing cc patches. The default is "binary".

    Returns
    -------
    transform : TYPE
        DESCRIPTION.

    """
    transform = {} # dict each image channel consist of dict of cc image
    for key, value in dic_rotated.items():
        _, one_rotated = value
        one = dic_cc[key]
        transform[key] = img_transform(one, one_rotated, opt)
    return transform
        
#=======================================================================
# TODO 7: Get GT and rescale the size of coordinates
#=======================================================================

def msra_gt(path, img):
    """
    GT utk MSRA-TD500
    file_name = 1 file
    output 4 point dalam np.array
    """
    scala=480/img.shape[1]
    list_lokasi = []
    list_angle = []
    with open (f"{path}",'r') as f:
    
        for line in f :
            _,_,x,y,w,h,angle = line.split()
            
            xc = (int(x)+(int(w)/2))*scala
            yc = (int(y)+(int(h)/2))*scala
            
            angle = np.degrees(float(angle))
            rect = (xc,yc),(int(w)*scala,int(h)*scala),angle
            box=cv2.boxPoints(rect)
            box = np.int0(box)    
               
            list_lokasi.append(box) 
            list_angle.append(angle)
            
    return  list_lokasi, list_angle


def bbox_in_image(box, img):
    """

    Parameters
    ----------
    box : TYPE
        DESCRIPTION.
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    stencil = np.zeros(img.shape).astype(img.dtype)
    contours = [np.array(box, dtype = np.int32)]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    result = cv2.bitwise_and(img, stencil)
    return result


def quard_point(box):
    return [[box[0][0],box[0][1]],[box[1][0],box[0][1]],
            [box[1][0],box[1][1]],[box[0][0],box[1][1]]]


# Intersection of Union for 2 bounding box
def iou_area(box1,box2,img):
    '''Find the iou score from 2 bbox 
    input : [[x1,y1],[x2,y2]]list coordinate '''
    
    # cek size
    if len(box1) == 2 :
        box1=quard_point(box1)
        box2=quard_point(box2)
    
    result1 = bbox_in_image(box1, img)
    result2 = bbox_in_image(box2, img)
    
    intersection = np.logical_and(result1, result2)
    union = np.logical_or(result1, result2)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def one2all_gt(img, box, boxes_gt):
   """ compare iou 1 cc with all gt bbox
   output list iou
   """ 
   return [iou_area(box, boxes_gt[i], img) for i in range (len(boxes_gt))]


def all2all_gt(img, dic_boxes, boxes_gt):
   """ dic result of iou of all cc with all gt"""
   list_keys = list(dic_boxes.keys())
   list_value = list(dic_boxes.values())
   return {key: one2all_gt(img, box, boxes_gt) 
           for key,box in zip(list_keys, list_value)}


def count_iou(dic_box, list_gt, img):
    """
    Processing iou for 1 image with 6 channel
    """
    dic_iou = {}
    dic_gt = {}
    for key in dic_box.keys():
        coordinate, _ = dic_box[key]
        iou_lists = all2all_gt(img, coordinate, list_gt)
        # convert 
        iou_all = np.array(list(iou_lists.values()))
        # get max IOU of 1 type image for each GT to calculate acc
        # then collect for 6 type image
        if np.size(iou_all) == 0: # no nontext
            dic_iou[key] = np.zeros(len(coordinate))
            dic_gt[key] = np.zeros(len(list_gt))
        else:
            dic_iou[key] = np.max(iou_all, axis=1) # iou each cc
            # iou each channel images, to measure accuracy early cc detection
            dic_gt[key] = np.max(iou_all, axis=0)  
    return dic_iou,  dic_gt
  

def get_img_iou(dic_img, dic_iou):
    """Get img and iou in one dictionary of 6 channel images"""
    result = {}
    for key in dic_img.keys():
        images = dic_img[key]
        iou_cc = dic_iou[key]
        result[key] = {index:(images[index], iou_cc[i]) for i, index in
                       zip(range(len(iou_cc)), images.keys())}
    return result


#=======================================================================
# TODO 8: LBP and HOG, collect all features for dataset
#=======================================================================
# import the necessary packages
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


def LBP_features(dic_cc, desc):
    """
    Create features from one channel image. Each channel consist of
    connectect components (image, iou)

    Parameters
    ----------
    dic_cc : dict
        {index:(cropping_cc, iou),...}.
    desc : LBP object 
        LBP object declare the parameter radius and number of neighbours.

    Returns
    -------
    features : numpy array 1x28 if number of neighbour is 24
        [index, number of neighbour + 2, iou] for all cc from one channel image
    """
    features = []
    for index in dic_cc.keys():
        im, iou = dic_cc[index]
        hist = desc.describe(im)
        features.append (np.c_[index, hist.reshape(1,np.size(hist)), 
                         np.array(iou).reshape(1,1)])
    return features


def get_features(cc_img, desc):
    """
    Collect all binary pattern of 6 channel image

    Parameters
    ----------
    cc_img : dic
        {gray: {index:(cropping_cc, iou)...}...}
    desc : LBP object 
        LBP object declare the parameter radius and number of neighbours.
    
    Returns
    -------
    dict 
        dictionary of features for each image channel, connected component.
    """
    return {index: LBP_features(cc_img[index], desc) 
            for index in cc_img.keys() if len(index)!=0}


def collect_features(features_LBP, neighbours):
    """
    Get only features from each cc and put in one numpy array 2d

    Parameters
    ----------
    features_LBP : np array
        all features from cc in one image.
    neighbours : skalar
        number of neighbour.

    Returns
    -------
    np array
        array of all features of 1 image file.
    """
    data = np.zeros((1,neighbours + 4))
    for features in features_LBP.values():
        if len(features) != 0: # cek empty list of cc from one channel image
            number_cc = np.size(features[0])
            one_channel = np.array(features).reshape(len(features), number_cc)
            data =np.concatenate((data,one_channel), axis = 0)
        else:
            continue
    return data[1:][:]


def list2array(data):
    """
    Convert list to array
    """
    all_data = data[0]
    for i in range(1,len(data)):
        one_data = data[i]
        all_data = np.block([[all_data], [one_data]])
    return all_data


#=======================================================================
# TODO 9: Saving and Open pickle file
#=======================================================================
import pickle
def open_pickle(path, type = "python"):
    file = open(path, "rb")
    if type == "data_frame":
        loaded_dictionary = pd.read_pickle(path)
    else:
        loaded_dictionary = pickle.load(file)
    return loaded_dictionary


def save_pickle (dic, path, type = "python"):
    file = open(path, "wb")
    if type == "data_frame":
        dic.to_pickle(path)
    else:
        pickle.dump(dic, file)



def check_target (iou_all, transform_images):  
    keys = list(transform_images.keys())
    if iou_all.size == 0 :  
        return {keys[i]:(transform_images[keys[i]], "nonteks") 
                for i in range(len(keys))}
    cc_iou = np.max(iou_all, axis=1)
    img_class = {}
    for i in range(len(cc_iou)):
        if cc_iou[i] == 0 :
            img_class[keys[i]] = (transform_images[keys[i]], "nonteks")
        elif cc_iou[i] < 0.2 :
            img_class[keys[i]] = (transform_images[keys[i]], "ambigu")
        else:
            img_class[keys[i]] = (transform_images[keys[i]], "teks")
    return img_class
    

#=======================================================================
# TODO 9: Get cropped horizontal cc
#=======================================================================
# Classify cc into long and square categories
def classify_size(img_cropped):
    h,w=img_cropped.shape
    
    if (w > 1.6*h) :  
        resized = cv2.resize(img_cropped, (32,16), interpolation = cv2.INTER_NEAREST)
        size_img = "long"   
    else:
        resized = cv2.resize(img_cropped, (24,24), interpolation = cv2.INTER_NEAREST)
        size_img = "square"
    return resized, size_img


# dictionary per one image channel
def cropped_horizontal(img, feature, horizontal = False):
    dic_cropped = {}
    for key, feat in feature.items():
        image = img[key]
        crop = image[feat[2]:feat[3]+1,feat[0]:feat[1]+1]
        if horizontal:
            # classify image
            crop, size = classify_size(crop)
            dic_cropped[key] = (crop,size)
        else:
            dic_cropped[key] = crop
    return dic_cropped

# dictionary per image
def horizontal_img(dic_img, dic_feature):
    return {key:cropped_horizontal(dic_img[key], dic_feature) 
            for key in dic_img.keys()}




