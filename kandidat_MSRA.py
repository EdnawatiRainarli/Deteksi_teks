# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 19:02:19 2022

@author: ednaw
"""


import praproses as pre
import cv2
import pickle
import numpy as np
import pandas as pd

# Data Set MSRA-TD500
#PATH = "D:\\MSRA-TD500\\train" ###
PATH =  "MSRA-TD500/test"
#SAVE_PATH = "MSRA_train3_horizontal.pickle"###791
SAVE_PATH = "MSRA_test2_horizontal.pickle"

EXTENTION = ".JPG"

def open_pickle():
    #file = open("D:\\Deteksi scene text\\XG_Boost\\MSRA_split.pickle", "rb")###
    file = open("MSRAtest_split.pickle", "rb")
    loaded_dictionary = pickle.load(file)
    return loaded_dictionary


def result_df(dic):
    rows = []
    for file, cc_file in dic.items():
        for channel, cc_dic in cc_file.items():
            for key, values in cc_dic.items():
                img, iou = values
                
                row = [file,channel,key,img,iou]
                rows.append(row)

    df = pd.DataFrame(rows)
    df = df.rename(columns={0: "name_of_file", 1: "channel_image",
                            2: "index_cc", 3: "normalized_cc", 4:"iou_score"})
    return df


def position_row(dic):
    rows = []
    for file, cc_file in dic.items():
        for channel, cc_dic in cc_file.items():
            for key, value in cc_dic.items():
                
                row = [file, channel, key, value[0:4], value[4], value[5], value[6], value[7]]
                rows.append(row)
   
    df = pd.DataFrame(rows)
    df = df.rename(columns={0: "name_of_file", 1: "channel_image",
                            2: "index_cc", 3: "(x1,x3),(y1,y3)", 4:"area", 
                            5:"parent", 6: "average", 7: "std"}) 
    return df


def cc_row(dic):
    rows = []
    for file, cc_file in dic.items():
        for channel, cc_dic in cc_file.items():
            coordinate, position = cc_dic 
            for key, value in coordinate.items():
                pos = position[key]
                row = [file,channel,key,pos,value]
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.rename(columns={0: "name_of_file", 1: "channel_image",
                            2: "index_cc", 3: "center_angle", 
                            4: "rotated_coordinate"})
    return df


def combine_df(df,df1,df2):
    location = df1.loc[:, "center_angle":"rotated_coordinate"]
    properties = df2.loc[:,"(x1,x3),(y1,y3)":"std"]
    
    df3 = df.join(properties)
    df4 = df3.join(location)
    return df4

def horizontal_df(dic):
    rows = []
    for file, cc_file in dic.items():
        for channel, cc_dic in cc_file.items():
            for key, value in cc_dic.items():
                cc, size = value
                row = [file,channel,key, cc, size]
                rows.append(row)
                
    df = pd.DataFrame(rows)
    df = df.rename(columns={0: "name_of_file", 1: "channel_image",
                            2: "index_cc", 3: "horizontal_cc"})   
    return df


def main():
    name_files = pre.list_file(PATH, type_file = EXTENTION)
    # get dic gt
    name_gt = open_pickle()
    # convert into list
    name_gt = list(name_gt.values())
    
    # choose number of file and gt to process
    files = name_files[100:200] #237:239 cek 1941-1947
    #gt = name_gt[0:100]   #
    
    all_horizontal = {}
    #all_result = {}
    #all_properties = {}
    #all_rotated = {}
    # looping to read file
    for i in range (len(files)):
        img = cv2.imread(f"{PATH}/{files[i]}")
    
        # resize one image
        img_rgb = pre.resize_img(img)
        dic_img = pre.preprocessing(img)
        
        # create dictionary of 6 maxtree 
        dic_mxt = {key: pre.mxt(value) for key, value in dic_img.items()}
        
        #===========================================================
        # 1. filtering based on variation value between node and it's parent
        #===========================================================
        dic_variation = {key:pre.filter_area(mxt) for key, mxt in 
                          zip(dic_img.keys(),dic_mxt.values())}
        
        #===========================================================
        # 2. filtering based on geometric rules
        #===========================================================
        dic_geometric = {key:pre.filter_geometric(cc) 
                         for key, cc in dic_variation.items()}
        
        #===========================================================
        # 3. prunning using Standar Deviation & cutting tree
        #===========================================================
        dic_SD = {key:pre.filtered_std(dic_mxt[key],cc) 
                          for key, cc in dic_geometric.items()}
        
        img_filter, cutting_img = pre.prunning_cc(dic_SD)
        
            
        #===========================================================
        # 4. Binerized/Grayscale, Rotated Bounding Box, Normalized and Cropping
        #===========================================================
        # binerized images before normalized
        #dic_biner = {key: pre.dic_rotated_bb(cutting_img[key], img_filter[key]) 
        #              for key in img_filter.keys()}
        
        # gray image
        dic_gray =  {key: pre.dic_rotated_bb(cutting_img[key], img_filter[key], 
                                             option = "rgb") for key in img_filter.keys()}
        
        
        # coordinate rotated bounding box dic_coordinate, dic_rotatedbb
        # (4,2) numpy array dan (xc,yc),(w,h),angle
        dic_rotated = {key: pre.dic_rbox (cutting_img[key], img_filter[key])
                       for key in img_filter.keys()}
        
        # get dictionary consist of cropping horizontal cc
        dic_horizontal = {key: pre.cropped_horizontal(dic_gray[key],cutting_img[key], horizontal = True)
                          for key in dic_gray.keys() } # Eror img_1723 1724
        
        # normalized images
        # dic_transform =  pre.cc_normalized(dic_rotated, dic_biner)
        #gray_transform = pre.cc_normalized(dic_rotated, dic_gray, opt = "rgb")
        
        # cropping normalized images
        # dic_cropped = {key:pre.get_cc(img_cc) 
        #               for key,img_cc in dic_transform.items()}
        # cropping normalized gray images 
        #gray_cropped = {key:pre.get_cc(img_cc) 
        #                for key,img_cc in gray_transform.items()}
        
        #===========================================================
        # 6. Counting IOU and the Normalized CC 
        #===========================================================
        # crop file name for labelling image after preprocessing
        f = files[i].rstrip(EXTENTION)
        print(f, flush=True)
       
        #iou, _ = pre.count_iou(dic_rotated, gt[i], img_rgb)
        
        #result = pre.get_img_iou(dic_cropped, iou)
        #result = pre.get_img_iou(gray_cropped, iou)
         
        #all_result[files[i]] = result
        #all_properties[files[i]] = cutting_img
        #all_rotated[files[i]] = dic_rotated
        all_horizontal[files[i]] = dic_horizontal
    
    # Combine into one dataframe
    #result_pd = result_df(all_result)
    #properties_pd = position_row(all_properties)
    #rotated_pd = cc_row(all_rotated)
    #df = combine_df(result_pd,rotated_pd,properties_pd)
    df = horizontal_df(all_horizontal)
    # Save file
    pre.save_pickle (df, SAVE_PATH, type = "data_frame")
    return df #all_result, all_properties, all_rotated

#all_properties,all_result, all_rotated = main()
#hasil = pre.open_pickle(SAVE_PATH, type = "data_frame")

# index, nama file, channel, index_cc, normalized, iou, (x1,y1)(x3,y3), area, 
#parent, average, std 

#gt = open_pickle()

if __name__=="__main__":
  main()
