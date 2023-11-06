# FARLAB - StreetFashion Project 
# Developer: @mattwfranchi, @bremers 

# This script houses a class to consolidate functionality provided in Alexandra's Jupyter Notebook -- generating color palette from a given image.

# Module Imports 
import os 
import sys 

sys.path.append(os.path.join(".."))
sys.path.append(os.path.join("..", ".."))

from src.utils.logger import setup_logger 
from src.utils.timer import timer 

from user.params.io import *

import json 
import colorsys 
import re 
from math import sqrt 
from ast import literal_eval 
from collections import Counter


import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import skimage 
import sklearn 
import cv2 
import scipy.spatial.distance
from PIL import Image
from skimage import color
from skimage import data
from sklearn.cluster import KMeans

# Class Definition

class ColorPalette:
    # Class Methods 
    def get_last_digits(num, digits=2):
        """
        returns the last digits of an int
        :param num: int
        :param digits: int
        
        :return: int
        """
        return num % 10**digits
    
    def apply_color_map(image_array, labels):
        """
        maps labels to colors specific in config 
        :param image_array: numpy array
        :param labels: list of dicts
        
        :return: numpy array
        """
        color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

        for label_id, label in enumerate(labels):
            ## set all pixels with the current label to the color of the current label
            color_array[image_array == label_id] = label["color"]

        return color_array

    def __init__(self): 
        # self.image = empty numpy array 
        self.image = None
        self.pixels = None 
    
    def load_image(self, image_path: str): 
        """
        loads image from path
        :param image_path: str
        """
       
        self.image=cv2.imread(image_path)
        self.image = np.array(self.image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB) # making sure the colors show up correctly when we plot it
        
        
        self.pixels = np.float32(self.image).reshape(-1, 3)



     


    


