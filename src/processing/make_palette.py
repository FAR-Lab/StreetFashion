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
from src.utils.to_csv import to_csv

from user.params.io import PALETTE_OUTPUT_DIR_PRE_AGG

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

from datetime import datetime

import asyncio
import aiofiles

# Class Definition


class ColorPalette:
    # Class Methods
    def get_last_digits(self, num, digits=2):
        """
        returns the last digits of an int
        :param num: int
        :param digits: int
        
        :return: int
        """
        return num % 10 ** digits

    def apply_color_map(self, image_array, labels):
        """
        maps labels to colors specific in config 
        :param image_array: numpy array
        :param labels: list of dicts
        
        :return: numpy array
        """
        color_array = np.zeros(
            (image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8
        )

        for label_id, label in enumerate(labels):
            ## set all pixels with the current label to the color of the current label
            color_array[image_array == label_id] = label["color"]

        return color_array

    def __init__(self):
        self.log = setup_logger("color_palette")

        # self.image = empty numpy array
        self.id = None
        self.image = None
        self.pixels = None

        self.labarr_r = None
        self.kmlabels_ = None
        self.rgbarr_r = None
        self.countsarr = None
        self.kmcluster_centers_ = None
        self.palettecounts = None

        self.palette_arr = []
        self.palettecounts_arr = []

        self.skipseg = False

    async def load_image(self, image_path: str):
        """
        loads image from path
        :param image_path: str
        """
        self.id = image_path.split("/")[-1].split(".")[0]

        async with aiofiles.open(image_path, mode="rb") as f:
            # read image asynchronosly with cv2
            self.image = await f.read()
            self.image = np.frombuffer(self.image, dtype=np.uint8)
            self.image = cv2.imdecode(self.image, cv2.IMREAD_COLOR)
            self.image = np.array(self.image)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.pixels = np.float32(self.image).reshape(-1, 3)

    def apply_kmeans(self):
        try: 
    
            pixelbins = (
                np.float32(self.pixels) // 16
            )  # calculates bins from 0-255 to 0-15 (16 values)
            binarr = []
            for row in range(
                len(pixelbins) - 1
            ):  # combine the pixel bins into one value per row, preceded by "1"
                binstr = "1{:02d}{:02d}{:02d}".format(
                    int(pixelbins[row, 0]),
                    int(pixelbins[row, 1]),
                    int(pixelbins[row, 2]),
                )
                binarr.append(int(binstr))
            counts = dict(
                Counter(binarr)
            )  # histogram (counts per item) is stored as dictionary

            ## step 2: for each bin, compute mean color in LAB space
            ## right now, taking bin edge color and not average color per bin
            ## convert back from combined int to RGB tuple
            labarr = []
            self.countsarr = []
            rgbarr = []
            for item in counts:
                b = (
                    self.get_last_digits(item, 2) + 1
                ) / 16  # converting from valu 0-15 into value 0-1
                g = (((self.get_last_digits(item, 4) - b) // 100) + 1) / 16
                r = (((self.get_last_digits(item, 6) - g - b) // 10000) + 1) / 16
                rgbitem = [r, g, b]
                rgbarr.append(rgbitem)
                ## convert from RGB to LAB
                labitem = skimage.color.rgb2lab([r, g, b])
                labarr.append(labitem)
                self.countsarr.append(counts.get(item))
            self.labarr_r = np.float32(labarr).reshape(-1, 3)
            self.rgbarr_r = np.float32(rgbarr).reshape(-1, 3)
            self.countsarr = np.float32(self.countsarr)

            ## step 3: we now have an array with counts for each color, and an array with the corresponding LAB colors
            ## (based on the bin edge). these are countsarr and labarr
            ## we use weighted k-means instead of k-means, as we are looking at counts of points and not just points
            X = self.labarr_r
            sample_weight = (
                self.countsarr
            )  # @bremers, why is this variable unused?
            clustercount = 5
            if len(self.countsarr) < 5:
                self.log.info("Skipping segment because the number of colors < k")
                self.skipseg = True
                return None, None
            km = KMeans(
                n_clusters=clustercount,
                init="k-means++",
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0,
            )
            y_km = km.fit_predict(
                X, sample_weight=self.countsarr
            )  # @bremers, why is this variable unused?

            self.kmlabels_ = km.labels_
            self.kmcluster_centers_ = km.cluster_centers_
            self.palettecounts = [
                np.sum(self.kmlabels_ == 0),
                np.sum(self.kmlabels_ == 1),
                np.sum(self.kmlabels_ == 2),
                np.sum(self.kmlabels_ == 3),
                np.sum(self.kmlabels_ == 4),
            ]
            self.log.debug(f"Palette counts: {self.palettecounts}")
            self.log.debug(self.kmcluster_centers_)
            self.log.debug(self.kmlabels_)
            self.log.debug(self.pixels)

            # palette=np.uint8([rgb_cluster_centers])
            self.palette_arr.append(np.uint8(self.kmcluster_centers_))
            self.palettecounts_arr.append(np.uint8(self.palettecounts))

            return (self.kmcluster_centers_, self.palettecounts)
        
        except Exception as e:
            self.log.error(e)
            return

    def make_3d_plot(self):
        try: 
            plt.rcParams["figure.figsize"] = [21.00, 10.50]
            plt.rcParams["figure.autolayout"] = True
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            data = self.labarr_r
            # plotting each color that appears in the image as a point
            L, a, b = self.labarr_r[:, 0], self.labarr_r[:, 1], self.labarr_r[:, 2]
            label = self.kmlabels_
            ax.scatter(
                L, a, b, c=self.rgbarr_r, s=self.countsarr * 6, alpha=0.5
            )  # use c=label for cluster colors
            # plotting the centers too
            try:
                Lc, ac, bc = (
                    self.kmcluster_centers_[:, 0],
                    self.kmcluster_centers_[:, 1],
                    self.kmcluster_centers_[:, 2],
                )
            except TypeError as e:
                self.log.error(e)
                return

            ax.scatter(Lc, ac, bc, marker="*", c="red", s=100, alpha=1)
            ax.set_xlabel("L")
            ax.set_ylabel("a")
            ax.set_zlabel("b")
            plt.grid()
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(f"{PALETTE_OUTPUT_DIR_PRE_AGG}/3dPlot_{self.id}_{now}.png")

            plt.close()
            plt.clf()
        except Exception as e:
            self.log.error(e)
            return

    def plot_seg_palette(self):
        if self.skipseg:
            return

        try: 
            rows = self.kmcluster_centers_.shape[0]
            rgb_cluster_centers = []
            for row in range(rows):
                lab = self.kmcluster_centers_[row, :]
                rgb = skimage.color.lab2rgb(lab) * 255
                rgb = rgb.reshape(1, 3)
                rgb_cluster_centers.append([rgb[0, 0], rgb[0, 1], rgb[0, 2]])

            palette = np.uint8([rgb_cluster_centers])
            self.palette_arr.append(np.uint8(rgb_cluster_centers))
            self.palettecounts_arr.append(np.uint8(self.palettecounts))

            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(3, 3))
            ax0.imshow(self.image)
            ax0.set_title("Human")
            ax0.axis("off")
            ax1.imshow(palette)
            ax1.set_title("Palette")
            ax1.axis("off")
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(f"{PALETTE_OUTPUT_DIR_PRE_AGG}/SegPalette_{self.id}_{now}.png")

            plt.close()
            plt.clf()
        except Exception as e: 
            self.log.error(e)
            return

    def plot_img_palette(self):
        try: 
            fig, (ax0) = plt.subplots(1, 1)
            ax0.imshow(self.palette_arr)
            ax0.set_title("Image h-palette")
            ax0.axis("off")
            fig.set_figwidth(1)
            fig.set_figheight(len(self.palette_arr))
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(f"{PALETTE_OUTPUT_DIR_PRE_AGG}/ImgPalette_{self.id}_{now}.png")

            plt.close()
            plt.clf()
        except Exception as e: 
            self.log.error(e)
            return
        
    @to_csv("df")
    def write_palette(self):
        try:
            paletteinfo = {
                "id": self.id,
                "centers": self.palette_arr,
                "counts": self.palettecounts_arr,
            }
            return pd.DataFrame.from_dict(paletteinfo)
        except Exception as e:
            self.log.error(e)
            return

    async def __call__(self, image_path: str):
        await self.load_image(image_path)
        self.apply_kmeans()
        self.make_3d_plot()
        self.plot_seg_palette()
        self.plot_img_palette()
        self.write_palette()

        return self.palette_arr, self.palettecounts_arr
