# FARLAB - StreetFashion
# Developer: @mattwfranchi, @bremers
# Last Edited: 11/06/2023

# This script houses a driver class to run batches of images through the color palette generator.

# Module Imports
import os
import sys

sys.path.append(os.path.join(".."))
sys.path.append(os.path.join("..", ".."))

import aiofiles
import asyncio

from glob import glob

from src.utils.logger import setup_logger
from src.utils.timer import timer
from src.utils.to_csv import to_csv

from src.processing.make_palette import ColorPalette

from user.params.io import PALETTE_OUTPUT_DIR_PRE_AGG

from random import shuffle 

class PaletteGenerator:
    def __init__(self, image_dir):
        self.log = setup_logger("palette_generator")
        self.log.setLevel("INFO")

        self.image_dir = image_dir
        self.images = glob(os.path.join(self.image_dir, "*.png"))

        # shuffle the images
        shuffle(self.images)

        self.log.info(f"Found {len(self.images)} images in {self.image_dir}")

        # make sure the output directory exists
        os.makedirs(PALETTE_OUTPUT_DIR_PRE_AGG, exist_ok=True)

    def generate_palette(self, image_path):
        """
        Generates a color palette from a given image. 
        :param image_path: str 
        :return: list 
        """
        palette = ColorPalette()
        return palette(image_path)

    async def __call__(self):
        self.log.info("Starting palette generation...")

        # Create a list of tasks to run
        tasks = [
            self.generate_palette(image_path) for image_path in self.images
        ]

        # Run the tasks
        results = await asyncio.gather(*tasks)

        # Convert the results to a datafram
        palette_arrs, palettecount_arrs = zip(*results)

        self.log.success("Finished palette generation.")

        return palette_arrs, palettecount_arrs
