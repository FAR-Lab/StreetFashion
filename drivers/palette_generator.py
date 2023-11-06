# FARLAB - StreetFashion
# Developer: @mattwfranchi, @bremers
# Last Edited: 11/06/2023

# This script houses a driver for the palette generator class.

# Module Imports
import os
import sys

sys.path.append(os.path.join(".."))
sys.path.append(os.path.join("..", ".."))

import asyncio

from src.processing.palette_generator_batch import PaletteGenerator

if __name__ == "__main__":

    image_dir = str(sys.argv[1])

    palette_generator = PaletteGenerator(image_dir)
    palette_arrs, palettecount_arrs = asyncio.run(palette_generator())
