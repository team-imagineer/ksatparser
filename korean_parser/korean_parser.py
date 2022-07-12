import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import cv2
from pathlib import Path
from tqdm import tqdm
import math
import pdfplumber

class Parser:
    def __init__(self, pdf_path):
        assert os.path.exists(pdf_path)
        book = convert_from_path(pdf_path)



    def parse(self):
        _divide_lines()
        pass

