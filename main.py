import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import urllib
import pandas as pd


### There will be an OS detection tool here for macOS.  
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context