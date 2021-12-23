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


parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True, help='Path to image')
args = parser.parse_args()

def problem_url_generator(dataframe):
    dataframe = dataframe['name'].tolist()
    
    operators = ['integrate', 'derivative']
    differentials = ['dx', 'dxdy']
    operands = ['x', '1/x', 'lnx', 'x^2', 'x^3', 'e^x', 'x^e', 'sqrt(x)', 'x/2', 'x/3', 'sinx', 'cosx', 'tanx', 'cotanx', 'dxdy', 'x+y', 'y^2', '(x+y)/y^2' ]
    
    problem_set = []
    
    for operator in dataframe:
        if not problem_set and operator in operators:
            problem_set.append(operator)
            dataframe.remove(operator)
        for operand in dataframe:
            if problem_set and problem_set[0] in operators and operand in operands:
                problem_set.append(operand)
                dataframe.remove(operand)
        for differential in dataframe:
            if problem_set and len(problem_set) == 2 and problem_set[1] in operands and differential in differentials:
                problem_set.append(differential)
                dataframe.remove(differential)
                
        problem = ' '.join(problem_set)
        url = "https://www.wolframalpha.com/input/?i=" + urllib.parse.quote(problem)
        
    return problem_set, problem, url

if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='problem-solver-32-416/exp17/weights/last.pt', force_reload=True)
    image = cv2.imread(args.image)
    result = model(image)
    problem_set = problem_url_generator(result.pandas().xyxy[0])

    print(f'We found "{problem_set[1]}" as your problem.')
    pass