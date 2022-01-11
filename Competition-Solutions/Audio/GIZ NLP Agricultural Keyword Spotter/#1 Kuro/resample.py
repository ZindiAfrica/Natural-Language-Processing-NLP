# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:41:07 2020

@author: Shiro


organisation of the folder : 
    
    
|-- resample.py
|-- audio_files
|-- AdditionalUtterances
|-- nlp_keywords
"""


import pandas as pd
import numpy as np
import os
import librosa
import scipy.io.wavfile
from tqdm import tqdm
tr = pd.read_csv("train.csv")


def get_additional_data(path):
    labels = []
    all_paths = []
    for name in os.listdir(path):
        pname = os.path.join(path, name)
        #print(pname)
        for filename in os.listdir(pname):
            fname = os.path.join(pname, filename)
            all_paths.append(fname)
            labels.append(name)
            
    return all_paths, labels

def get_train_test_data(path):
    all_paths = []
    for name in os.listdir(path):
        pname = os.path.join(path, name)
        all_paths.append(pname)
    return all_paths

def write_file(arr, path, sr=22050):
    directory = os.path.split(path)[0] 
    if not os.path.exists(directory):
        os.makedirs(directory)

    scipy.io.wavfile.write(path,data=arr, rate=sr)



## - choose sampling RATE   
SR = 48000 # 22050 #16000

# additional data1
# source folder    
paths =  "nlp_keywords"#"AdditionalUtterances/latest_keywords" #  

# destination folder
paths2 =  "nlp_keywords-48000"#"AdditionalUtterances-48000/latest_keywords" # 


all_paths, labels = get_additional_data(paths)

s = set()
for p in tqdm(all_paths):
    y, sr = librosa.load(p, sr=SR) # load + resample
    p2 = p.replace(paths, paths2)
    #p2 = p.replace(NAME, NAME+f"-{SR}")
    write_file(y, p2, sr=SR)

# additional data2
# source folder    
paths = "AdditionalUtterances/latest_keywords" #  

# destination folder
paths2 =  "AdditionalUtterances-48000/latest_keywords" # 


all_paths, labels = get_additional_data(paths)

s = set()
for p in tqdm(all_paths):
    y, sr = librosa.load(p, sr=SR) # load + resample
    p2 = p.replace(paths, paths2)
    #p2 = p.replace(NAME, NAME+f"-{SR}")
    write_file(y, p2, sr=SR)


## convert train test
NAME = "audio_files" 
all_paths = get_train_test_data(NAME)

s = set()
for p in tqdm(all_paths):
    y, sr = librosa.load(p, sr=SR) # load + resample
    p#2 = p.replace(paths, paths2)
    p2 = p.replace(NAME, NAME+f"-{SR}")
    write_file(y, p2, sr=SR)
