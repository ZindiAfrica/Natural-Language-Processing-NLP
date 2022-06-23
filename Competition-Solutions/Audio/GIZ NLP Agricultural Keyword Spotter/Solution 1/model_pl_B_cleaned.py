# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 00:08:10 2020

@author: Logteta

"""


import librosa
print(librosa.__version__)
import scipy.io.wavfile
from efficientnet_pytorch import EfficientNet

from torch import nn
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import *
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler 
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torch.nn.utils.rnn import *
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler 
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain

import torchvision
import torchvision.models as models

import librosa
import librosa.display

from joblib import Parallel, delayed
import os
import numpy as np

import pandas as pd


import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from sklearn.metrics import f1_score
import pickle
import random 
from sklearn.model_selection import StratifiedKFold
SEED = 42


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
seed_everything(SEED)


import torch
import torch.nn as nn
import torch.nn.functional as F


# Data augmentation

class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes. 
        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]    # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input, replacement):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
              self.transform_slice(input[n], total_width, replacement=replacement[n])
                
            return input


    def transform_slice(self, e, total_width, replacement=0.):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]
            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = replacement
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = replacement


class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, 
        freq_stripes_num, replace="mean"):
        """Spec augmetation. 
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D. 
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method 
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width, 
            stripes_num=time_stripes_num)

        self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width, 
            stripes_num=freq_stripes_num)
        self.replace = replace
        
    def forward(self, input):
        if self.replace == "zero":
            replacement = torch.zeros(len(input)).to(input.device)
        else:
            replacement = input.mean(-1).mean(-1)
        x = self.time_dropper(input, replacement=replacement)
        x = self.freq_dropper(x, replacement=replacement)
        return x
    



class AudioClassifier(nn.Module):
    def __init__(self, backbone, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(AudioClassifier, self).__init__()    

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.bn = nn.BatchNorm2d(3)
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.backbone =  backbone  


   

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, length, data_length)
        """
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        with autocast(False):
          x = self.logmel_extractor(x)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        x = self.bn(torch.cat([x,x,x], dim=1))
        x = self.backbone(x)

        return x

    
class AudioClassifierHub(nn.Module):
    def __init__(self, backbone, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(AudioClassifierHub, self).__init__()    

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.bn = nn.BatchNorm2d(3)
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.backbone =  backbone  
        
        in_feat = backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_feat, classes_num)

   

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, length, data_length)
        """
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        with autocast(False):
          x = self.logmel_extractor(x)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        x = self.bn(torch.cat([x,x,x], dim=1))
        x = self.backbone(x)

        return x
    


class AgrinetDatasetSampler(Sampler):
    def __init__(self, dataset):
        self.num_samples = len(dataset)
        self.indices = list(range(self.num_samples))

        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    
    def _get_label(self, dataset, idx):
        label = dataset.get_label(idx)
        return label
    
    
class AudioGeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, path_audio, y, resample_freq = 32000, max_length=3, augmentation=[], validation=False, num_class=264, pseudo_labels=None):
        self.labels2idx = {'Pump': 0, 'Spinach': 1,  'abalimi': 2,  'afukirira': 3,  'agriculture': 4, 'akammwanyi': 5,  'akamonde': 6, 'akasaanyi': 7, 'akatunda': 8, 'akatungulu': 9,
      'akawuka': 10, 'amakoola': 11, 'amakungula': 12, 'amalagala': 13, 'amappapaali': 14, 'amatooke': 15, 'banana': 16, 'beans': 17, 'bibala': 18, 'bulimi': 19, 'butterfly': 20, 'cabbages': 21,
      'cassava': 22, 'caterpillar': 23, 'caterpillars': 24, 'coffee': 25, 'crop': 26, 'ddagala': 27, 'dig': 28, 'disease': 29, 'doodo': 30, 'drought': 31, 'ebbugga': 32, 'ebibala': 33, 'ebigimusa': 34,
      'ebijanjaalo': 35, 'ebijjanjalo': 36, 'ebikajjo': 37, 'ebikolo': 38, 'ebikongoliro': 39, 'ebikoola': 40, 'ebimera': 41, 'ebinyebwa': 42, 'ebirime': 43, 'ebisaanyi': 44, 'ebisooli': 45,
      'ebisoolisooli': 46, 'ebitooke': 47, 'ebiwojjolo': 48, 'ebiwuka': 49, 'ebyobulimi': 50, 'eddagala': 51, 'eggobe': 52, 'ejjobyo': 53, 'ekibala': 54, 'ekigimusa': 55, 'ekijanjaalo': 56,
      'ekikajjo': 57, 'ekikolo': 58, 'ekikoola': 59, 'ekimera': 60, 'ekirime': 61, 'ekirwadde': 62, 'ekisaanyi': 63, 'ekitooke': 64, 'ekiwojjolo': 65, 'ekyeya': 66, 'emboga': 67, 'emicungwa': 68,
      'emisiri': 69, 'emiyembe': 70, 'emmwanyi': 71, 'endagala': 72, 'endokwa': 73, 'endwadde': 74, 'enkota': 75, 'ennima': 76, 'ennimiro': 77, 'ennyaanya': 78, 'ensigo': 79, 'ensiringanyi': 80, 'ensujju': 81,
      'ensuku': 82, 'ensukusa': 83, 'enva endiirwa': 84, 'eppapaali': 85, 'faamu': 86, 'farm': 87, 'farmer': 88, 'farming instructor': 89, 'fertilizer': 90, 'fruit': 91, 'fruit picking': 92,
      'garden': 93, 'greens': 94, 'ground nuts': 95, 'harvest': 96, 'harvesting': 97, 'insect': 98, 'insects': 99, 'irish potatoes': 100, 'irrigate': 101, 'kaamulali': 102, 'kasaanyi': 103, 'kassooli': 104,
      'kikajjo': 105, 'kikolo': 106, 'kisaanyi': 107, 'kukungula': 108, 'leaf': 109, 'leaves': 110, 'lumonde': 111, 'lusuku': 112, 'maize': 113, 'maize stalk borer': 114, 'maize streak virus': 115, 'mango': 116, 'mangoes': 117, 'matooke': 118,
      'matooke seedlings': 119, 'medicine': 120, 'miceere': 121, 'micungwa': 122, 'mpeke': 123, 'muceere': 124, 'mucungwa': 125, 'mulimi': 126, 'munyeera': 127, 'muwogo': 128,
      'nakavundira': 129, 'nambaale': 130, 'namuginga': 131, 'ndwadde': 132, 'nfukirira': 133, 'nnakati': 134, 'nnasale beedi': 135, 'nnimiro': 136, 'nnyaanya': 137, 'npk': 138, 'nursery bed': 139,
      'obulimi': 140, 'obulwadde': 141, 'obumonde': 142, 'obusaanyi': 143, 'obutunda': 144, 'obutungulu': 145, 'obuwuka': 146, 'okufukirira': 147, 'okufuuyira': 148, 'okugimusa': 149, 'okukkoola': 150,
      'okukungula': 151, 'okulima': 152, 'okulimibwa': 153, 'okunnoga': 154, 'okusaasaana': 155, 'okusaasaanya': 156, 'okusiga': 157,
      'okusimba': 158, 'okuzifuuyira': 159, 'olusuku': 160, 'omuceere': 161, 'omucungwa': 162, 'omulimi': 163, 'omulimisa': 164, 'omusiri': 165, 'omuyembe': 166,
      'onion': 167, 'orange': 168, 'pampu': 169, 'passion fruit': 170, 'pawpaw': 171, 'pepper': 172, 'plant': 173, 'plantation': 174, 'ppaapaali': 175, 'pumpkin': 176, 'rice': 177, 'seed': 178,
      'sikungula': 179, 'sow': 180, 'spray': 181, 'spread': 182, 'suckers': 183, 'sugarcane': 184, 'sukumawiki': 185, 'super grow': 186, 'sweet potatoes': 187, 'tomatoes': 188, 'vegetables': 189,
      'watermelon': 190, 'weeding': 191, 'worm': 192}
        
        self.idx2labels = {k:v for v,k in self.labels2idx.items()}
        identity = np.eye(num_class)
        self.augmentation = set(augmentation)
        self.samples = path_audio #+ path_augment
        self.max_length = max_length # 99% are shorter than 3 sec
        self.resample_freq=resample_freq
        self.validation = validation
        self.y = np.array([identity[self.labels2idx[t]] for t in y]).astype(np.float32) #+ [self.labels2idx[t] for t in y_aug]
        self.num_class = num_class
        self.noise = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.6),
                                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.6),
                                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                                Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
                                Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.6), 
                                ])
    
        if pseudo_labels is not None:
            self.add_pl(pseudo_labels[0], pseudo_labels[1])
    def load_raw_audio(self, x):
        signal_f = np.zeros((self.max_length*self.resample_freq)).astype(np.float32)
        signal, sr_orig = librosa.load(x, sr=self.resample_freq)
        if sr_orig != self.resample_freq:
            signal = librosa.resample(signal, orig_sr=sr_orig, target_sr=self.resample_freq, res_type="kaiser_best")
        
        shape = len(signal)
        if self.validation:
                signal = signal[:self.max_length*self.resample_freq]
                signal_f[:len(signal)] = signal
        else:
            if shape > self.max_length*self.resample_freq:
                start = np.random.randint(0, shape - self.max_length*self.resample_freq)
                signal_f = signal[start:start+self.max_length*self.resample_freq]
            elif shape == self.max_length*self.resample_freq:
                signal = signal[:self.max_length*self.resample_freq]
                signal_f[:len(signal)] = signal
            else:
                start = np.random.randint(0, self.max_length*self.resample_freq-shape)
                shape = len(signal[start:start+self.max_length*self.resample_freq])
                signal_f[start:start+shape] = signal[start:start+self.max_length*self.resample_freq]
        return signal_f.astype(np.float32)
    
    def add_pl(self, path_test, pseudo_labels):
        print("#old ", len(self.y), len(self.samples))
        
        self.y = np.concatenate((self.y, pseudo_labels), axis=0)
        
        self.samples.extend(path_test)
        print("#new ", len(self.y), len(self.samples))

        
    def normalise(self, x):
        """
        

        Parameters
        ----------
        x : numpy array
            DESCRIPTION.

        Returns
        -------
        x_norm : Normalise signal between 0 and 1 value
            DESCRIPTION.

        """
        min_ = x.min()
        max_ = x.max()
        if min_ == max_:
            x_norm = np.zeros(x.shape).astype(np.float32)
        else:
            x_norm =  (x-min_)/(max_ - min_)
        return x_norm
    
    def __getitem__(self, index):
        l = []
        # label
        labels_one_hot = torch.as_tensor(self.y[index]).type(torch.float32)
            
        # load signal
        signal_raw =  self.load_raw_audio(self.samples[index] )

        # add Environment Noise
        
        if "noise" in self.augmentation:
                signal_raw = self.noise(samples=signal_raw, sample_rate=self.resample_freq)
        
        
        l.append( torch.tensor(signal_raw) )
        l.append(labels_one_hot)
        l.append(torch.tensor(index))
        return tuple(l)
    
    def __len__(self):
        return len(self.samples)
    
    def get_label(self, idx):
        label = self.y[idx]
        return label
    
    
    
class AudioGeneratorDatasetTest(torch.utils.data.Dataset):
    def __init__(self, path_audio, resample_freq = 32000, max_length=3, num_class=264, ):
        self.labels2idx = {'Pump': 0, 'Spinach': 1,  'abalimi': 2,  'afukirira': 3,  'agriculture': 4, 'akammwanyi': 5,  'akamonde': 6, 'akasaanyi': 7, 'akatunda': 8, 'akatungulu': 9,
      'akawuka': 10, 'amakoola': 11, 'amakungula': 12, 'amalagala': 13, 'amappapaali': 14, 'amatooke': 15, 'banana': 16, 'beans': 17, 'bibala': 18, 'bulimi': 19, 'butterfly': 20, 'cabbages': 21,
      'cassava': 22, 'caterpillar': 23, 'caterpillars': 24, 'coffee': 25, 'crop': 26, 'ddagala': 27, 'dig': 28, 'disease': 29, 'doodo': 30, 'drought': 31, 'ebbugga': 32, 'ebibala': 33, 'ebigimusa': 34,
      'ebijanjaalo': 35, 'ebijjanjalo': 36, 'ebikajjo': 37, 'ebikolo': 38, 'ebikongoliro': 39, 'ebikoola': 40, 'ebimera': 41, 'ebinyebwa': 42, 'ebirime': 43, 'ebisaanyi': 44, 'ebisooli': 45,
      'ebisoolisooli': 46, 'ebitooke': 47, 'ebiwojjolo': 48, 'ebiwuka': 49, 'ebyobulimi': 50, 'eddagala': 51, 'eggobe': 52, 'ejjobyo': 53, 'ekibala': 54, 'ekigimusa': 55, 'ekijanjaalo': 56,
      'ekikajjo': 57, 'ekikolo': 58, 'ekikoola': 59, 'ekimera': 60, 'ekirime': 61, 'ekirwadde': 62, 'ekisaanyi': 63, 'ekitooke': 64, 'ekiwojjolo': 65, 'ekyeya': 66, 'emboga': 67, 'emicungwa': 68,
      'emisiri': 69, 'emiyembe': 70, 'emmwanyi': 71, 'endagala': 72, 'endokwa': 73, 'endwadde': 74, 'enkota': 75, 'ennima': 76, 'ennimiro': 77, 'ennyaanya': 78, 'ensigo': 79, 'ensiringanyi': 80, 'ensujju': 81,
      'ensuku': 82, 'ensukusa': 83, 'enva endiirwa': 84, 'eppapaali': 85, 'faamu': 86, 'farm': 87, 'farmer': 88, 'farming instructor': 89, 'fertilizer': 90, 'fruit': 91, 'fruit picking': 92,
      'garden': 93, 'greens': 94, 'ground nuts': 95, 'harvest': 96, 'harvesting': 97, 'insect': 98, 'insects': 99, 'irish potatoes': 100, 'irrigate': 101, 'kaamulali': 102, 'kasaanyi': 103, 'kassooli': 104,
      'kikajjo': 105, 'kikolo': 106, 'kisaanyi': 107, 'kukungula': 108, 'leaf': 109, 'leaves': 110, 'lumonde': 111, 'lusuku': 112, 'maize': 113, 'maize stalk borer': 114, 'maize streak virus': 115, 'mango': 116, 'mangoes': 117, 'matooke': 118,
      'matooke seedlings': 119, 'medicine': 120, 'miceere': 121, 'micungwa': 122, 'mpeke': 123, 'muceere': 124, 'mucungwa': 125, 'mulimi': 126, 'munyeera': 127, 'muwogo': 128,
      'nakavundira': 129, 'nambaale': 130, 'namuginga': 131, 'ndwadde': 132, 'nfukirira': 133, 'nnakati': 134, 'nnasale beedi': 135, 'nnimiro': 136, 'nnyaanya': 137, 'npk': 138, 'nursery bed': 139,
      'obulimi': 140, 'obulwadde': 141, 'obumonde': 142, 'obusaanyi': 143, 'obutunda': 144, 'obutungulu': 145, 'obuwuka': 146, 'okufukirira': 147, 'okufuuyira': 148, 'okugimusa': 149, 'okukkoola': 150,
      'okukungula': 151, 'okulima': 152, 'okulimibwa': 153, 'okunnoga': 154, 'okusaasaana': 155, 'okusaasaanya': 156, 'okusiga': 157,
      'okusimba': 158, 'okuzifuuyira': 159, 'olusuku': 160, 'omuceere': 161, 'omucungwa': 162, 'omulimi': 163, 'omulimisa': 164, 'omusiri': 165, 'omuyembe': 166,
      'onion': 167, 'orange': 168, 'pampu': 169, 'passion fruit': 170, 'pawpaw': 171, 'pepper': 172, 'plant': 173, 'plantation': 174, 'ppaapaali': 175, 'pumpkin': 176, 'rice': 177, 'seed': 178,
      'sikungula': 179, 'sow': 180, 'spray': 181, 'spread': 182, 'suckers': 183, 'sugarcane': 184, 'sukumawiki': 185, 'super grow': 186, 'sweet potatoes': 187, 'tomatoes': 188, 'vegetables': 189,
      'watermelon': 190, 'weeding': 191, 'worm': 192}
        
        self.idx2labels = {k:v for v,k in self.labels2idx.items()}
        self.samples = path_audio #+ path_augment
        self.max_length = max_length # 99% are shorter than 3 sec
        self.resample_freq=resample_freq
        self.num_class = num_class

    
    def load_raw_audio(self, x):
        signal_f = np.zeros((self.max_length*self.resample_freq)).astype(np.float32)
        signal, sr_orig = librosa.load(x, sr=self.resample_freq)

        if sr_orig != self.resample_freq:
            signal = librosa.resample(signal, orig_sr=sr_orig, target_sr=self.resample_freq, res_type="kaiser_best")
        
        shape = len(signal)
        
        signal = signal[:self.max_length*self.resample_freq]
        signal_f[:len(signal)] = signal
        return signal_f.astype(np.float32)
    

    def normalise(self, x):
        """
        Parameters
        ----------
        x : numpy array
            DESCRIPTION.

        Returns
        -------
        x_norm : Normalise signal between 0 and 1 value
            DESCRIPTION.

        """
        min_ = x.min()
        max_ = x.max()
        if min_ == max_:
            x_norm = np.zeros(x.shape).astype(np.float32)
        else:
            x_norm =  (x-min_)/(max_ - min_)
        return x_norm
    
    def __getitem__(self, index):
        
        l = []    
        # load signal
        signal_raw =  self.load_raw_audio(self.samples[index] )

        # add Environment Noise
        
       
        l.append( torch.tensor(signal_raw) )
        l.append(torch.tensor(index))
        return tuple(l)
    
    def __len__(self):
        return len(self.samples)   
    
## -- Fonction Loop  -- ##
    
def train_fn(model, dataloader, optimizer, loss_fn, cfg, accumulation=2, l_mixup=1.0,verbose=False):
    model.train()
    total_loss = 0.
    t=tqdm(dataloader, disable=not verbose )
    scaler = GradScaler()
    optimizer.zero_grad()
    N = 0.
    
    if l_mixup>0:
        mixup = Mixup(mixup_alpha=l_mixup, random_seed=SEED)
        
    for i, batch in enumerate(t):
        inputs, labels, indices = batch
        inputs = inputs.to(cfg.device, dtype=torch.float)
        labels = labels.to(cfg.device, dtype=torch.float)
        
        
        lambda_mixup = None
        if l_mixup>0:
            lambda_mixup = torch.as_tensor(mixup.get_lambda(batch_size=len(inputs))).to(cfg.device, dtype=torch.float)
            labels = do_mixup(labels, lambda_mixup)
            labels[labels>1.0] = 1.0
    
        
        with autocast(cfg.use_apex):
            outputs = model(inputs, lambda_mixup)
            loss = loss_fn(outputs, labels )
            
            N += len(inputs)
            if len(loss.shape) == 2:
              loss = loss.sum(1).mean()
            else:
              loss = loss.mean()
        if torch.isnan(loss):
            print("loss error")
            print(torch.isnan(outputs).sum())
            loss[torch.isnan(loss)] = 0.0
            
            total_loss += loss.item()
        else:
            total_loss += loss.item() 


        if cfg.use_apex:
            loss = loss/accumulation
            scaler.scale(loss).backward()
        else:
            loss = loss/accumulation
            loss.backward()

        



        if (i+1)%accumulation == 0 or i-1 == len(t):
            if cfg.use_apex:
                scaler.step(optimizer)
    
                # Updates the scale for next iteration.
                scaler.update()
                optimizer.zero_grad()
            else:                
                optimizer.step()
                optimizer.zero_grad()
            
        
        t.set_description("Loss : {0}".format(total_loss/(i+1)))
        t.refresh()       

    return total_loss/N
        
def evals_fn(model, dataloader, optimizer, cfg, loss_fn, activation=False, verbose=False):
    total_loss = 0.
    t=tqdm(dataloader, disable=~verbose)
    y_true = []
    y_preds = []
    model.eval()
    device = cfg.device
    with torch.no_grad():
        for i, batch in enumerate(t):
            
            inputs,  labels, indices = batch

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs = model(inputs, None)
            if activation:
              output = torch.softmax(outputs, dim=-1)
            
            loss = loss_fn(outputs, labels )
            #print(loss.shape)
            if len(loss.shape) == 2:
              loss = loss.sum(1).mean()
            else:
              loss = loss.mean()     
            total_loss += loss.item()
            
            t.set_description("Loss : {0}".format(total_loss/(i+1)))
            t.refresh()
        
            y_true.append(labels.detach().cpu().numpy())
            y_preds.append( outputs.cpu().detach().numpy())
            
    return np.concatenate(y_preds), np.concatenate(y_true), total_loss/(i+1)


        
def inference_fn(model, dataloader, optimizer, cfg,activation=False, verbose=False):
    total_loss = 0.
    t=tqdm(dataloader,disable=~verbose)
    y_true = []
    y_preds = []
    model.eval()
    device = cfg.device
    with torch.no_grad():
        for i, batch in enumerate(t):
            
            inputs,  indices = batch

            inputs = inputs.to(device, dtype=torch.float)

            outputs = model(inputs, None)
            y_preds.append( outputs.cpu().detach().numpy())
            
    return np.concatenate(y_preds)


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)
    
    
def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out


def do_mixup_length(x):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = torch.stack( (x[0 :: 2] , x[1 :: 2]) ).max(0)[0]
    return out


## -- FONCTION DE COUTS -- ##

def bce(outputs, targets):
    eps = 1e-5
    p1=targets*(torch.log(outputs+eps))
    p0=(1-targets)*torch.log(1-outputs+eps)
    
    loss = p0 + p1
    return -loss


def get_additional_data(path):
    labels = []
    all_paths = []
    for name in os.listdir(path):
        pname = os.path.join(path, name)
        for filename in os.listdir(pname):
            fname = os.path.join(pname, filename)
            all_paths.append(fname)
            labels.append(name)
            
    return all_paths, labels


class Config():
    def __init__(self):
        self.num_class = 193
        self.resample_freq=48000
        self.max_length=3 # seconds
        self.device = "cuda:0"
        self.use_apex =True
        self.verbose=False
        self.epochs = 205
        self.accumulation = 1
        self.batch_size = 16
        self.l_mixup=1.0
        self.background = False
        self.kfold=5
        self.name = "model_pl_B"
        self.save_name = f"{self.name}-GIZ-{self.kfold}kfolds" 
import time
cfg = Config()

labels2idx = {'Pump': 0, 'Spinach': 1,  'abalimi': 2,  'afukirira': 3,  'agriculture': 4, 'akammwanyi': 5,  'akamonde': 6, 'akasaanyi': 7, 'akatunda': 8, 'akatungulu': 9,
      'akawuka': 10, 'amakoola': 11, 'amakungula': 12, 'amalagala': 13, 'amappapaali': 14, 'amatooke': 15, 'banana': 16, 'beans': 17, 'bibala': 18, 'bulimi': 19, 'butterfly': 20, 'cabbages': 21,
      'cassava': 22, 'caterpillar': 23, 'caterpillars': 24, 'coffee': 25, 'crop': 26, 'ddagala': 27, 'dig': 28, 'disease': 29, 'doodo': 30, 'drought': 31, 'ebbugga': 32, 'ebibala': 33, 'ebigimusa': 34,
      'ebijanjaalo': 35, 'ebijjanjalo': 36, 'ebikajjo': 37, 'ebikolo': 38, 'ebikongoliro': 39, 'ebikoola': 40, 'ebimera': 41, 'ebinyebwa': 42, 'ebirime': 43, 'ebisaanyi': 44, 'ebisooli': 45,
      'ebisoolisooli': 46, 'ebitooke': 47, 'ebiwojjolo': 48, 'ebiwuka': 49, 'ebyobulimi': 50, 'eddagala': 51, 'eggobe': 52, 'ejjobyo': 53, 'ekibala': 54, 'ekigimusa': 55, 'ekijanjaalo': 56,
      'ekikajjo': 57, 'ekikolo': 58, 'ekikoola': 59, 'ekimera': 60, 'ekirime': 61, 'ekirwadde': 62, 'ekisaanyi': 63, 'ekitooke': 64, 'ekiwojjolo': 65, 'ekyeya': 66, 'emboga': 67, 'emicungwa': 68,
      'emisiri': 69, 'emiyembe': 70, 'emmwanyi': 71, 'endagala': 72, 'endokwa': 73, 'endwadde': 74, 'enkota': 75, 'ennima': 76, 'ennimiro': 77, 'ennyaanya': 78, 'ensigo': 79, 'ensiringanyi': 80, 'ensujju': 81,
      'ensuku': 82, 'ensukusa': 83, 'enva endiirwa': 84, 'eppapaali': 85, 'faamu': 86, 'farm': 87, 'farmer': 88, 'farming instructor': 89, 'fertilizer': 90, 'fruit': 91, 'fruit picking': 92,
      'garden': 93, 'greens': 94, 'ground nuts': 95, 'harvest': 96, 'harvesting': 97, 'insect': 98, 'insects': 99, 'irish potatoes': 100, 'irrigate': 101, 'kaamulali': 102, 'kasaanyi': 103, 'kassooli': 104,
      'kikajjo': 105, 'kikolo': 106, 'kisaanyi': 107, 'kukungula': 108, 'leaf': 109, 'leaves': 110, 'lumonde': 111, 'lusuku': 112, 'maize': 113, 'maize stalk borer': 114, 'maize streak virus': 115, 'mango': 116, 'mangoes': 117, 'matooke': 118,
      'matooke seedlings': 119, 'medicine': 120, 'miceere': 121, 'micungwa': 122, 'mpeke': 123, 'muceere': 124, 'mucungwa': 125, 'mulimi': 126, 'munyeera': 127, 'muwogo': 128,
      'nakavundira': 129, 'nambaale': 130, 'namuginga': 131, 'ndwadde': 132, 'nfukirira': 133, 'nnakati': 134, 'nnasale beedi': 135, 'nnimiro': 136, 'nnyaanya': 137, 'npk': 138, 'nursery bed': 139,
      'obulimi': 140, 'obulwadde': 141, 'obumonde': 142, 'obusaanyi': 143, 'obutunda': 144, 'obutungulu': 145, 'obuwuka': 146, 'okufukirira': 147, 'okufuuyira': 148, 'okugimusa': 149, 'okukkoola': 150,
      'okukungula': 151, 'okulima': 152, 'okulimibwa': 153, 'okunnoga': 154, 'okusaasaana': 155, 'okusaasaanya': 156, 'okusiga': 157,
      'okusimba': 158, 'okuzifuuyira': 159, 'olusuku': 160, 'omuceere': 161, 'omucungwa': 162, 'omulimi': 163, 'omulimisa': 164, 'omusiri': 165, 'omuyembe': 166,
      'onion': 167, 'orange': 168, 'pampu': 169, 'passion fruit': 170, 'pawpaw': 171, 'pepper': 172, 'plant': 173, 'plantation': 174, 'ppaapaali': 175, 'pumpkin': 176, 'rice': 177, 'seed': 178,
      'sikungula': 179, 'sow': 180, 'spray': 181, 'spread': 182, 'suckers': 183, 'sugarcane': 184, 'sukumawiki': 185, 'super grow': 186, 'sweet potatoes': 187, 'tomatoes': 188, 'vegetables': 189,
      'watermelon': 190, 'weeding': 191, 'worm': 192}


# Options for Logmel
mel_bins = 64
fmin = 20
fmax = 24000
window_size = 1102
hop_size = 99
early_stopping = 40
audioset_classes_num = cfg.num_class

augmentations = ["noise"]

pseudo_labels=pd.read_csv("submission_ensemblingv5.csv")
train = pd.read_csv("Train.csv")
test = pd.read_csv("SampleSubmission.csv")

train["fn"] = train["fn"].apply(lambda x: x.replace("audio_files", "audio_files-48000"))
test["fn"] = test["fn"].apply(lambda x: x.replace("audio_files", "audio_files-48000"))
pseudo_labels["fn"] = pseudo_labels["fn"].apply(lambda x: x.replace("audio_files", "audio_files-48000"))


loss_fnt =nn.BCEWithLogitsLoss(reduction="none") #  CELoss#bce#
loss_bce = nn.BCEWithLogitsLoss(reduction="none")

paths_add, labels_add = get_additional_data("AdditionalUtterances-48000/latest_keywords")

paths_add2, labels_add2 = get_additional_data("nlp_keywords-48000")
    
skf = StratifiedKFold(n_splits=cfg.kfold, random_state=42, shuffle=True)
paths = np.array(train.fn.values.tolist() + paths_add + paths_add2)
targets_names = np.array(train.label.values.tolist() + labels_add+ labels_add2)

len(paths), len(targets_names)
dataset_test = AudioGeneratorDatasetTest(test.fn.values.tolist(), resample_freq=cfg.resample_freq, max_length=cfg.max_length, num_class=cfg.num_class )

test_dataloader = DataLoader(dataset_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
columns_labels = [dataset_test.idx2labels[i] for i in range(cfg.num_class)]

if __name__ == "__main__":
    oof_preds = []
    oof_targets = []
    oof_loss = []
    
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(paths)), targets_names)):
        print(f"## FOLD {fold}")
        start = time.time()
        dataset = AudioGeneratorDataset(paths[train_idx].tolist(), targets_names[train_idx],  
                                        resample_freq=cfg.resample_freq, max_length=cfg.max_length, 
                                      augmentation=augmentations, validation=False, num_class=cfg.num_class, pseudo_labels=(pseudo_labels.fn.values.tolist(), pseudo_labels[columns_labels].values) )
        
        dataset_val = AudioGeneratorDataset(paths[val_idx].tolist(), targets_names[val_idx],
                                          resample_freq=cfg.resample_freq, max_length=cfg.max_length, augmentation=[], validation=True,
                                          num_class=cfg.num_class )
    
    
    
    
        #train_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, drop_last=True, sampler=AgrinetDatasetSampler( dataset))
        train_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, drop_last=True)
    
    
        val_dataloader = DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=1)
    
        ## -- LOAD MODEL -- ##

    
        backbone = EfficientNet.from_pretrained('efficientnet-b6', num_classes=cfg.num_class)
        model = AudioClassifier(backbone, cfg.resample_freq, window_size, hop_size, mel_bins, fmin, fmax, cfg.num_class).to(cfg.device)

    
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False, weight_decay=1e-5) 
        reducer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, verbose=True, min_lr=1e-6)
        best_score_bce= np.inf
        best_score_ce = np.inf
        count = 0
        for e in range(cfg.epochs):
            train_fn(model, train_dataloader, optimizer, loss_fnt, cfg, accumulation=cfg.accumulation, l_mixup=cfg.l_mixup, verbose=cfg.verbose)
    
            preds5, targets5, val_loss_bce =evals_fn(model, val_dataloader, optimizer, cfg, loss_bce, activation=False)
    
    
            reducer.step(val_loss_bce)
            if best_score_bce > val_loss_bce :
                count = 0
                best_score_bce = val_loss_bce
                torch.save(model.state_dict(), cfg.save_name + f"-fold{fold}-bce.pth")
                print("epoch: {}/{} bceloss - score improved: {} time: {}".format(e + 1, cfg.epochs, round(val_loss_bce, 4), round(time.time()-start)))
            else:
                count += 1
                print("epoch: {}/{} bceloss score not improve: {}  Best: {} time: {}". format(e + 1, cfg.epochs, round(val_loss_bce, 4), round(best_score_bce, 4), round(time.time()-start))) 
            if count == early_stopping:
                print(" ========= Early stopping ========= ") 
                break
    
        model.load_state_dict(torch.load(cfg.save_name + f"-fold{fold}-bce.pth"))
        preds, targets, val_loss_bce =evals_fn(model, val_dataloader, optimizer, cfg, loss_bce, activation=False)
        oof_preds.append(preds)
        oof_targets.append(targets)
        oof_loss.append(val_loss_bce)
    oof_preds = np.concatenate(oof_preds, axis=0)
    oof_targets = np.concatenate(oof_targets)
    
    
    print("final loss : ", oof_loss)
    
    results = nn.BCEWithLogitsLoss(reduction="none")(torch.as_tensor(oof_preds), torch.as_tensor(oof_targets)).sum(1).mean().item()
    print(results)
    
    
    preds_test = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(paths)), targets_names)):
        print(f"## FOLD {fold}")
        model.load_state_dict(torch.load(cfg.save_name + f"-fold{fold}-bce.pth"))
        preds_test_fold =inference_fn(model, test_dataloader, optimizer, cfg)
        preds_test.append(torch.sigmoid(torch.as_tensor(preds_test_fold)).numpy())
    preds_test = np.stack(preds_test)
    ids = np.array([dataset_test.labels2idx[x] for x in test.columns[1:]])
    
    preds_test_withids = preds_test.mean(0)[:, ids]
    test[test.columns[1:]] = preds_test_withids
    
    
    test = pd.read_csv("SampleSubmission.csv")
    test[test.columns[1:]] = preds_test_withids
    if cfg.l_mixup == 0:
        test.to_csv(f"{cfg.name}-{cfg.kfold}folds-CV-seed{SEED}-bs{cfg.batch_size}.csv", index=False)
    else:
        test.to_csv(f"{cfg.name}-{cfg.kfold}folds-CV-seed{SEED}-bs{cfg.batch_size}-mixup.csv", index=False)
      

