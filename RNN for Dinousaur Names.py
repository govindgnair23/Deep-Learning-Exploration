# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 19:26:02 2019

@author: learningmachine
"""
##################Import required packages#################
import numpy as np
from sklearn.preprocessing import OneHotEncoder


#############################Read in and process input data#################
dino_names = open('dinos.txt','r').read()
dino_names = dino_names.lower()
chars = list(set(dino_names))
data_size, vocab_size = len(dino_names),len(chars)
print('There are %d total characters and %d unique characters in the data.'%(data_size,vocab_size))


###Dictionaries mapping character to index and vice versa
char_to_idx = {ch:i for i,ch in enumerate(sorted(chars))}
ix_to_char =  {i:ch for i,ch in enumerate(sorted(chars))}
print(ix_to_char)


###Encoded version of din names
encoded_dino_names = np.array([char_to_idx[char] for char in dino_names])
encoded_dino_names[:100]

####Create one hot encoder for each character in the dictionary###
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(np.array(list(char_to_idx.values())).reshape(-1,1))


