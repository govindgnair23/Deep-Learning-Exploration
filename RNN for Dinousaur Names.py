# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 19:26:02 2019

@author: learningmachine
"""
##################Import required packages#################
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader



#############################Read in and process input data#################
dino_names = open('dinos.txt','r').read()
dino_names = dino_names.lower().split('\n')
#Use a period as ebd of name token
dino_names = [dino_name+"." for dino_name in dino_names]
chars = list(set(''.join(dino_names)))
data_size, vocab_size = len(dino_names),len(chars)
print('There are %d total characters and %d unique characters in the data.'%(data_size,vocab_size))

####Longest dinosaur name####
longest_name   = max(dino_names,key=len)
longest_name_len = len(longest_name)




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

#### Create train and test set for use in dataloader
####I will pad the end of each dinosaur name with periods so that each
####dinosaur name is as long as Longest_name_len

dino_name_deficit = [27 - len(dino) for dino in dino_names]
train_names = [ dino_name + '.'*extra_len for dino_name,extra_len in zip(dino_names,dino_name_deficit)]
##Create test set by using charcaters 2 to 27 and adding a period
test_names = [train_name[1:]+'.' for train_name in train_names]


