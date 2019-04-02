# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 19:26:02 2019
Train the model using 1 example at a time
@author: learningmachine
"""
##################Import required packages#################
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch import nn
import torch
#from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import os
import random
import matplotlib.pyplot as plt

os.chdir('C:/Users/learningmachine/Documents/Learning and Development/Miscellaneous/Deep Learning Exploration/Character RNN')

#############################Read in and process input data#################
dino_names = open('dinos.txt','r').read()
dino_names = dino_names.lower().split('\n')
random.shuffle(dino_names)


chars = sorted(list(set(''.join(dino_names))) + ['.']) # Period acts as <EOS> token
data_size, vocab_size = len(dino_names),len(chars)
print('There are %d total characters and %d unique characters in the data.'%(data_size,vocab_size))

###Dictionaries mapping character to index and vice versa
char_to_idx = {ch:i for i,ch in enumerate(sorted(chars))}
ix_to_char =  {i:ch for i,ch in enumerate(sorted(chars))}
print(ix_to_char)

##Note that <EOS> is denoted by 0

###Encoded version of dino names
encoded_dino_names = list(map(lambda x: [char_to_idx[char] for char in x], dino_names))
encoded_dino_arrays = list(map(lambda x: np.array(x),encoded_dino_names))

###Split into training(70%) and validation(30%) sets
split_idx = int(len(encoded_dino_arrays)*0.7)
train_dinos = encoded_dino_names[:split_idx]
valid_dinos = encoded_dino_names[split_idx:]


####Create one hot encoder for each character in the dictionary###
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(np.array(list(char_to_idx.values())).reshape(-1,1))
#one_hot_encoder.transform(np.array([1,2,3]).reshape(-1,1)).toarray()

##Target is obtained by shifting the inputs forward by 1 time step###
#input_arr = encoded_dino_array
#target_arr =  np.column_stack((input_arr[:,1:],np.zeros(input_arr.shape[0]))).astype(int)

#Create training and validation data
#val_idx = int(len(input_arr)*(1-0.1))
#train_X,train_y = input_arr[:val_idx],target_arr[:val_idx]
#valid_X,valid_y = input_arr[val_idx:],target_arr[val_idx:]




#print("\t\t Feature Shapes:")
#print("Train_set:\t\t{}".format(train_X.shape))
#print("\nValidation Set: \t{}".format(valid_X.shape))

###DataLoaders and Batching
#train_data = TensorDataset(torch.from_numpy(train_X),torch.from_numpy(train_y))
#valid_data = TensorDataset(torch.from_numpy(valid_X),torch.from_numpy(valid_y))

#batch_size = 2

#train_loader = DataLoader(train_data,batch_size=batch_size)
#valid_loader = DataLoader(valid_data,batch_size=batch_size)


# check if GPU is available
train_on_gpu = torch.cuda.is_available()


def get_train_example():
    
    
    '''Create a generator that returns successive dino names from training set.
       Two values are returned, a one hot encoded input and target
       
       Arguments
       ---------
       None
       
    '''
     
    
    for i in range(len(train_dinos)):
        x = np.array(train_dinos[i])
        y = np.append(x[1:],0) #Pad with EOS chracter denoted by 0
        ###Convert to one hot encoded arrays
        x_oh = one_hot_encoder.transform(x.reshape(-1,1)).toarray()
        
        yield x_oh,y
        
def get_valid_example():
    
    
    '''Create a generator that returns successive dino names from training set.
       Two values are returned, a one hot encoded input and target
       
       Arguments
       ---------
       None
       
    '''
     
    
    for i in range(len(valid_dinos)):
        x = np.array(valid_dinos[i])
        y = np.append(x[1:],0) #Pad with EOS chracter denoted by 0
        ###Convert to one hot encoded arrays
        x_oh = one_hot_encoder.transform(x.reshape(-1,1)).toarray()
        
        yield x_oh,y    


    
####Define Neural  Net with LSTMS##########
        
class CharRNN(nn.Module):
    
    
    def __init__(self,tokens,n_hidden=128,n_layers=2,drop_prob=0.1,lr=0.001):
         super().__init__()
         self.drop_prob = drop_prob
         self.n_layers = n_layers
         self.n_hidden = n_hidden
         self.lr =lr
         
         #creating character dictionaries
         self.chars = tokens
         self.int2char = {i:ch for i,ch in enumerate(sorted(chars))}
         self.char2int = {ch:i for i,ch in enumerate(sorted(chars))}
         
         ##Define LSTM
         self.lstm = nn.LSTM(len(self.chars),n_hidden,n_layers,dropout = drop_prob)
         #Define a drop out layer
         self.dropout = nn.Dropout(drop_prob)
         ##Fully connected layer
         self.fc = nn.Linear(n_hidden,len(self.chars))
         
        
    def forward(self,x,hidden):
        
        ''' Forward pass through the network.
            The inputs are x and the hidden cell/state  is hidden'''
        
        #Pass input and incoming hidden through lstm
        lstm_out, hidden = self.lstm(x,hidden)
        
        #Pass through dropout layer
        out = self.dropout(lstm_out)
        
        # Stack up LSTM outputs using view
        out = lstm_out.contiguous().view(-1, self.n_hidden)
        
        
      
        ###Pass through fully connected layer
        out = self.fc(out)
        
        return(out,hidden)
        
        
    
    def init_hidden(self):
        '''Initalize hidden state'''
        
        weight = next(self.parameters()).data
        
        if(train_on_gpu):
            hidden = (weight.new(self.n_layers,1,self.n_hidden).zero_().cuda(),
                       weight.new(self.n_layers,1,self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers,1, self.n_hidden).zero_(),
                      weight.new(self.n_layers,1, self.n_hidden).zero_())
        
        return(hidden) 
    


################Define a plain vanilla RNN####

class CharRNN2(nn.Module):
    def __init__(self,tokens,hidden_size,drop_prob = 0.1, lr = 0.001):
        super(CharRNN2,self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.lr = lr
        self.input_size = len(tokens)
        self.output_size = len(tokens)
        #creating character dictionaries
        self.chars = tokens
        self.int2char = {i:ch for i,ch in enumerate(sorted(chars))}
        self.char2int = {ch:i for i,ch in enumerate(sorted(chars))}
        
        #Creating layers
        self.i2h = nn.Linear(self.input_size+hidden_size,hidden_size)
        self.i2o = nn.Linear(self.input_size+hidden_size,self.output_size)
        self.o2o = nn.Linear(self.output_size + hidden_size, self.output_size)
        self.dropout = nn.Dropout(drop_prob)
        
        
    
    def forward(self,x,hidden):
        input_combined = torch.cat((x,hidden),dim = 1)
        hidden = F.relu(self.i2h(input_combined))
        output = F.relu(self.i2o(input_combined))
        output_combined = torch.cat((hidden,output),dim = 1)
        output = F.relu(self.o2o(output_combined))
        output = self.dropout(output)
        return(output,hidden)
        
    def init_hidden(self):
        
        if(train_on_gpu):
            return(torch.zeros(1,self.hidden_size).cuda())
        else:
            return(torch.zeros(1,self.hidden_size))
        




 ###Below function borrowed from Udacity course   
def train(net,epochs=10,lr =0.001,clip =5,print_every =10,plot_every =100,model_type = 'SimpleRNN'):
    
    
    '''
         net : CharRNN Network
         input_data: text data to train network
         target_data: text data to predict
         epochs: No of epochs to train
         batch_size : No of dinosaur names to train at a time
         seq_length: No of character steps per batch(27)/dinosaur name
         lr : Learning rate
         clip: gradient clipping
         print_every: No of steps for printing and training and validation loss
    '''
    
    
    opt = torch.optim.Adam(net.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()
    

    
    if(train_on_gpu):
        net.cuda()
    
    net.train() # Set to training mode
    
    counter = 0
    plot_loss_list = []
    
    for e in range(epochs):
        #initialize hidden state
        if(model_type != 'SimpleRNN'):
            h = net.init_hidden()
        plot_loss = 0
        for inputs,targets in get_train_example():
            inputs = torch.from_numpy(inputs).unsqueeze(1).to(dtype =torch.float32)
            targets = torch.from_numpy(targets).unsqueeze(-1).long()
            
            
            counter+=1
            
            
                
            if (train_on_gpu):
                inputs, targets = inputs.cuda(),targets.cuda().long()
                
            ##Create new variables for hidden state as we don't
            ##want to backprop between batches
            
            if(model_type == 'SimpleRNN'):
                h = net.init_hidden()
            else:
                h = tuple([each.data for each in h])
            
            #zero accumulated gradients
            net.zero_grad()
            
            loss = 0
            for i in range(len(inputs[0])):
                 if model_type == 'SimpleRNN':
                     input_i = inputs[i]
                 else:
                    input_i = inputs[i].unsqueeze(0) 
                 output,h = net(input_i,h)
                 l = criterion(output,targets[i])
                 loss += l
            
            plot_loss += loss
            loss.backward()
            
            #Perform gradient clipping to prevent gradient explosion
            nn.utils.clip_grad_norm(net.parameters(),clip)
            opt.step()
            
            if counter%plot_every ==0:
                plot_loss_list.append(plot_loss/plot_every)
                
            
            
            
            ##Get loss stats from validation data
            
            if counter%print_every==0:
                if(model_type != 'SimpleRNN'):
                    val_h = net.init_hidden()
                val_losses =[]
                net.eval() # Set to evaluation mode
                ##Get validation data
                for inputs,targets in get_valid_example():
                    inputs = torch.from_numpy(inputs).unsqueeze(1).to(dtype =torch.float32)
                    targets = torch.from_numpy(targets).unsqueeze(-1).long()
                    
                
                    if(model_type == 'SimpleRNN'):
                        val_h = net.init_hidden()
                    else:
                        val_h = tuple([each.data for each in val_h])##Create new hidden state for each batch
                    
                    if train_on_gpu:
                        inputs,targets = inputs.cuda(), targets.cuda().long()
                    
                    val_loss = 0
                    
                    for i in range(len(inputs[0])):
                        if model_type == 'SimpleRNN':
                            input_i = inputs[i]
                        else:
                            input_i = inputs[i].unsqueeze(0) 
                        output,h = net(input_i,h)
                        l = criterion(output,targets[i])
                        val_loss += l
                         
                    val_losses.append(val_loss)
                    
                
                net.train()
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean([x.item() for x in val_losses])))
        
        
        
    return(plot_loss_list)        
 ####Instantiating the model##########
n_hidden= 128
n_layers =2
 
#net = CharRNN(chars,n_hidden,n_layers) 
net = CharRNN2(chars,n_hidden)
net_lstm = CharRNN(chars,n_hidden,n_layers) 

###############Train model1 ###################

n_epochs = 5
plot_loss_list_m1 = train(net,epochs= n_epochs,lr = 0.001,clip =5,print_every =10,model_type = 'SimpleRNN')

####Plot training losses

plt(plot_loss_list_m1)

###############Train model2

n_epochs = 5
plot_loss_list_m2 = train(net_lstm,epochs= n_epochs,lr =0.05,clip =5,print_every =10,model_type = 'LSTM')
 
####Plot training losses

plt(plot_loss_list_m2)
 
############Save model for later use###########
model_name = 'simple_rnn_20epoch.net'

checkpoint = {'hidden_size':net.hidden_size,
              'tokens': net.chars,
              'state_dict':net.state_dict()}

with open(model_name,'wb') as f:
    torch.save(checkpoint,f)
    
model_name = 'lstm_rnn_20epoch.net'
    
checkpoint = {'n_hidden':net_lstm.n_hidden,
              'n_layers': net_lstm.n_layers,
              'state_dict':net_lstm.state_dict(),
              'tokens': net_lstm.chars}

with open(model_name,'wb') as f:
    torch.save(checkpoint,f)
##################################################3
###Function to predict next character given an input character###
## Returns the predicted character and hidden state##############
    
def predict(net,char,h=None,top_k =None,model_type = 'SimpleRNN'):
    
    #Get inputs in one hot encoded tensor form
    x = np.array([[net.char2int[char]]])
    x = one_hot_encoder.transform(x).toarray()
    
    if(model_type != 'SimpleRNN'):
        inputs = torch.from_numpy(x).unsqueeze(1).to(dtype = torch.float32)
    else:
        inputs = torch.from_numpy(x).to(dtype = torch.float32)
    if(train_on_gpu): 
        inputs = inputs.cuda()
        
    #detach hidden state from history
    if(model_type != 'SimpleRNN'):
        h = tuple(each.data for each in h)
    #Get output of the model
    out,h = net(inputs,h)

    #Get charcater probabilties
    p = F.softmax(out,dim=1).data
    if(train_on_gpu): 
        p = p.cpu() # move to cpu
        
    #get top characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else: 
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()
        
    # select next likely character
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch,p = p/p.sum())
    
    #return enocded value of next character and hidden state
    return(net.int2char[char],h)
    

###Priming and generating text###
def sample(net,max_length =27,prime ='t', top_k =None,model_type= 'SimpleRNN'): 
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()
        
    net.eval() #eval mode
    
    ##Run through priming characters
    chars = [ch for ch in prime]
    
    h = net.init_hidden()
    for ch in prime:
        char,h = predict(net,ch,h,top_k = top_k,model_type =model_type)
        
    chars.append(char)
    
    ii = len(prime) # Length of dinosaur name
    ##Now pass in previous charcaters to get new one
    while(char != "." and ii <= max_length):
        char,ch = predict(net,chars[-1],h, top_k = top_k,model_type=model_type)
        chars.append(char)
        ii +=1
    
    return ''.join(chars)

print(sample(net,max_length =27,prime ='m',top_k =3,model_type= 'SimpleRNN'))
print(sample(net_lstm,max_length =10,prime ='m',top_k =3,model_type= 'LSTM'))  
    
    
        
    
        
        
    
    
    
    
    
    
         
         
        
        
            
    
 
    
    
    
    












