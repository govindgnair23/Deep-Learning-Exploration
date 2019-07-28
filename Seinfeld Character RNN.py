# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:02:17 2019

@author: learningmachine
"""

############Script to generate a monlogue for any of the four major Seinfeld caracters: Jerry,George,Elaine &
############Kramer###########################################################################################
from collections import Counter
import string
import numpy as np
import pandas as pd
import random
import re
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from argparse import Namespace
from tqdm import tqdm
##############################Prepare dataset#################

####Read in file###
with open('Seinfeld_Scripts.txt','r',encoding= "utf8") as f:
    text = f.readlines()

text2 = [t.split(":") for t in text if t != '\n']   
characters = [l[0] for l in text2]


counter = Counter(characters)
counter.most_common(10) 


###Get only dialogues from Jerry,Kramer,Elaine and George    
jkeg = {'jerry':[],'george':[],'elaine':[],'kramer':[]}

##Table to strip punctuations:
table = str.maketrans("","",string.punctuation)



for l in text2:
    char = l[0]
    #Replace anything in braces
    dialogue = re.sub(r'\(.*\)','',l[1])
    #Strip punctuations and end of line character
    dialogue = dialogue.translate(table).rstrip()
    if char  == 'jerry':
        jkeg['jerry'].append(dialogue)
    elif char == 'george':
        jkeg['george'].append(dialogue)
    elif char == 'elaine':
        jkeg['elaine'].append(dialogue)
    elif char == 'kramer':
        jkeg['kramer'].append(dialogue)
            
#Convert to dataframe
seinfeld_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in jkeg.items()]))        

##COnvert to long format
seinfeld_df["id"] = seinfeld_df.index
seinfeld_df = pd.melt(seinfeld_df,id_vars="id",value_vars = ['jerry','george','elaine','kramer'],
                      var_name = "character",value_name = "line")

#####Assign lines to train and validation sets####
seinfeld_df.dropna(inplace=True)
category_counts = seinfeld_df.character.value_counts()

 ###Ensure each character has proportionate representation in train and validation splits
split = []
random.seed(1)

for category,n_obs in category_counts.items():
    n_train = int(0.7*n_obs)
    n_val = n_obs - n_train
    category_split = ['train']*n_train + ['val']*n_val
    random.shuffle(category_split)
    split.extend(category_split)

seinfeld_df['split'] = split         

####Create a vocabulary class #######
class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""
    
    def __init__(self,token_to_idx = None):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
        """
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx:token for token,idx in self._token_to_idx.items()}
        
        
    def add_token(self,token):
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            try:
                index = max(self._token_to_idx.values())+1
            except ValueError:
                index = 0 # Occurs only when dictionary has no elements
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        
        return index
    
    
    def add_many(self,tokens):
        """Add a list of tokens into the Vocabulary
        
        Args:
            tokens (list): a list of string tokens
        Returns:
            indices (list): a list of indices corresponding to the tokens
        """
        
        return [self.add_token(token) for token in tokens]
    
    
    def lookup_index(self,index):
        """Return the token associated with the index
        
        Args: 
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) not in Vocabulary"% index)
        return self._idx_to_token[index]
    
    def lookup_token(self,token):
        """Retrieve the index associated with the token 
        
        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        """
        return self._token_to_idx[token]
        
    
    
    def __str__(self):
        return "<Vocabulary(size = %d)>" % len(self)
    
    def __len__(self):
        return len(self._token_to_idx)
    
###Create a Sequence Vocabulary class#######

class SequenceVocabulary(Vocabulary):      
    def __init__(self,token_to_idx=None,unk_token="<UNK>",mask_token = "<MASK>",
                 begin_seq_token = "<BEGIN>",end_seq_token = "<END>"):
        super(SequenceVocabulary,self).__init__(token_to_idx)
        
        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token
        
        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)
        
    def lookup_token(self,token):
        """Retrieve the index associated with the token 
          or the UNK index if token isn't present.
        
        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary) 
              for the UNK functionality 
        """
        return self._token_to_idx.get(token,self.unk_index)
        
        
########################Create a vectorizer that converts text to a vector of numbers

class SeinfeldVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def __init__(self,char_vocab,seinfeld_char_vocab):
        """
        Args:
            char_vocab (SequenceVocabulary): maps alphabets to integers
            seinfeld_char_vocab (Vocabulary): maps seinfeld characters  to integers
        """
        
        self.char_vocab = char_vocab
        self.seinfeld_char_vocab = seinfeld_char_vocab
        
    
    def vectorize(self,line,vector_length = -1):
        """Vectorize a surname into a vector of observations and targets
        
        The outputs are the vectorized line split into two vectors:
            line[:-1] and line[1:]
        At each timestep, the first vector is the observation and the second vector is the target. 
        
        Args:
            surname (str): the surname to be vectorized
            vector_length (int): an argument for forcing the length of index vector
        Returns:
            a tuple: (from_vector, to_vector)
            from_vector (numpy.ndarray): the observation vector 
            to_vector (numpy.ndarray): the target prediction vector
        """
        
        indices = [self.char_vocab.begin_seq_index]
        indices.extend(self.char_vocab.lookup_token(token) for token in line)
        indices.append(self.char_vocab.end_seq_index)
        
        if vector_length < 0:
            vector_length = len(indices) - 1
            
        from_vector = np.zeros(vector_length,np.int64)
        from_indices = indices[:-1]
        from_vector[:len(from_indices)] = from_indices
        from_vector[len(from_indices):] = self.char_vocab.mask_index
        
        to_vector = np.zeros(vector_length,np.int64)
        to_indices = indices[1:]
        to_vector[:len(to_indices)] = to_indices
        to_vector[len(to_indices):] = self.char_vocab.mask_index
        
        
        return from_vector,to_vector
    
    
    @classmethod
    def from_dataframe(cls,seinfeld_df):
        """Instantiate the vectorizer from the dataset dataframe
        
        Args:
            seinfeld_df (pandas.DataFrame): the seinfeld dialogue dataset
        Returns:
            an instance of the SeinfeldVectorizer
        """
        
        char_vocab = SequenceVocabulary()
        seinfeld_char_vocab = Vocabulary()
        
        for index,row in seinfeld_df.iterrows():
            for char in set(row.line):
                char_vocab.add_token(char)
            seinfeld_char_vocab.add_token(row.character)
        
        return(cls(char_vocab,seinfeld_char_vocab))
        
        
###############Dataset###################

class SeinfeldDataset(Dataset):
    def __init__(self,seinfeld_df,vectorizer):
        """
        Args:
            surname_df (pandas.DataFrame): the dataset
            vectorizer (SurnameVectorizer): vectorizer instatiated from dataset
        """
        self.seinfeld_df = seinfeld_df
        self._vectorizer = vectorizer
        
        self.max_seq_length = max(map(len,self.seinfeld_df.line))+2
        
        self.train_df = self.seinfeld_df[self.seinfeld_df.split == 'train']
        self.train_size = len(self.train_df)
        
        self.val_df = self.seinfeld_df[self.seinfeld_df.split == 'val']
        self.val_size = len(self.val_df)
        
        self._lookup_dict = {'train':(self.train_df,self.train_size),
                             'val': (self.val_df,self.val_size)}
        
        self.set_split('train')
        
        
        
    @classmethod
    def load_dataset_and_make_vectorizer(cls,seinfeld_df):
        """Load dataset and make a new vectorizer from scratch
        
        Args:
            surname_df(pandas df): dataset
        Returns:
            an instance of SeinfeldDataset
        """
        return cls(seinfeld_df,SeinfeldVectorizer.from_dataframe(seinfeld_df))
        
            
    def get_vectorizer(self):
        """ returns the vectorizer"""
        return self._vectorizer
        
    def set_split(self,split = "train"):
        self._target_split = split
        self._target_df,self._target_size = self._lookup_dict[split]
        
        
    def __len__(self):
        return self._target_size
    
    def __getitem__(self,index):
        """ the primary entrypoint method for pytorch datasets
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point: (x_data, y_target, class_index)
        """
        
        row = self._target_df.iloc[index]
        
        from_vector,to_vector = self._vectorizer.vectorize(row.line,self.max_seq_length)
        character_index = self._vectorizer.seinfeld_char_vocab.lookup_token(row.character)
        
        return {'x_data': from_vector,
                'y_target': to_vector,
                'class_index':character_index}
        
    def get_num_batches(self,batch_size):
        
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self)//batch_size
    
    
def generate_batches(dataset,batch_size,shuffle=True,drop_last=True,
                     device = "cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    
    dataloader = DataLoader(dataset = dataset,batch_size=batch_size,
                            shuffle=shuffle, drop_last= drop_last)
    
    for data_dict in dataloader:
        out_data_dict = {}
        for name,tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict
        

###########The MOdel#############
        
class SeinfeldLineGenerationModel(nn.Module):   
    def __init__(self,char_embedding_size,char_vocab_size,num_seinfeld_chars,
                 rnn_hidden_size,batch_first=True,padding_idx=0,dropout_p = 0.5):
        """
        Args:
            char_embedding_size (int): The size of the character embeddings
            char_vocab_size (int): The number of characters to embed
            num_seinfeld_characters (int): The number of seinfeld characters to embed
            rnn_hidden_size (int): The size of the RNN's hidden state
            batch_first (bool): Informs whether the input tensors will 
                have batch or the sequence on the 0th dimension
            padding_idx (int): The index for the tensor padding; 
                see torch.nn.Embedding
            dropout_p (float): the probability of zeroing activations using
                the dropout method.  higher means more likely to zero.
        """
        
        super(SeinfeldLineGenerationModel,self).__init__()
        
        self.char_emb = nn.Embedding(num_embeddings = char_vocab_size,
                                     embedding_dim = char_embedding_size,
                                     padding_idx = padding_idx)
        
        
        self.seinfeld_char_emb = nn.Embedding(num_embeddings = num_seinfeld_chars,
                                              embedding_dim = rnn_hidden_size)
        
        
        self.rnn = nn.GRU(input_size = char_embedding_size,
                          hidden_size = rnn_hidden_size,
                          batch_first = batch_first)
        
        self.fc = nn.Linear(in_features = rnn_hidden_size,
                            out_features = char_vocab_size)
        
        self.dropout_p = dropout_p
        
      
    def forward(self,x_in, seinfeld_character_index,apply_softmax = False):
        """The forward pass of the model
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, max_seq_size)
            seinfeld_character_index (torch.Tensor): The index of the character for each data point
                Used to initialize the hidden state of the RNN
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, char_vocab_size)
        """
        
        x_embedded = self.char_emb(x_in)
        seinfeld_character_embedded = self.seinfeld_char_emb(seinfeld_character_index).unsqueeze(0)
        
        y_out,_  = self.rnn(x_embedded,seinfeld_character_embedded)
        
        batch_size,seq_size,feat_size = y_out.shape
        
        y_out = y_out.contiguous().view(batch_size*seq_size,feat_size)
        
        y_out = self.fc(F.dropout(y_out,p=self.dropout_p))
        
        if apply_softmax:
            y_out = F.softmax(y_out,dim = 1)
        
        new_feat_size = y_out.shape[-1]
        y_out = y_out.view(batch_size,seq_size,new_feat_size)
        
        return y_out
    
def sample_from_model(model,vectorizer,sein_characters,sample_size=30,temperature= 1.0):
    """Sample a sequence of indices from the model
    
    Args:
        model (SeinfeldLineGenerationModel): the trained model
        vectorizer (SeinfeldVectorizer): the corresponding vectorizer
        sein_characters(list): a list of integers representing Seinfeld characters
        sample_size (int): the max length of the samples
        temperature (float): accentuates or flattens 
            the distribution. 
            0.0 < temperature < 1.0 will make it peakier. 
            temperature > 1.0 will make it more uniform
    Returns:
        indices (torch.Tensor): the matrix of indices; 
        shape = (num_samples, sample_size)
    """        
    
    num_samples = len(sein_characters)
    begin_seq_index = [vectorizer.char_vocab.begin_seq_index for _ in range(num_samples)]
    begin_seq_index = torch.tensor(begin_seq_index,dtype = torch.int64).unsqueeze(dim=1)
    indices = [begin_seq_index]
    sein_characters_indices = torch.tensor(sein_characters,dtype = torch.int64).unsqueeze(dim=0)
    h_t = model.seinfeld_char_emb(sein_characters_indices)
    
    for time_step in range(sample_size):
        x_t = indices[time_step]
        x_emb_t = model.char_emb(x_t)
        rnn_out_t,h_t = model.rnn(x_emb_t,h_t)
        prediction_vector = model.fc(rnn_out_t.squeeze(dim=1))
        probability_vector = F.softmax(prediction_vector/temperature,dim=1)
        indices.append(torch.multinomial(probability_vector,num_samples=1))
    
    indices = torch.stack(indices).squeeze().permute(1,0)
    return indices

def decode_samples(sampled_indices,vectorizer):
    '''
    Transform indices into the string for of a line
    
    Args:
        sampled_indices(torch.Tensor): the indices from sample_from_model
        vectorizer (SurnameVectorizer): the corresponding vector 

    '''
    decoded_lines = []
    vocab = vectorizer.char_vocab
    
    for sample_index in range(sampled_indices.shape[0]):
        line = ""
        for time_step in range(sampled_indices.shape[1]):
            sample_item = sampled_indices[sample_index,time_step].item()
            if sample_item == vocab.begin_seq_index:
                continue
            elif sample_item == vocab.end_seq_index:
                break
            else:
                line += vocab.lookup_index(sample_item)
                
        decoded_lines.append(line)
    
    return decoded_lines
    
    
    

###########################Training Routine##################
def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}


def update_train_state(args,model,train_state):
    """Handle the training state updates.
    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better
    
    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """
    
    #Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(),train_state['model_filename'])
        train_state['stop_early'] = False
    
    #Save model if performance improved    
    elif train_state['epoch_index'] >= 1:
        loss_tm1,loss_t = train_state['val_loss'][-2:]
        
        #If loss worsened
        if loss_t >= loss_tm1:
            #Update step
            train_state['early_stopping_step'] +=1
        #loss decreased
        else:
            #Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(),train_state['model_filename'])
                train_state['early_stopping_best_val'] = loss_t
                
            #Reset ealry stopping step
            train_state['early_stopping_step'] = 0
            
        #Stop early?
        train_state['stop_early'] = train_state['early_stopping_step'] >= \
                args.early_stopping_criteria
                
    return train_state

def normalize_sizes(y_pred,y_true):
    """Normalize tensor sizes
    
    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1,y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    
    return y_pred,y_true

def compute_accuracy(y_pred,y_true,mask_index):
    y_pred, y_true  = normalize_sizes(y_pred,y_true)
    
    _,y_pred_indices = y_pred.max(dim=1)
    
    correct_indices= torch.eq(y_pred_indices,y_true).float()
    valid_indices = torch.ne(y_true,mask_index).float()
    
    n_correct = (correct_indices*valid_indices).sum().item()
    n_valid = valid_indices.sum().item()
    
    return n_correct/n_valid


def sequence_loss(y_pred,y_true,mask_index):
    y_pred,y_true = normalize_sizes(y_pred,y_true)
    return F.cross_entropy(y_pred,y_true,ignore_index = mask_index)


##General Utilities

def set_seed_everywhere(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        
args = Namespace(
    # Data and Path information
    model_state_file="model.pth",
    save_dir="model_storage",
    # Model hyper parameters
    char_embedding_size=32,
    rnn_hidden_size=32,
    # Training hyper parameters
    seed=1337,
    learning_rate=0.001,
    batch_size=128,
    num_epochs=20,
    early_stopping_criteria=5,
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=True
)

args.device = torch.device("cuda" if args.cuda else "cpu")

#########Initialize model#####

dataset = SeinfeldDataset.load_dataset_and_make_vectorizer(seinfeld_df)

vectorizer = dataset.get_vectorizer()

model = SeinfeldLineGenerationModel(char_embedding_size = args.char_embedding_size,
                                    char_vocab_size = len(vectorizer.char_vocab),
                                    num_seinfeld_chars = len(vectorizer.seinfeld_char_vocab),
                 rnn_hidden_size = args.rnn_hidden_size,
                 batch_first=True,padding_idx= vectorizer.char_vocab.mask_index,
                 dropout_p = 0.5)


    
#############################Training Loop###################
mask_index = vectorizer.char_vocab.mask_index    

model = model.to(args.device)


optimizer = optim.Adam(model.parameters(),lr = args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                 mode = 'min',
                                                 factor = 0.5,patience = 1)

train_state = make_train_state(args)

epoch_bar = tqdm(desc='training_routine',total = args.num_epochs,position = 0)

dataset.set_split('train')
train_bar = tqdm(desc = 'split=train',\
                         total = dataset.get_num_batches(args.batch_size),\
                         position = 1,
                         leave = True)

dataset.set_split('val')
val_bar = tqdm(desc = 'split=val',
                         total = dataset.get_num_batches(args.batch_size),
                         position = 1,leave = True)



try:
    for epoch_index in range(args.num_epochs):
        train_state['epoch_index'] = epoch_index
        
        #Iterate over training set
        
        #setup: batch genertor, set loss and acc to 0, set train_mode on
        dataset.set_split('train')
        batch_generator = generate_batches(dataset,
                                           batch_size = args.batch_size,
                                           device = args.device)

        running_loss = 0.0
        running_acc = 0.0
        model.train()
        
        for batch_index, batch_dict in enumerate(batch_generator):
            
            #step 1. zero the gradients
            optimizer.zero_grad()
            
            #step2. compute the output
            y_pred = model(x_in = batch_dict['x_data'],
                           seinfeld_character_index = batch_dict['class_index'])
            
            #step3. compute the loss
            loss = sequence_loss(y_pred,batch_dict['y_target'],mask_index)
            
            #Step 4. Use loss to produce gradients
            loss.backward()
            
            #Step 5. Use optimizer to take gradient step
            optimizer.step()
            
            #Compute running loss and accuracy
            running_loss += (loss.item() - running_loss)/(batch_index + 1)
            acc_t = compute_accuracy(y_pred,batch_dict['y_target'],mask_index)
            running_acc += (acc_t - running_acc)/(batch_index + 1)
            
            #update bar
            train_bar.set_postfix(loss = running_loss,
                                  acc = running_acc,
                                  epoch = epoch_index)
            train_bar.update(100000)
      
            #del train_bar
            
        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)
        
        
        #Iterate over val dataset
        dataset.set_split('val')
        
        batch_generator = generate_batches(dataset,batch_size = args.batch_size,
                                           device = args.device)
        
        running_loss = 0.
        running_acc= 0.
        model.eval()
        
        for batch_index, batch_dict in enumerate(batch_generator):
            #compute the output
            y_pred = model(x_in = batch_dict['x_data'],
                           seinfeld_character_index = batch_dict['class_index'])
            
            #Compute the loss
            loss = sequence_loss(y_pred,batch_dict['y_target'],mask_index)
            
            #Compute tunning loss and accuracy
            running_loss += (loss.item() - running_loss) / (batch_index + 1)
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
        
            #Update bar
            val_bar.set_postfix(loss = running_loss,
                                acc = running_acc,
                                epoch = epoch_index)
            val_bar.update(10000)
    
            #del val_bar
        
        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)
        
        train_state = update_train_state(args =args,model=model,
                                         train_state = train_state)
        
        scheduler.step(train_state['val_loss'][-1])
        
        if train_state['stop_early']:
            break
        
        #move model to cpu for sampling
        sein_characters = np.random.choice(np.arange(len(vectorizer.seinfeld_char_vocab)),
                                              replace = True, size = 2)
        
        model = model.cpu()
        sampled_lines = decode_samples(
                sample_from_model(model,vectorizer, sein_characters = sein_characters),
                vectorizer)
        
        sample1 = "{} -> {}".format(vectorizer.seinfeld_char_vocab.lookup_index(sein_characters[0]),
                                   sampled_lines[0])
        
        sample2 = "{} -> {}".format(vectorizer.seinfeld_char_vocab.lookup_index(sein_characters[1]),
                                   sampled_lines[1])
        
        epoch_bar.set_postfix(sample1 = sample1,
                              sample2 = sample2)
        #Move model back to reight device
        
        model = model.to(args.device)
        
        epoch_bar.update()
        
except KeyboardInterrupt:
    print("Exiting Loop")    
        
        
        
#############Sample from final model################

model = model.cpu()
for index in range(len(vectorizer.seinfeld_char_vocab)):
    seinfeld_char = vectorizer.seinfeld_char_vocab.lookup_index(index)
    print("Sampled for {}:".format(seinfeld_char))
    sampled_indices = sample_from_model(model,vectorizer,
                                        sein_characters = [index]*3,
                                        temperature = 0.7)

    for sampled_line in decode_samples(sampled_indices,vectorizer):
        print("- "+sampled_line)


        
            
        
    

                

        
    














    
    
    
        
        
    
    
    
    
    
        
            
            
                
    
    
    
