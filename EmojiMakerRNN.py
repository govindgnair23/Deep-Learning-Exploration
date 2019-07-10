# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:46:37 2019

@author: learningmachine
"""
##########The template of the code below is from the book NLP with Pytorch#############
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch
import random
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from argparse import Namespace
from tqdm import tqdm

##############################Utility functions##################
def load_glove_from_file(glove_filepath):
    """
    Load glove embeddings
    
    Args:
        glove_filepath: path to glove mebeddings file
    Returns:
        word_to_index(dict),embeddings(numpy.nparray)
    
    """
    word_to_index = {}
    embeddings = []
    
    with open(glove_filepath,"r",encoding="utf8") as fp:
        for index,line in enumerate(fp):
            line = line.split(" ")
            word_to_index[line[0]] = index
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    
    return word_to_index,np.stack(embeddings)


def make_embedding_matrix(glove_filepath,words):
    """
    Create embedding matrix for a specific set of words.
    
    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """
    
    word_to_idx,glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]
    
    final_embeddings = np.zeros((len(words),embedding_size))
    
    for i,word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i,:] = glove_embeddings[word_to_idx[word],]
        else:
            embedding_i = torch.ones(1,embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i,:] = embedding_i
            
    return final_embeddings
            


def read_csv(train_file = 'train_emoji.csv',test_file = 'tesss.csv'):
    train_df = pd.read_csv(train_file,header = None,usecols= [0,1], names = ["Sentence","category"])
    train_df = train_df.sort_values(by=['category'])
    
    
    #######Get no of observations in each category####
    category_counts = dict(train_df.category.value_counts().sort_index())
    
    ###Ensure each catgeory has proportionate representation in train and test splits
    split = []
    random.seed(1)
    for category,n_obs in category_counts.items():
        n_train = int(0.7*n_obs)
        n_val = n_obs - n_train
        category_split = ['train']*n_train + ['val']*n_val
        random.shuffle(category_split)
        split.extend(category_split)
    
    train_df["split"] = split
    
    test_df = pd.read_csv(test_file,header = None, names = ["Sentence","category"])
    test_df["split"] = "test"
    
    df = pd.concat([train_df,test_df],ignore_index = True)
    
    return df



#################Vocabulary for sentences and emojis###########
    
class Vocabulary(object):
    "Class to process and extract vocabulary for mapping"
    def __init__(self,token_to_idx=None):
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
            index =  self.token_to_idx[token]
        else: 
            try:
                index = max(self._token_to_idx.values())+1
            except ValueError:
                index = 0
                
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
    
    def lookup_token(self,token):
        """Retrieve the index associated with the token 
        
        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        """
        return self._token_to_idx[token]           
            
            
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
            raise KeyError("the index(%d) is not in vocabulary" %index)
        return self._idx_to_token[index]
    
    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)
    
    def __len__(self):
        return len(self._token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(self,token_to_idx = None,unk_token = "<UNK>",
                 mask_token = "<MASK>",begin_token = "<BGN>", end_token = "<END>"):
        
        super(SequenceVocabulary,self).__init__(token_to_idx)
        
        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_token = begin_token
        self._end_token = end_token
        
        
        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_index = self.add_token(self._begin_token)
        self.end_index = self.add_token(self._end_token)
    
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
    
######################Vectorizer#####################3
        
class Vectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use""" 
    def __init__(self,sentence_vocab,emoji_vocab):
        self.sentence_vocab = sentence_vocab
        self.emoji_vocab = emoji_vocab
        
    def vectorize(self,sentence,vector_length = -1):
        """
        Args:
            sentence (str): the string of words separated by a space
            vector_length (int): an argument for forcing the length of index vector
        Returns:
            the vetorized title (numpy.array)
        """
        indices = [self.sentence_vocab.begin_index]
        indices.extend([self.sentence_vocab.lookup_token(token) for token in sentence.split(" ")])
        indices.append(self.sentence_vocab.end_index)
        
        if vector_length < 0:
            vector_length = len(indices) - 1
            
        out_vector = np.zeros(vector_length, dtype = np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.sentence_vocab.mask_index
        
        return out_vector,len(indices)
    
    @classmethod
    def from_dataframe(cls,df,cutoff = 5):
        """Instantiate the vectorizer from the dataset dataframe
        
        Args:
            df (pandas.DataFrame): the target dataset
            cutoff (int): frequency threshold for including in Vocabulary 
        Returns:
            an instance of the NewsVectorizer
        """
        
        emoji_vocab = Vocabulary()
        for category in sorted(set(df.category)):
            emoji_vocab.add_token(category)
        
        word_counts = Counter()
        for sentence in df.Sentence:
            for token in sentence.split(" "):
                word_counts[token] += 1
                
        sentence_vocab = SequenceVocabulary()
        for word,word_count in word_counts.items():
            if word_count>=cutoff:
                sentence_vocab.add_token(word)
        
        return cls(sentence_vocab,emoji_vocab)
                    
###################Dataset################3

class EmojiDataset(Dataset):
    def __init__(self,df,vectorizer):
        
        """
        Args:
            df (pandas.DataFrame): the dataset
            vectorizer (NewsVectorizer): vectorizer instatiated from dataset
        """
        
        self.df = df
        self._vectorizer = vectorizer
        
        measure_len = lambda sentence: len(sentence.split(" "))
        self._max_seq_length = max(map(measure_len,df.Sentence)) + 2
        
        self.train_df = self.df[self.df.split == 'train']
        self.train_size = len(self.train_df)
        
        self.val_df = self.df[self.df.split=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.df[self.df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}
        
        self.set_split('train')
        
        
    @classmethod 
    def load_dataset_and_make_vectorizer(cls,train_file = 'train_emoji.csv',test_file = 'test_emoji.csv'):
        """
        Load dataset and make a new vectorizer from scratch
        
        Args:
            train_file(csv): location of the train dataset
            test_file(csv): location of the test dataset
        Returns:
            an instance of EmojiDataset
       
        
        """
        df = read_csv(train_file,test_file)
        train_df = df[df.split == 'train']
        return cls(df,Vectorizer.from_dataframe(train_df,cutoff=1))



    def get_vectorizer(self):
        """ return the vectorizer"""
        return self._vectorizer
    
    
    def set_split(self,split = 'train'):
        """ selects the split in the dataset using a column in the dataframe"""
        self._target_split = split
        self._target_df, self._target_size  = self._lookup_dict[split]
        
    
    def __len__(self):
        return self._target_size
    
    def __getitem__(self,index):
        """the primary entry point method for PyTorch datasets
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]
        
        sentence_vector,sentence_len = self._vectorizer.vectorize(row.Sentence,self._max_seq_length)
        emoji_index = self._vectorizer.emoji_vocab.lookup_token(row.category)
        
        return {'x_data': sentence_vector,
                'y_target': emoji_index,
                'x_length': sentence_len}
    
    
    def get_num_batches(self,batch_size):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self)//batch_size
        
        
def generate_batches(dataset,batch_size,shuffle=True,
                     drop_last = True,device = "cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    
    dataloader = DataLoader(dataset = dataset,batch_size=batch_size,
                            shuffle = shuffle, drop_last = drop_last)
    
    for data_dict in dataloader:
        out_data_dict = {}
        
        for name,tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        
        yield out_data_dict
        
    

#######################Model################

##Utility function ###

def column_gather(y_out,x_lengths):
    '''
    'Get a specific vector from each batch datapoint in `y_out`.
    
    More precisely, iterate over batch row indices, get the vector that's at
    the position indicated by the corresponding value in `x_lengths` at the row
    index.
    
    Args:
        y_out: (torch.FloatTensor, torch.cuda.FloatTensor)
            shape: (batch, sequence, feature)
        x_lengths (torch.LongTensor, torch.cuda.LongTensor)
            shape: (batch,)
            
    Returns:
        y_out (torch.FloatTensor, torch.cuda.FloatTensor)
            shape: (batch, feature)
            
    '''
    
    x_lengths = x_lengths.long().detach().cpu().numpy() - 1
    
    out = []
    
    for batch_index,column_index in enumerate(x_lengths):
        out.append(y_out[batch_index,column_index])
        
    return torch.stack(out)


class EmojiRecommender(nn.Module):
    """A Classifier with an RNN to extract features and an MLP to classify """
    def __init__(self,embedding_size,num_embeddings,num_classes,
                 rnn_hidden_size,batch_first = True,pretrained_embeddings = None,
                 padding_idx = 0,dropout_p = 0.3):
        """
        Args:
            embedding_size (int): The size of the word embedding
            num_embeddings (int): The number of  words to embed
            num_classes (int): The size of the prediction vector 
                Note: the number of emojis
            rnn_hidden_size (int): The size of the RNN's hidden state
            batch_first (bool): Informs whether the input tensors will 
                have batch or the sequence on the 0th dimension
            pretrained_embeddings (numpy.array): previously trained word embeddings
                default is None. If provided,
            padding_idx (int): The index for the tensor padding; 
                see torch.nn.Embedding
        """
        
        super(EmojiRecommender,self).__init__()
        
        if pretrained_embeddings is None:
            self.emb = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings = num_embeddings,
                                    padding_idx = padding_idx)
        
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim = embedding_size,
                                    num_embeddings = num_embeddings,
                                    padding_idx = padding_idx,
                                    _weight = pretrained_embeddings)
        
        
        
        self.rnn = nn.GRU(input_size = embedding_size,
                          num_layers = 1,
                          hidden_size = rnn_hidden_size,
                          batch_first = batch_first,
                          dropout = dropout_p)
        
        
        self.fc = nn.Linear(in_features = rnn_hidden_size,
                            out_features = num_classes)
        
        
        self.dropout_p = dropout_p
        
        
    def forward(self,x_in,x_lengths = None,apply_softmax = False):
        """ The forward pass of the model
        
        Args:
            x_in(torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
                
        Returns:
            the resulting tensor. tensor.shape should be (batch, char_vocab_size)
        """ 
        
        x_embedded = self.emb(x_in)
        y_out,_ = self.rnn(x_embedded)
        
        if x_lengths is not None:
            y_out = column_gather(y_out,x_lengths)
        else:
            y_out = y_out[:,-1,:]
            
        y_out = self.fc(y_out)
        
        
        if apply_softmax == True:
            y_out = F.softmax(y_out , dim =1)
        
        
        return y_out
            
            
###########Settings########

args = Namespace(
    # Data and path information
    train_file = 'train_emoji.csv',
    test_file = 'tesss.csv',
    save_dir="model_storage",
    # Model hyper parameter
    word_embedding_size=100,
    rnn_hidden_size=64,
    # Training hyper parameter
    num_epochs=100,
    batch_size=8,
    seed=1337,
    early_stopping_criteria=5,
    # Runtime hyper parameter
    cuda=True,
    catch_keyboard_interrupt=True,
    reload_from_files=False,
    expand_filepaths_to_save_dir=True,
    use_glove = True,
    glove_filepath = 'glove.6B.50d.txt',
    learning_rate = 0.001,
    model_state_file="model.pth"
)        
        
dataset =  EmojiDataset.load_dataset_and_make_vectorizer(args.train_file,args.test_file) 
vectorizer =  dataset.get_vectorizer()       
        
if args.use_glove:
    words = vectorizer.sentence_vocab._token_to_idx.keys()
    embeddings = make_embedding_matrix(args.glove_filepath,words)
    print("Using pre-trained embeddings")
else:
    embeddings = None
    print("Not using pre-trained embeddings")

emoji_recommender = EmojiRecommender(args.word_embedding_size,num_embeddings = len(vectorizer.sentence_vocab),
                                     num_classes = len(vectorizer.emoji_vocab),rnn_hidden_size = args.rnn_hidden_size,
                                     batch_first = True,pretrained_embeddings = None,
                                     padding_idx = 0,dropout_p = 0.5)    




args.device = torch.device("cuda" if args.cuda else "cpu")
##########################Training######################

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
    
    
    #Save one model atleast
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(),train_state['model_filename'])
        train_state['stop_early'] = False
     
    #save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1,loss_t = train_state['val_loss'][-2:]
        
        # if loss worsened
        if loss_t > loss_tm1:
            #Update step
            train_state['early_stopping_step'] += 1
        
        else:
            #decreased
            if loss_t < train_state['early_stopping_best_val']:
                
                torch.save(model.state_dict(),train_state['model_filename'])
                train_state['early_stopping_best_val'] = loss_t
            
            #reset
            train_state['early_stopping_step'] = 0
            
        #Stop early?
        train_state['stop_early'] =  train_state['early_stopping_step'] >= args.early_stopping_criteria
        
    
    return train_state
            
def compute_accuracy(y_pred,y_target):
    _,y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices,y_target).sum().item()
    return n_correct/len(y_pred_indices)*100    
   



emoji_recommender = emoji_recommender.to(args.device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(emoji_recommender.parameters(),lr = args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                 mode = 'min',factor = 0.5,patience = 1)

train_state = make_train_state(args)




epoch_bar = tqdm(desc='training routine',
                 total = args.num_epochs,
                 position = 0)


try:
    for epoch_index in range(args.num_epochs):
        train_state['epoch_index'] = epoch_index
        
        # Iterate over training dataset
        
        dataset.set_split('train')
        batch_generator = generate_batches(dataset,
                                           batch_size = args.batch_size,
                                           device = args.device)
        
        running_loss = 0.0
        running_acc = 0.0
        
        emoji_recommender.train()
        
        for batch_index, batch_dict in enumerate(batch_generator):
            
            #Zero the gradient
            optimizer.zero_grad()
            
            #Compute the output
            y_pred = emoji_recommender(x_in = batch_dict['x_data'],
                                       x_lengths = batch_dict['x_length'])
            
            #Compute the loss
            loss = loss_func(y_pred,batch_dict['y_target'])
            
            running_loss += (loss.item() - running_loss)/(batch_index + 1)
            
            #Get gradients
            loss.backward()
            
            #Optimize
            optimizer.step()
            
            #compute the accuracy
            acc_t = compute_accuracy(y_pred,batch_dict['y_target'])
            running_acc += (acc_t - running_acc)/(batch_index + 1)
            
        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)
        
        #Iterate over val dataset
        dataset.set_split('val')
        batch_generator = generate_batches(dataset,
                                           batch_size = args.batch_size,
                                           device = args.device)
        
        running_loss = 0.0
        running_acc = 0.0
        emoji_recommender.eval()
        
        for batch_index,batch_dict in enumerate(batch_generator):
            #compute the output
            y_pred = emoji_recommender(x_in = batch_dict['x_data'],
                                       x_lengths = batch_dict['x_length'])
            
            #Compute loss
            loss = loss_func(y_pred,batch_dict['y_target'])
            running_loss += (loss.item() - running_loss)/(batch_index + 1)
            
            #Compute the accuracy
            acc_t = compute_accuracy(y_pred,batch_dict['y_target'])
            running_acc += (acc_t - running_acc)/(batch_index + 1)
        
        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)
        
        train_state = update_train_state(args = args, model = emoji_recommender,
                                         train_state = train_state)
        
        scheduler.step(train_state['val_loss'][-1])
        
        epoch_bar.set_postfix(train_loss = train_state['train_loss'][-1],
                              train_acc = train_state['train_acc'][-1],
                              valid_loss = train_state['val_loss'][-1],
                              valid_acc = train_state['val_acc'][-1])
        epoch_bar.update()
        
        if train_state['stop_early']:
            break

except KeyboardInterrupt:
    print("Exiting Loop")
            
        
#######Compute loss and accuracy using best available model##
emoji_recommender.load_state_dict(torch.load(train_state['model_filename']))

emoji_recommender = emoji_recommender.to(args.device)
loss_func = nn.CrossEntropyLoss()

dataset.set_split('test')
batch_generator = generate_batches(dataset,batch_size = args.batch_size,device=args.device)

running_loss = 0.0
running_acc = 0.0
emoji_recommender.eval()


test_preds = np.empty(0)
test_labels = np.empty(0)

for batch_index,batch_dict in enumerate(batch_generator):
    #compute the output
    y_pred = emoji_recommender(x_in = batch_dict['x_data'], x_lengths = batch_dict['x_length'])
    
    #compute the loss
    loss = loss_func(y_pred,batch_dict['y_target'])
    loss_t = loss.item()
    running_loss += (loss_t - running_loss)/(batch_index + 1)
    
    #compute the accuracy
    acc_t = compute_accuracy(y_pred,batch_dict['y_target'])
    running_acc += (acc_t - running_acc)/(batch_index + 1)
    
    _,pred_index = y_pred.detach().max(dim=1)
    pred_index = pred_index.cpu().numpy()
    
    test_preds = np.append(test_preds,pred_index)
    test_labels = np.append(test_labels,batch_dict['y_target'].cpu().numpy())


pd.crosstab(test_labels,test_preds,rownames=['Actual'],colnames=['Predicted'])
 
 
train_state['test_loss'] = running_loss
train_state['test_acc'] = running_acc


print("Test Loss:{}".format(train_state['test_loss']))
print("Test Accuracy:{}".format(train_state['test_acc']))


        
        

        
        
    
    
        









        
        
        
        
