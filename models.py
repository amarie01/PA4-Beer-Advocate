import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as torch_init

from torch.autograd import Variable
from config import *


class myGRU(nn.Module):
    def __init__(self, config):
        super(myGRU, self).__init__()
        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
        
        self.gru = nn.GRUCell(config['input_dim'], config['hidden_dim'])
        self.h2char = nn.Linear(config['hidden_dim'], config['output_dim'])
        
        self.init_hx = torch.zeros(config['batch_size'], config['hidden_dim']).cuda()
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.metadata = None
        
    def sample(self, outputs, config):
    # helper function to sample an index from a probability array
        indices = []
        
        outputs = outputs.div(config['gen_temp'])
        outputs = nn.Softmax(dim=1)(outputs)
        outputs = torch.multinomial(outputs, 1)
       
        return outputs
    
    def forward(self, sequence):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)
        hx = self.init_hx
        
        output = []
        
        for i in range(sequence.shape[1] - 1):
            hx = self.gru(sequence[:,i,:], hx)
            output.append(self.softmax(self.h2char(hx)))   
        output = torch.stack(output, dim=1)
        
        return output
    
    # 'metadata' is a numpy array of the beer style appended w/ beer rating. 'metadata'
    # is of the form (batch_size x encoding_length), where encoding_length is 
    # num_beer_styles + 1
    def generate(self, config, metadata):   
        hx = self.init_hx
        
        output = []
        
        init_char = np.zeros((config['batch_size'], n_letters))  # [1, 0, 0 ... 0] represents SOS
        init_char[:,0] = 1
        
        # Append metadata | SOS 
        gru_input = torch.from_numpy(np.concatenate((metadata, init_char), axis=1)).float().cuda()
               
        for _ in range(config['max_gen_len']):
            hx = self.gru(gru_input, (hx))
            out = self.softmax(self.h2char(hx))
                        
            # Convert next_char to one-hot encoding
            next_chars = np.zeros((config['batch_size'], n_letters))
            indices = self.sample(out, config)  
                        
            row_count = 0
            for i in indices:
                next_chars[row_count][i[0]] = 1
                row_count += 1

            # Append metadata | next_char
            gru_input = torch.from_numpy(np.concatenate(
                (metadata, next_chars), axis=1)).float().cuda()
            
            output.append(torch.from_numpy(next_chars))   
        output = torch.stack(output, dim=1)
            
        return output

    
class myLSTM(nn.Module):
    def __init__(self, config):
        super(myLSTM, self).__init__()
        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
                
        self.lstm = nn.LSTMCell(config['input_dim'], config['hidden_dim'])
        self.h2char = nn.Linear(config['hidden_dim'], config['output_dim'])
        
        self.dropout = nn.Dropout(config['dropout'])
        
        if config['cuda'] == True:
            self.init_cx = torch.zeros(config['batch_size'], config['hidden_dim']).cuda()
            self.init_hx = torch.zeros(config['batch_size'], config['hidden_dim']).cuda()
        else:
            self.init_cx = torch.zeros(config['batch_size'], config['hidden_dim']).cpu()
            self.init_hx = torch.zeros(config['batch_size'], config['hidden_dim']).cpu()
        
        self.softmax = nn.LogSoftmax(dim=1)
        self.metadata = None
        
    def sample(self, outputs, config):
    # helper function to sample an index from a probability array
        indices = []
        
        outputs = outputs.div(config['gen_temp'])
        outputs = nn.Softmax(dim=1)(outputs)
        outputs = torch.multinomial(outputs, 1)
      
        return outputs
    
    def forward(self, sequence):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)
        cx = self.init_cx
        hx = self.init_hx
        
        output = []
        
        for i in range(sequence.shape[1] - 1):
            hx, cx = self.lstm(sequence[:,i,:], (hx, cx))
            hx = self.dropout(hx)
            
            output.append(self.softmax(self.h2char(hx)))         
        output = torch.stack(output, dim=1)
        
        return output
    
    # 'metadata' is a numpy array of the beer style appended w/ beer rating. 'metadata'
    # is of the form (batch_size x encoding_length), where encoding_length is 
    # num_beer_styles + 1
    def generate(self, config, metadata):
        cx = self.init_cx
        hx = self.init_hx
        
        output = []
        
        init_char = np.zeros((config['batch_size'], n_letters))  # [1, 0, 0 ... 0] represents SOS
        init_char[:,0] = 1
        
        # Append metadata | SOS 
        lstm_input = torch.from_numpy(np.concatenate((metadata, init_char), axis=1)).float().cuda()
               
        for _ in range(config['max_gen_len']):
            hx, cx = self.lstm(lstm_input, (hx, cx))
            out = self.softmax(self.h2char(hx))
                        
            # Convert next_char to one-hot encoding
            next_chars = np.zeros((config['batch_size'], n_letters))
            indices = self.sample(out, config)  
                        
            row_count = 0
            for i in indices:
                next_chars[row_count][i[0]] = 1
                row_count += 1

            # Append metadata | next_char
            lstm_input = torch.from_numpy(np.concatenate(
                (metadata, next_chars), axis=1)).float().cuda()
            
            output.append(torch.from_numpy(next_chars))   
        output = torch.stack(output, dim=1)
            
        return output

