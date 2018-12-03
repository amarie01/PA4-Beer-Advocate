import re
import os
import string
import operator
import time

from models import *
from config import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as torch_init

from torch.autograd import Variable
from nltk.translate import bleu_score

# -------------------
# -------------------
#   Reads from the csv file given by 'fname' and returns a 
#   pandas DataFrame of the read csv
def load_data(fname):
    return pd.read_csv(fname)

# -------------------
# -------------------
#   Converts each character in 'review_txt' string to a hot-encoded value. 
#   
#   ** NOTE **
#
#   All letters are converted to lowercase, and all special characters 
#   (except for <space>, <period>, <comma>, <semicolon>, <dash>, <parentheses> 
#   <forward slash>) are ignored
def text_to_onehot(review_txt):
    encoded = []
    
    for char in review_txt:
        c = np.zeros(n_letters)
        char = char.lower()

        if char in all_letters:
            c[all_letters.find(char) + 1] = 1  # Shift right 1 for SOS char
            encoded.append(c)
    
    # Return an numpy array
    return np.array(encoded)

# -------------------
# -------------------
#   Converts each array in 'onehot' numpy array to a its corresponding 
#   character: either a lowercase letter, a digit, a <space>, a <period>, 
#   a <comma>, a <semicolon>, a <dash>, a <parentheses>, a <forward slash>, 
#   or a padding character (<SOS> or <EOS>) 
#   
#   ** NOTE **
#
#  'onehot' is a numpy array of numpy arrays, where each subarray represents
#  a one-hot encoding of a single character in the review text.
def onehot_to_text(onehot):
    text = ""
    
    for encoded_char in onehot:
        i = encoded_char.max(0)[1] # argmax
        
        if i == 0:
            text += ("<SOS>")
        elif i == (n_letters - 1):
            text += ("<EOS>")
        elif i == (n_letters - 2):
            text += ("<PAD>")
        else:
            text += (all_letters[i - 1])
    
    return text

# -------------------
# -------------------
#   For each review in batch, this function: 
#   1) Appends the metadata to the start of each char
#   2) Converts the labels in 'y_batch' from one hot
#      encodings to integers
#
#   E.G. [0, 1, 0, 0, ... , 0] -> "1"
def batch_to_sequence(X_batch, y_batch):
    sequence = []
    labels = []
    
    for i in range(len(y_batch)):
        review = []
        label = []
        
        for char in y_batch[i]:
            style = X_batch[i][0]
            rating = X_batch[i][1]
            
            char_index = char.argmax(axis=0)
            
            if char_index != 0:
                label.append(char.argmax(axis=0))     
            
            review.append(torch.from_numpy(
                np.concatenate((style, np.array([rating]), char)))) 
          
        labels.append(torch.from_numpy(np.array(label)))
        
        review = torch.stack(review,dim=0)
        sequence.append(review)
        
    sequence = torch.stack(sequence,dim=0)
    labels = torch.stack(labels,dim=0)
    
    return sequence, labels

# -------------------
# -------------------
#   Computes BLEU score between actual text (reference) and 
#   predicted text (hypothesis)
def calc_BLEUscore(reference, hypothesis):
    references = re.compile('\w+').findall(reference)
    hypotheses = re.compile('\w+').findall(hypothesis)
    
    # 1-gram score
    return bleu_score.sentence_bleu([references], hypotheses, weights=(1, 0, 0, 0))

# -------------------
# -------------------
#   Input 'data' is a pandas DataFrame. This func returns 
#   a numpy array that has all features (including all
#   text characters in one hot encoded form).
def process_train_data(data):
    unique_styles = list(beer_styles.keys())
    
    features = []
    labels = []
    
    # Iterate over pandas DataFrame
    for i, review in enumerate(data.iterrows()):
        review = review[1]
        
        text = review['review/text']
        style = review['beer/style']
        rating = review['review/overall']
        
        onehot_style = np.zeros(len(unique_styles))   # Initialize one-hot encoded
        onehot_style[unique_styles.index(style)] = 1  # One-hot encode beer style
        
        print("---Process Training---Index: " + str(i) + "; Percent complete: " + 
              str(round((i/ (1.0 * len(data)) * 100), 2)) + "%")
        
        text = text.replace("!", ".")    # Set all ending punctuation to "."
        text = text.replace("?", ".")    # Set all ending punctuation to "."
        text = re.sub('\s+', ' ', text)  # Remove \n and \t
        text = text.lower()              # Remove all uppercase letters
        
        features.append(np.array([onehot_style, rating]))
        labels.append(text)
        
    return np.array(features), np.array(labels)

# -------------------
# -------------------
#   Takes in training data and labels as numpy array and applies a 90-10 split
#   for the training and validation data, respectively. Returns a set of 
#   4 numpy arrays, where each corresponds to the train_data, train_labels, 
#   valid_data, and valid_labels.
def train_valid_split(data, labels):
    X_train = []
    y_train = []
    X_valid = []
    y_valid = []
    
    val_split = int(0.1 * labels.size)  # 10% of data allocated to validation set
    
    all_indices = np.indices(labels.shape)[0]
    val_indices = np.random.choice(all_indices, val_split, replace=False)
    
    for i in range(labels.size):
        
        if i in val_indices:
            X_valid.append(data[i])
            y_valid.append(labels[i])
        else:
            X_train.append(data[i])
            y_train.append(labels[i])
            
    return np.array(X_train), np.array(y_train), np.array(X_valid), np.array(y_valid)

# -------------------
# -------------------
#   Takes in pandas DataFrame and returns a numpy array that has all 
#   input features. Note that test data does not contain any review
#   text, so no need to hot-encode any text.
def process_test_data(data):     
    unique_styles = list(beer_styles.keys())
    features = []
    
    # Iterate over pandas DataFrame
    for i, review in enumerate(data.iterrows()):
        review = review[1]
        
        style = review['beer/style']
        rating = review['review/overall']
        
        if style not in unique_styles:
            style = max(beer_styles.items(), key=operator.itemgetter(1))[0]
        
        onehot_style = np.zeros(len(unique_styles))   # Initialize one-hot encoded
        onehot_style[unique_styles.index(style)] = 1  # One-hot encode beer style
        
        print("---Process Testing---Index: " + str(i) + "; Percent complete: " + 
              str(round((i/ (1.0 * len(data)) * 100), 2)) + "%")

        features.append(np.array([onehot_style, rating]))
        
    return np.array(features)

# -------------------
# -------------------
#   This function appends each review in 'orig_data' with enough <PAD> 
#   characters s.t. each review is the same length as the longest
#   review in the batch. Also, this function pads each review w/
#   the <SOS> and <EOS> characters.
#
#   ** NOTE **
#
#   It is assumed that 'orig_data' is a numpy array of hot-encoded characters
def pad_data(orig_data):
    # Find longest review in batch
    reviews = orig_data.tolist()
    max_len = len(max(reviews, key=len))
    
    SOS_val = np.zeros(n_letters)  # [1, 0, 0 ... 0] represents SOS
    SOS_val[0] = 1
    
    EOS_val = np.zeros(n_letters)  # [0, 0, 0 ... 1] represents EOS
    EOS_val[n_letters - 1] = 1
    
    # Loop over all reviews in batch
    for i in range(orig_data.shape[0]):
      
        # Pad with PAD character s.t. all reviews are of same length
        padding = np.zeros(((max_len - orig_data[i].shape[0]), n_letters))
        
        # [0, 0, ... 1, 0] represents PAD
        padding[:, (n_letters - 2)] = 1
        
        orig_data[i] = np.concatenate(([SOS_val], orig_data[i], 
                                       [EOS_val], padding), axis=0)
    return orig_data

# -------------------
# -------------------
#   Training function
def train(model, model_name, X_train, y_train, X_valid, y_valid, cfg, computing_device, 
          optimizer, criterion):
    train_loss = []
    valid_loss = []
    bleu_scores = []
    
    start = time.time()
    
    min_valid_loss = 100000000
    early_stop_count = 0
       
    for epoch in range(cfg['epochs']):
        epoch_train_loss = []
        
        # ------------------- Training 
    
        for minibatch_count in range(0, len(X_train), cfg['batch_size']):
            
            train_feats = X_train[minibatch_count:(minibatch_count+cfg['batch_size'])]
            
            train_labels  = y_train[minibatch_count:(minibatch_count+cfg['batch_size'])]    
            onehot_labels = []
            
            # One-hot encode labels
            for review_count in range(train_labels.shape[0]):
                onehot_labels.append(text_to_onehot(train_labels[review_count][:cfg['max_train_len']]))
            
            # Pad labels with encoded <SOS> and <EOS> values
            onehot_labels = np.array(onehot_labels)
            onehot_labels = pad_data(onehot_labels)
            
            # Concatenate [ beer_style | beer_rating | char_encoding ] 
            # for forward pass
            sequence, labels = batch_to_sequence(train_feats, onehot_labels)
        
                    
            # Put minibatch data in CUDA tensors and run on GPU if supported
            sequence, labels = sequence.float().to(computing_device), labels.to(computing_device)
            
            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()
            # Perform the forward pass through the network and compute the loss
            outputs = model(sequence)
            
            loss = 0
            for i in range(outputs.shape[1]):
                ispadded = (labels[:,i] != (n_letters - 2))
                
                # Ignore the gradients for <PAD> characters
                l = torch.mul(criterion(outputs[:,i], labels[:,i]), ispadded.float())
                
                # Divide by number of characters
                divisor = torch.tensor(outputs.shape[1])
                divisor = divisor.expand_as(l).float().to(computing_device)
           
                l = torch.div(l, divisor)
                loss += torch.mean(l, 0)
            
            # Automagically compute the gradients and backpropagate the loss through the network
            loss.backward()
            epoch_train_loss.append(loss.cpu().item())
            
            # Update the weights
            optimizer.step()
            
            if minibatch_count % 100 == 0: 
                print("--- Epoch: " + str(epoch) + "; Minibatch: " + str(minibatch_count) + 
                      "; Avg Train Loss: " + str(np.mean(np.array(epoch_train_loss))))
                print("--- Original Review ---")
                print(train_labels[0])
                print("--- Outputted Review ---")
                print(onehot_to_text(outputs[0]))
            
        train_loss.append(np.mean(np.array(epoch_train_loss)))
        
        # ------------------- Validation
        epoch_valid_loss = []
        epoch_bleu_score = []
        
        with torch.no_grad():
            for minibatch_count in range(0, len(X_valid), cfg['batch_size']):

                valid_feats = X_valid[minibatch_count:(minibatch_count+cfg['batch_size'])]

                valid_labels  = y_valid[minibatch_count:(minibatch_count+cfg['batch_size'])]    
                onehot_labels = []

                # One-hot encode labels
                for review_count in range(valid_labels.shape[0]):
                    onehot_labels.append(text_to_onehot(valid_labels[review_count][:cfg['max_train_len']]))

                # Pad labels with encoded <SOS> and <EOS> values
                onehot_labels = np.array(onehot_labels)
                onehot_labels = pad_data(onehot_labels)

                # Concatenate [ beer_style | beer_rating | char_encoding ] 
                # for forward pass
                sequence, labels = batch_to_sequence(valid_feats, onehot_labels)
             
                # Put minibatch data in CUDA tensors and run on GPU if supported
                sequence, labels = sequence.float().to(computing_device), labels.to(computing_device)

                # Perform the forward pass through the network and compute the loss
                outputs = model(sequence)
            
                loss = 0
                for i in range(outputs.shape[1]):
                    ispadded = (labels[:,i] != (n_letters - 2))
                    # Ignore the gradients for <PAD> characters
                    l = torch.mul(criterion(outputs[:,i], labels[:,i]), ispadded.float())
                    
                    # Divide by number of characters
                    divisor = torch.tensor(outputs.shape[1])
                    divisor = divisor.expand_as(l).float().to(computing_device)
           
                    l = torch.div(l, divisor)
                    loss += torch.mean(l, 0)

                epoch_valid_loss.append(loss.cpu().item())
        
                # Calculate the batch bleu scores
                batch_bleu_score = []
                
                for i in range(len(outputs)):                 
                    reference  = valid_labels[i]
                    hypothesis = onehot_to_text(outputs[i])
                                        
                    batch_bleu_score.append(calc_BLEUscore(reference, hypothesis))
                epoch_bleu_score.append(np.mean(batch_bleu_score))
        
        epoch_bleu_score = np.mean(np.array(epoch_bleu_score))
        
        bleu_scores.append(epoch_bleu_score)     
        valid_loss.append(np.mean(np.array(epoch_valid_loss))) 
        
        epoch_loss = np.mean(np.array(epoch_valid_loss))
        
        print("--- Epoch: " + str(epoch) + "; Avg Valid Loss: " + 
              str(epoch_loss))
        
        # Early stopping 
        if epoch_loss >= min_valid_loss:
            early_stop_count += 1
            
            if early_stop_count == 3:
                print("--- Early Stopping @ Epoch: " + str(epoch) + 
                      "; Min. Validation Loss: " + str(min_valid_loss))
                break
        else:
            early_stop_count = 0
            min_valid_loss = epoch_loss
            
        print("Finished training on " + str(epoch + 1) + 
              " epoch; Bleu Score: " + str(epoch_bleu_score) + 
              "; Seconds Lapsed: " + str(round((time.time() - start), 2)))
        print()
        print()
        
    print("Training complete after " + str(epoch + 1) + " epochs")
    print("Saving model")
    torch.save(model, "models/" + model_name + "_epoch" + str(epoch + 1) + '.pt')
    print("Save complete")
    
    return train_loss, valid_loss, bleu_scores 

# -------------------
# -------------------
#   Writes output to 'filename'. This function 
#   is called in generate
def save_to_file(outputs, out_file):
        for output in outputs:
            text = onehot_to_text(output)
            text = text.split('<EOS>')[0]
            out_file.write(text + '\n')

# -------------------
# -------------------
#   Generate function
def generate(model, X_test, cfg, computing_device, filename):
    # TODO: Given n rows in test data, generate a list of n strings, where each string is the review
    # corresponding to each input row in test data.
    
    with open(filename, "a") as out_file:
        with torch.no_grad():
            for minibatch_count in range(0, len(X_test), cfg['batch_size']):
                # Concatenate [ beer_style | beer_rating | char_encoding ] 
                # for forward pass
                feats = X_test[minibatch_count:(minibatch_count+cfg['batch_size'])]
               
                metadata = []

                for i in range(len(feats)):
                    style = feats[i][0]
                    rating = feats[i][1]

                    metadata.append(torch.from_numpy(
                        np.concatenate((style, np.array([rating])))))

                metadata = torch.stack(metadata, dim=0) 
                # Put minibatch data in CUDA tensors and run on GPU if supported
                metadata = metadata.float().to(computing_device)

                # Perform the forward pass through the network and compute the loss
                outputs = model.generate(cfg, metadata)
    
                save_to_file(outputs, out_file)

                if minibatch_count % 1000 == 0: 
                    print("--- Minibatch: " + str(minibatch_count))
                    print("--- Generated Review ---")
                    print(onehot_to_text(outputs[0]))

                    
if __name__ == "__main__":
    train_data_fname = "Beeradvocate_Train.csv"
    test_data_fname = "Beeradvocate_Test.csv"
    out_fname = ""
    
    train_data = load_data(train_data_fname) # Generating the pandas DataFrame
    test_data = load_data(test_data_fname)   # Generating the pandas DataFrame

    train_data.replace("", np.nan, inplace=True)
    train_data.dropna(inplace=True)

    # Cut-down data for runtime purposes
    train_data = train_data[:4800]
    
    # Find all unique beer styles
    beer_types = sorted(train_data['beer/style'].unique(), key=str.lower)
    temp = dict.fromkeys(beer_types)

    for key in temp:
        temp[key] = train_data[train_data['beer/style'] == key].shape[0]

        beer_styles = temp
        print(list(beer_styles.keys()))
    
    # Converting DataFrame to numpy array
    train_feats, train_labels = process_train_data(train_data)
    
    # Splitting the train data into train-valid data
    X_train, y_train, X_valid, y_valid = train_valid_split(train_feats, train_labels)
    
    # Converting DataFrame to numpy array
    X_test = process_test_data(test_data)  
    
    if cfg['cuda']:
        computing_device = torch.device("cuda")
        print("CUDA is supported")
    else:
        computing_device = torch.device("cpu")
        print("CUDA NOT supported")
    
    # ------------------- Generate reviews using best model
    
    best_model = torch.load("models/best_LSTM.pt")
    temperatures = [0.4, 0.01, 10]


    for t in range(len(temperatures)):
        cfg['gen_temp'] = temperatures[t]
        out_file = "reviews_tau_" + str(cfg['gen_temp']) + ".txt"
        
        generate(best_model, X_test, cfg, computing_device, out_file)
    
    

'''
train_data_fname = "Beeradvocate_Train.csv"
test_data_fname = "Beeradvocate_Test.csv"
out_fname = ""

train_data = load_data(train_data_fname) # Generating the pandas DataFrame
test_data = load_data(test_data_fname)   # Generating the pandas DataFrame

train_data.replace("", np.nan, inplace=True)
train_data.dropna(inplace=True)

# Cut-down data for runtime purposes
train_data = train_data[:4800]

# Find all unique beer styles
beer_types = sorted(train_data['beer/style'].unique(), key=str.lower)
temp = dict.fromkeys(beer_types)

for key in temp:
    temp[key] = train_data[train_data['beer/style'] == key].shape[0]

beer_styles = temp
print(list(beer_styles.keys()))

# Converting DataFrame to numpy array
train_feats, train_labels = process_train_data(train_data)

# Splitting the train data into train-valid data
X_train, y_train, X_valid, y_valid = train_valid_split(train_feats, train_labels)

# Converting DataFrame to numpy array
X_test = process_test_data(test_data)    

if cfg['cuda']:
    computing_device = torch.device("cuda")
    print("CUDA is supported")
else:
    computing_device = torch.device("cpu")
    print("CUDA NOT supported")

criterion = nn.NLLLoss(reduction='none')


# ------------------- Train LSTM and GRU on 1st set of hyperparams

hyperparams = {}

hyperparams['dropout'] = [0.0, 0.1]
hyperparams['hidden_dim'] = [64, 32]
hyperparams['learning_rate'] = [0.005, 0.01]

cfg['dropout'] = hyperparams['dropout'][0]
cfg['hidden_dim'] = hyperparams['hidden_dim'][0]
cfg['learning_rate'] = hyperparams['learning_rate'][0]

lstm1 = myLSTM(cfg)
lstm1.to(computing_device)

gru1 = myGRU(cfg)
gru1.to(computing_device)

LSTM1optimizer = optim.Adam(lstm1.parameters(), lr=cfg['learning_rate'])
GRU1optimizer = optim.Adam(gru1.parameters(), lr=cfg['learning_rate'])

# Train the model
lstm1_train_losses, lstm1_valid_losses, lstm1_bleu_scores = train(
    lstm1, "LSTM_model1", X_train, y_train, X_valid, y_valid, cfg, 
    computing_device, LSTM1optimizer, criterion) 

plt.plot(range(len(lstm1_train_losses)), lstm1_train_losses, 'b--', label='Training Loss')
plt.plot(range(len(lstm1_valid_losses)), lstm1_valid_losses, 'r--', label='Validation Loss')

plt.grid(True)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Model 1: Training vs. Validation Losses")
plt.legend(loc='upper right')

plt.savefig("images/lstm1_losses.png")

plt.plot(range(len(lstm1_bleu_scores)), lstm1_bleu_scores, 'b--', label='Bleu Scores')

plt.grid(True)

plt.xlabel("Epoch")
plt.ylabel("Bleu Score")
plt.title("LSTM Model 1: Bleu Scores on Validation Set")
plt.legend(loc='lower right')

plt.savefig("images/lstm1_bleus.png")

# Train the model
gru1_train_losses, gru1_valid_losses, gru1_bleu_scores = train(
    gru1, "GRU_model1", X_train, y_train, X_valid, y_valid, cfg, 
    computing_device, GRU1optimizer, criterion) 

plt.plot(range(len(gru1_train_losses)), gru1_train_losses, 'b--', label='Training Loss')
plt.plot(range(len(gru1_valid_losses)), gru1_valid_losses, 'r--', label='Validation Loss')

plt.grid(True)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GRU Model 1: Training vs. Validation Losses")
plt.legend(loc='upper right')

plt.savefig("images/gru1_losses.png")

plt.plot(range(len(gru1_bleu_scores)), gru1_bleu_scores, 'b--', label='Bleu Scores')

plt.grid(True)

plt.xlabel("Epoch")
plt.ylabel("Bleu Score")
plt.title("GRU Model 1: Bleu Scores on Validation Set")
plt.legend(loc='lower right')

plt.savefig("images/gru1_bleus.png")

# ------------------- Train LSTM and GRU on 2nd set of hyperparams

cfg['dropout'] = hyperparams['dropout'][1]
cfg['hidden_dim'] = hyperparams['hidden_dim'][1]
cfg['learning_rate'] = hyperparams['learning_rate'][1]

lstm2 = myLSTM(cfg)
lstm2.to(computing_device)

cfg['dropout'] = hyperparams['dropout'][0]

gru2 = myGRU(cfg)
gru2.to(computing_device)

LSTM2optimizer = optim.SGD(lstm2.parameters(), lr=cfg['learning_rate'])
GRU2optimizer = optim.SGD(gru2.parameters(), lr=cfg['learning_rate'])

# Train the model
lstm2_train_losses, lstm2_valid_losses, lstm2_bleu_scores = train(
    lstm2, "LSTM_model2", X_train, y_train, X_valid, y_valid, cfg, 
    computing_device, LSTM2optimizer, criterion)  

plt.plot(range(len(lstm2_train_losses)), lstm2_train_losses, 'b--', label='Training Loss')
plt.plot(range(len(lstm2_valid_losses)), lstm2_valid_losses, 'r--', label='Validation Loss')

plt.grid(True)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Model 2: Training vs. Validation Losses")
plt.legend(loc='upper right')

plt.savefig("images/lstm2_losses.png")

plt.plot(range(len(lstm2_bleu_scores)), lstm2_bleu_scores, 'b--', label='Bleu Scores')

plt.grid(True)

plt.xlabel("Epoch")
plt.ylabel("Bleu Score")
plt.title("LSTM Model 2: Bleu Scores on Validation Set")
plt.legend(loc='lower right')

plt.savefig("images/lstm2_bleus.png")

cfg['dropout'] = hyperparams['dropout'][0]

gru2 = myGRU(cfg)
gru2.to(computing_device)

# Train the model
gru2_train_losses, gru2_valid_losses, gru2_bleu_scores = train(
    gru2, "GRU_model2", X_train, y_train, X_valid, y_valid, cfg, 
    computing_device, GRU2optimizer, criterion) 

plt.plot(range(len(gru2_train_losses)), gru2_train_losses, 'b--', label='Training Loss')
plt.plot(range(len(gru2_valid_losses)), gru2_valid_losses, 'r--', label='Validation Loss')

plt.grid(True)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GRU Model 2: Training vs. Validation Losses")
plt.legend(loc='upper right')

plt.savefig("images/gru2_losses.png")

plt.plot(range(len(gru2_bleu_scores)), gru2_bleu_scores, 'b--', label='Bleu Scores')

plt.grid(True)

plt.xlabel("Epoch")
plt.ylabel("Bleu Score")
plt.title("GRU Model 2: Bleu Scores on Validation Set")
plt.legend(loc='lower right')

plt.savefig("images/gru2_bleus.png")

# ------------------- Generate reviews using best model

lstm1_avg_bleu = np.mean(lstm1_bleu_scores)
lstm2_avg_bleu = np.mean(lstm2_bleu_scores)
gru1_avg_bleu = np.mean(gru1_bleu_scores)
gru2_avg_bleu = np.mean(gru2_bleu_scores)

avg_bleu_scores = {lstm1: lstm1_avg_bleu, lstm2: lstm2_avg_bleu, 
                   gru1: gru1_avg_bleu, gru2: gru2_avg_bleu}

best_model = max(avg_bleu_scores.items(), key=operator.itemgetter(1))[0]

print("Saving BEST model: " + str(best_model))
torch.save(best_model, 'models/best_model.pt')
print("Save complete")

best_model = torch.load("models/best_LSTM.pt")

print(best_model)

temperatures = [10, 0.01, 10]

for t in range(len(temperatures)):
    cfg['gen_temp'] = temperatures[t]
    out_file = "reviews_tau_" + str(cfg['gen_temp']) + ".txt"
    generate(best_model, X_test, cfg, computing_device, outfile)
'''
