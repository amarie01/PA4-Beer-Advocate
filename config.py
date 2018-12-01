import string

beer_styles = {}
all_letters = string.ascii_lowercase + string.digits + " .,;:/-()"

# Account for SOS, PAD, and EOS markers
n_letters = len(all_letters) + 3     

cfg = {}

cfg['input_dim'] = 122         # input dimension to LSTM
cfg['hidden_dim'] = 256        # hidden dimension for LSTM
cfg['output_dim'] = n_letters  # output dimension of the model

# dropout useful only when layers > 1; between 0 and 1
cfg['dropout'] = 0.1    # dropout rate between two layers of LSTM 

cfg['cuda'] = True           # True or False depending whether you run your model on a GPU or not
cfg['epochs'] = 15           # number of epochs for which the model is trained
cfg['batch_size'] = 16       # batch size of input
cfg['learning_rate'] = 0.001 # learning rate to be used

cfg['gen_temp'] = 0.01       # temperature to use while generating reviews
cfg['max_gen_len'] = 1000    # maximum character length of the generated reviews
cfg['max_train_len'] = 500   # maximum character length of review accepted during training

# True ~> model is in training mode; False ~> model is not being used to generate reviews
cfg['train'] = True 

# cfg['layers'] =          # number of layers of LSTM
# cfg['bidirectional'] =   # True or False; True means using a bidirectional LSTM
# cfg['L2_penalty'] =      # weighting constant for L2 regularization term; used in optimizer definition