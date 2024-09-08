# the model RNN, LSTM, GRU are reference from pytorch official website https://pytorch.org/
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class RNN_Model(nn.Module):
    def __init__(self,n_input,n_hidden,n_class=1):
        super(RNN_Model, self).__init__()
        self.rnn = nn.RNN(n_input, n_hidden, num_layers=2, batch_first=True)
        # Linear layer for output
        self.linear = nn.Linear(n_hidden,n_class)

    def forward(self, x):
        rnn_output, h_n = self.rnn(x)
        x_last = rnn_output[:,-1,:] 
        x = self.linear(x_last)
        return x