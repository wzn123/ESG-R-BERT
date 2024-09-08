# the model RNN, LSTM, GRU are reference from pytorch official website https://pytorch.org/
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class GRU_Model(nn.Module):
    def __init__(self,n_input,n_hidden,n_class=1):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(n_input, n_hidden, num_layers=2, batch_first=True)
        # Linear layer for output
        self.linear = nn.Linear(n_hidden,n_class)

    def forward(self, x):
        gru_output, h_n = self.gru(x)
        x_last = gru_output[:,-1,:] 
        x = self.linear(x_last)
        return x
