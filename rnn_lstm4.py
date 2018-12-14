import torch
import torch.nn as nn
import torch.nn.functional as F

# Model based on CellNuclei.ipynb reference [1]

class RNNet(nn.Module):
    def __init__(self):
        super().__init__()
        # (Batch_size, hidden_size = 3, seq_len = 196)
        batch_size = 64
        pad11 = samePad(5, 1)
        self.conv11 = nn.Conv1d(in_channels = 3, out_channels = 48, kernel_size = 5, padding = pad11)
        nn.init.xavier_uniform_(self.conv11.weight)
        # (Batch_size, 48, 196)
        pad21 = samePad(5, 1)
        self.conv21 = nn.Conv1d(in_channels = 48, out_channels = 64, kernel_size = 5, padding = pad21)
        nn.init.xavier_uniform_(self.conv21.weight)
#       (Batch_size, 64, 196)
        pad31 = samePad(3, 1)
        self.conv31 = nn.Conv1d(in_channels = 64, out_channels = 96, kernel_size = 3, padding = pad31)
        nn.init.xavier_uniform_(self.conv31.weight)
        #(Batch_size, 96, 196)
        self.lstm41 = nn.LSTM(input_size= 96, hidden_size = 48, dropout = 0.0, bidirectional = True)

        self.lstm51 = nn.LSTM(input_size = 96, hidden_size = 48, dropout = 0.0, bidirectional = True)

        self.forward61 = nn.Linear(18816, batch_size)
        nn.init.xavier_uniform_(self.forward61.weight)

        self.forward71 = nn.Linear(batch_size, 5)
        nn.init.xavier_uniform_(self.forward71.weight)


    def forward(self, x):
        batch_size = 64
        x1 = self.conv11(x)
        x2 = self.conv21(F.tanh(x1))
        x3 = self.conv31(F.tanh(x2))

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        x_rnn = x3.transpose(1, 2).transpose(0, 1)

        output4, (x4_hidden, x4_cell) = self.lstm41(x_rnn)
        output5, (x5_hidden, x5_cell) = self.lstm51(output4, (x4_hidden, x4_cell))

        seq_len, batch, out = output5.size()
        out_transpose = output5.transpose(1, 2)

        out_same = out_transpose.contiguous()
        x_forward = out_same.view(seq_len*out, batch)
        x_f = x_forward.transpose(0, 1)

        x6 = self.forward61(x_f)
        x7 = self.forward71(F.tanh(x6))

        x_out = F.softmax(x7)

        return x_out

def samePad(filterSize, stride):
    return int(float(filterSize - stride)/2)
