import torch
import torch.nn as nn

class LSTM_CLOUD(nn.Module):

    def __init__(self, feature_num, hidden_size, seq_len, hidden_num=2):
        '''Batch, Sequence number, Input Size(Features)'''
        super(LSTM_CLOUD, self).__init__()
        self.hidden_layers = nn.LSTM(input_size=feature_num, hidden_size=hidden_size, num_layers=hidden_num, dropout=0.2, batch_first=True)
        self.mlp = nn.ModuleList([
            nn.Linear(hidden_size, 32),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64),
            nn.Dropout(p=0.2),
            nn.Linear(64, seq_len)
        ])

    def forward(self, x):
        x = x.float()
        assert len(x.shape) == 3
        lstm_out, _  = self.hidden_layers(x)
        lstm_last_time_out = lstm_out[:,-1, :] # only use last timepoint's features
        out= lstm_last_time_out
        for _, dense_layer in enumerate(self.mlp):
            out = dense_layer(out)
        return torch.sigmoid(out)