import torch
import torch.nn as nn
import torch.nn.functional as F
from model import layers

class MLP(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, enable_bias, K, droprate):
        super(MLP, self).__init__()
        modules = []
        self.K = K
        modules.append(nn.Linear(in_features=n_feat, out_features=n_hid, bias=enable_bias))
        for k in range(1, K-1):
            modules.append(nn.Linear(in_features=n_hid, out_features=n_hid, bias=enable_bias))
        modules.append(nn.Linear(in_features=n_hid, out_features=n_class, bias=enable_bias))
        self.linear_layers = nn.Sequential(*modules)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        for k in range(self.K-1):
            x = self.linear_layers[k](x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.linear_layers[-1](x)
        x = self.log_softmax(x)

        return x

class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, enable_bias, K, droprate):
        super(GCN, self).__init__()
        self.graph_convs = nn.ModuleList()
        self.K = K
        self.graph_convs.append(layers.GraphConv(in_features=n_feat, out_features=n_hid, bias=enable_bias))
        for k in range(1, K-1):
            self.graph_convs.append(layers.GraphConv(in_features=n_hid, out_features=n_hid, bias=enable_bias))
        self.graph_convs.append(layers.GraphConv(in_features=n_hid, out_features=n_class, bias=enable_bias))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, filter):
        for k in range(self.K-1):
            x = self.graph_convs[k](x, filter)
            x = self.relu(x)
        x = self.dropout(x)
        x = self.graph_convs[-1](x, filter)
        x = self.log_softmax(x)

        return x