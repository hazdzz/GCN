import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor

class GraphConv(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features, out_features, bias):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, filter):
        if x.is_sparse:
            first_mul = torch.sparse.mm(x, self.weight)
        else:
            first_mul = torch.mm(x, self.weight)
        if filter.is_sparse:
            second_mul = torch.sparse.mm(filter, first_mul)
        else:
            second_mul = torch.mm(filter, first_mul)
        
        if self.bias is not None:
            graph_conv = torch.add(input=second_mul, other=self.bias, alpha=1)
        else:
            graph_conv = second_mul
        
        return graph_conv

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )