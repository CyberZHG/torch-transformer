import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FeedForward']


class FeedForward(nn.Module):
    """Position-wise feed forward"""

    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 bias=True,
                 activation=F.relu):
        super(FeedForward, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.activation = activation
        self.linear_h = nn.Linear(in_features, hidden_features, bias)
        self.linear_o = nn.Linear(hidden_features, out_features, bias)

    def forward(self, x):
        h = self.linear_h(x)
        if self.activation is not None:
            h = self.activation(h)
        return self.linear_o(h)

    def extra_repr(self):
        return 'in_features={}, hidden_feature, out_features={}, bias={}, activation={}'.format(
            self.in_features, self.hidden_features, self.out_features, self.bias, self.activation
        )
