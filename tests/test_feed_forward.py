import torch
import torch.nn as nn
import torch.nn.functional as F
import keras
import numpy as np
from unittest import TestCase
from keras_position_wise_feed_forward import FeedForward as KerasFeedForward
from torch_transformer import FeedForward


class TestFeedForward(TestCase):

    @staticmethod
    def gen_torch_net(in_dim, hidden_dim, bias, activation, w1, b1, w2, b2):
        net = FeedForward(
            in_features=in_dim,
            hidden_features=hidden_dim,
            out_features=in_dim,
            bias=bias,
            activation=activation,
        )
        net.linear_h.weight = nn.Parameter(torch.from_numpy(w1).transpose(1, 0))
        net.linear_h.bias = nn.Parameter(torch.from_numpy(b1))
        net.linear_o.weight = nn.Parameter(torch.from_numpy(w2).transpose(1, 0))
        net.linear_o.bias = nn.Parameter(torch.from_numpy(b2))
        return net

    @staticmethod
    def gen_keras_net(in_dim, hidden_dim, bias, activation, w1, b1, w2, b2):
        input_layer = keras.layers.Input(shape=(None, in_dim))
        output_layer = KerasFeedForward(
            units=hidden_dim,
            activation=activation,
            use_bias=bias,
            weights=[w1, b1, w2, b2],
        )(input_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        return model

    def test_same(self):
        batch_size = np.random.randint(1, 4)
        seq_len = np.random.randint(1, 10)
        in_dim = np.random.randint(1, 5)
        hidden_dim = np.random.randint(1, 8)
        w1 = np.random.standard_normal((in_dim, hidden_dim))
        b1 = np.random.standard_normal((hidden_dim,))
        w2 = np.random.standard_normal((hidden_dim, in_dim))
        b2 = np.random.standard_normal((in_dim,))
        x = np.random.standard_normal((batch_size, seq_len, in_dim))

        torch_net = self.gen_torch_net(in_dim, hidden_dim, True, F.relu, w1, b1, w2, b2)
        keras_net = self.gen_keras_net(in_dim, hidden_dim, True, 'relu', w1, b1, w2, b2)
        print(torch_net)
        torch_y = torch_net(torch.from_numpy(x)).detach().numpy()
        keras_y = keras_net.predict(x)
        self.assertTrue(np.allclose(keras_y, torch_y))
