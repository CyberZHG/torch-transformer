import torch
import torch.nn as nn
import torch.nn.functional as F
import keras
import numpy as np
from unittest import TestCase
from keras_transformer import get_decoder_component
from torch_transformer import DecoderComponent


class TestDecoderComponent(TestCase):

    @staticmethod
    def torch_to_numpy(x):
        return torch.from_numpy(x).type(torch.get_default_dtype())

    def gen_torch_net(self, in_dim, hidden_dim, head_num,
                      swq, sbq, swk, sbk, swv, sbv, swo, sbo,
                      wq, bq, wk, bk, wv, bv, wo, bo,
                      fwh, fbh, fwo, fbo):
        net = DecoderComponent(
            in_features=in_dim,
            hidden_features=hidden_dim,
            head_num=head_num,
            attention_activation=None,
            feed_forward_activation=F.relu
        )
        net.self_attention.layer.linear_q.weight = nn.Parameter(self.torch_to_numpy(swq).transpose(1, 0))
        net.self_attention.layer.linear_q.bias = nn.Parameter(self.torch_to_numpy(sbq))
        net.self_attention.layer.linear_k.weight = nn.Parameter(self.torch_to_numpy(swk).transpose(1, 0))
        net.self_attention.layer.linear_k.bias = nn.Parameter(self.torch_to_numpy(sbk))
        net.self_attention.layer.linear_v.weight = nn.Parameter(self.torch_to_numpy(swv).transpose(1, 0))
        net.self_attention.layer.linear_v.bias = nn.Parameter(self.torch_to_numpy(sbv))
        net.self_attention.layer.linear_o.weight = nn.Parameter(self.torch_to_numpy(swo).transpose(1, 0))
        net.self_attention.layer.linear_o.bias = nn.Parameter(self.torch_to_numpy(sbo))

        net.attention.layer.linear_q.weight = nn.Parameter(self.torch_to_numpy(wq).transpose(1, 0))
        net.attention.layer.linear_q.bias = nn.Parameter(self.torch_to_numpy(bq))
        net.attention.layer.linear_k.weight = nn.Parameter(self.torch_to_numpy(wk).transpose(1, 0))
        net.attention.layer.linear_k.bias = nn.Parameter(self.torch_to_numpy(bk))
        net.attention.layer.linear_v.weight = nn.Parameter(self.torch_to_numpy(wv).transpose(1, 0))
        net.attention.layer.linear_v.bias = nn.Parameter(self.torch_to_numpy(bv))
        net.attention.layer.linear_o.weight = nn.Parameter(self.torch_to_numpy(wo).transpose(1, 0))
        net.attention.layer.linear_o.bias = nn.Parameter(self.torch_to_numpy(bo))

        net.feed_forward.layer.linear_h.weight = nn.Parameter(self.torch_to_numpy(fwh).transpose(1, 0))
        net.feed_forward.layer.linear_h.bias = nn.Parameter(self.torch_to_numpy(fbh))
        net.feed_forward.layer.linear_o.weight = nn.Parameter(self.torch_to_numpy(fwo).transpose(1, 0))
        net.feed_forward.layer.linear_o.bias = nn.Parameter(self.torch_to_numpy(fbo))
        return net

    @staticmethod
    def gen_keras_net(in_dim, hidden_dim, head_num,
                      swq, sbq, swk, sbk, swv, sbv, swo, sbo,
                      wq, bq, wk, bk, wv, bv, wo, bo,
                      fwh, fbh, fwo, fbo):
        input_layer = keras.layers.Input(shape=(None, in_dim))
        encoded_layer = keras.layers.Input(shape=(None, in_dim))
        output_layer = get_decoder_component(
            'Encoder',
            input_layer=input_layer,
            encoded_layer=encoded_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=None,
            feed_forward_activation='relu'
        )
        model = keras.models.Model(inputs=[input_layer, encoded_layer], outputs=output_layer)
        model.summary()
        model.compile(optimizer='adam', loss='mse')
        model.get_layer(name='Encoder-MultiHeadSelfAttention').set_weights([swq, sbq, swk, sbk, swv, sbv, swo, sbo])
        model.get_layer(name='Encoder-MultiHeadQueryAttention').set_weights([wq, bq, wk, bk, wv, bv, wo, bo])
        model.get_layer(name='Encoder-FeedForward').set_weights([fwh, fbh, fwo, fbo])
        return model

    def test_same(self):
        batch_size = np.random.randint(1, 4)
        seq_len = np.random.randint(1, 10)
        in_dim, hidden_dim, head_num = 16, 64, 4

        swq = np.random.standard_normal((in_dim, in_dim))
        sbq = np.random.standard_normal((in_dim,))
        swk = np.random.standard_normal((in_dim, in_dim))
        sbk = np.random.standard_normal((in_dim,))
        swv = np.random.standard_normal((in_dim, in_dim))
        sbv = np.random.standard_normal((in_dim,))
        swo = np.random.standard_normal((in_dim, in_dim))
        sbo = np.random.standard_normal((in_dim,))

        wq = np.random.standard_normal((in_dim, in_dim))
        bq = np.random.standard_normal((in_dim,))
        wk = np.random.standard_normal((in_dim, in_dim))
        bk = np.random.standard_normal((in_dim,))
        wv = np.random.standard_normal((in_dim, in_dim))
        bv = np.random.standard_normal((in_dim,))
        wo = np.random.standard_normal((in_dim, in_dim))
        bo = np.random.standard_normal((in_dim,))

        fwh = np.random.standard_normal((in_dim, hidden_dim))
        fbh = np.random.standard_normal((hidden_dim,))
        fwo = np.random.standard_normal((hidden_dim, in_dim))
        fbo = np.random.standard_normal((in_dim,))

        x = np.random.standard_normal((batch_size, seq_len, in_dim))
        encoded = np.random.standard_normal((batch_size, seq_len, in_dim))
        torch_net = self.gen_torch_net(in_dim, hidden_dim, 4,
                                       swq, sbq, swk, sbk, swv, sbv, swo, sbo,
                                       wq, bq, wk, bk, wv, bv, wo, bo,
                                       fwh, fbh, fwo, fbo)
        keras_net = self.gen_keras_net(in_dim, hidden_dim, 4,
                                       swq, sbq, swk, sbk, swv, sbv, swo, sbo,
                                       wq, bq, wk, bk, wv, bv, wo, bo,
                                       fwh, fbh, fwo, fbo)
        print(torch_net)
        torch_y = torch_net(self.torch_to_numpy(x), self.torch_to_numpy(encoded)).detach().numpy()
        keras_y = keras_net.predict([x, encoded])
        self.assertTrue(np.allclose(keras_y, torch_y, rtol=0.0, atol=1e-4), (keras_y, torch_y))
