import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_position_embedding import PositionEmbedding
from torch_layer_normalization import LayerNormalization
from torch_multi_head_attention import MultiHeadAttention
from torch_embed_sim import EmbeddingSim
from .feed_forward import FeedForward


__all__ = [
    'BlockWrapper', 'EncoderComponent', 'DecoderComponent', 'Encoder', 'Decoder', 'EncoderDecoder',
]


class BlockWrapper(nn.Module):

    def __init__(self,
                 in_features,
                 layer,
                 dropout_rate=0.0):
        super(BlockWrapper, self).__init__()
        self.in_features = in_features
        self.layer = layer
        self.dropout_layer = None
        if dropout_rate > 0:
            self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.normal_layer = LayerNormalization(normal_shape=in_features)

    def forward(self, x, *args, **kwargs):
        y = self.layer(x, *args, **kwargs)
        if self.dropout_layer is not None:
            y = self.dropout_layer(y)
        return self.normal_layer(x + y)


class EncoderComponent(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 head_num,
                 attention_activation=None,
                 feed_forward_activation=F.relu,
                 dropout_rate=0.0):
        super(EncoderComponent, self).__init__()
        self.attention = BlockWrapper(
            in_features,
            layer=MultiHeadAttention(
                in_features=in_features,
                head_num=head_num,
                activation=attention_activation,
            ),
            dropout_rate=dropout_rate,
        )
        self.feed_forward = BlockWrapper(
            in_features,
            layer=FeedForward(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=in_features,
                activation=feed_forward_activation,
            ),
            dropout_rate=dropout_rate,
        )

    def forward(self, x, mask):
        return self.feed_forward(self.attention(x, x, x, mask=mask))


class DecoderComponent(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 head_num,
                 attention_activation=None,
                 feed_forward_activation=F.relu,
                 dropout_rate=0.0):
        super(DecoderComponent, self).__init__()
        self.self_attention = BlockWrapper(
            in_features,
            layer=MultiHeadAttention(
                in_features=in_features,
                head_num=head_num,
                activation=attention_activation,
            ),
            dropout_rate=dropout_rate,
        )
        self.attention = BlockWrapper(
            in_features,
            layer=MultiHeadAttention(
                in_features=in_features,
                head_num=head_num,
                activation=attention_activation,
            ),
            dropout_rate=dropout_rate,
        )
        self.feed_forward = BlockWrapper(
            in_features,
            layer=FeedForward(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=in_features,
                activation=feed_forward_activation,
            ),
            dropout_rate=dropout_rate,
        )

    def forward(self, x, encoded, mask):
        y = self.self_attention(x, x, x, mask=MultiHeadAttention.gen_history_mask(x))
        y = self.attention(y, encoded, encoded, mask=mask)
        return self.feed_forward(y)


class Encoder(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 encoder_num,
                 head_num,
                 attention_activation=None,
                 feed_forward_activation=F.relu,
                 dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.components = []
        for i in range(encoder_num):
            component = EncoderComponent(
                in_features=in_features,
                hidden_features=hidden_features,
                head_num=head_num,
                attention_activation=attention_activation,
                feed_forward_activation=feed_forward_activation,
                dropout_rate=dropout_rate,
            )
            self.add_module('encoder_%d' % i, component)
            self.components.append(component)

    def forward(self, x, mask):
        for component in self.components:
            x = component(x, mask=mask)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 decoder_num,
                 head_num,
                 attention_activation=None,
                 feed_forward_activation=F.relu,
                 dropout_rate=0.0):
        super(Decoder, self).__init__()
        self.components = []
        for i in range(decoder_num):
            component = DecoderComponent(
                in_features=in_features,
                hidden_features=hidden_features,
                head_num=head_num,
                attention_activation=attention_activation,
                feed_forward_activation=feed_forward_activation,
                dropout_rate=dropout_rate,
            )
            self.add_module('decoder_%d' % i, component)
            self.components.append(component)

    def forward(self, x, encoded, mask):
        for component in self.components:
            x = component(x, encoded=encoded, mask=mask)
        return x


class EncoderDecoder(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 encoder_num,
                 decoder_num,
                 head_num,
                 attention_activation=None,
                 feed_forward_activation=F.relu,
                 dropout_rate=0.0):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(
            in_features=in_features,
            hidden_features=hidden_features,
            encoder_num=encoder_num,
            head_num=head_num,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
        )
        self.decoder = Decoder(
            in_features=in_features,
            hidden_features=hidden_features,
            decoder_num=decoder_num,
            head_num=head_num,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
        )

    def forward(self, encoder_input, encoder_mask, decoder_input):
        encoded = self.encoder(encoder_input, encoder_mask)
        return self.decoder(decoder_input, encoded, encoder_mask)
